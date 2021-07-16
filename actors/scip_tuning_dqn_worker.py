""" Worker class copied and modified from https://github.com/cyoon1729/distributedRL """
import pyarrow as pa
import zmq
from sklearn.metrics import f1_score
from pyscipopt import Sepa, SCIP_RESULT
from time import time
import numpy as np
from warnings import warn
import utils.scip_models
from utils.data import Transition, get_data_memory
from utils.misc import get_img_from_fig
from utils.event_hdlrs import DebugEvents, BranchingEventHdlr
import os
import math
import random
from gnn.models import SCIPTuningQnet
import torch
import scipy as sp
import torch.optim as optim
from torch_scatter import scatter_mean, scatter_max, scatter_add
from utils.functions import get_normalized_areas, truncate
from collections import namedtuple
import matplotlib as mpl
import pickle
from utils.scip_models import maxcut_mccormic_model, mvc_model, set_aggresive_separation, CSResetSepa
from copy import deepcopy
mpl.rc('figure', max_open_warning=0)
import matplotlib.pyplot as plt
import wandb


StateActionContext = namedtuple('StateActionQValuesContext', ('scip_state', 'action', 'q_values', 'transformer_context'))


class SCIPTuningDQNWorker(Sepa):
    def __init__(self,
                 worker_id,
                 hparams,
                 use_gpu=False,
                 gpu_id=None,
                 **kwargs
                 ):
        """
        Sample scip.Model state every time self.sepaexeclp is invoked.
        Store the generated data object in
        """
        super(SCIPTuningDQNWorker, self).__init__()
        self.name = 'SCIP Tuning DQN Worker'
        self.hparams = hparams

        # learning stuff
        cuda_id = 'cuda' if gpu_id is None else f'cuda:{gpu_id}'
        self.device = torch.device(cuda_id if use_gpu and torch.cuda.is_available() else "cpu")
        self.batch_size = hparams.get('batch_size', 64)
        self.gamma = hparams.get('gamma', 0.999)
        self.eps_start = hparams.get('eps_start', 0.9)
        self.eps_end = hparams.get('eps_end', 0.05)
        self.eps_decay = hparams.get('eps_decay', 200)
        self.policy_net = SCIPTuningQnet(hparams, use_gpu, gpu_id).to(self.device)
        # value aggregation method for the target Q values
        if hparams.get('value_aggr', 'mean') == 'max':
            self.value_aggr = scatter_max
        elif hparams.get('value_aggr', 'mean') == 'mean':
            self.value_aggr = scatter_mean
        self.nstep_learning = hparams.get('nstep_learning', 1)
        self.dqn_objective = hparams.get('dqn_objective', 'db_auc')
        self.use_per = True

        # training stuff
        self.num_env_steps_done = 0
        self.num_sgd_steps_done = 0
        self.num_param_updates = 0
        self.i_episode = 0
        self.training = True
        self.walltime_offset = 0
        self.start_time = time()
        self.last_time_sec = self.walltime_offset
        self.datasets = None
        self.trainset = None
        self.graph_indices = None
        self.cur_graph = None

        # instance specific data needed to be reset every episode
        # todo unifiy x and y to x only (common for all combinatorial problems)
        self.G = None
        self.x = None
        self.instance_info = None
        self.scip_seed = None
        self.action = None
        self.prev_action = None
        self.prev_state = None
        self.episode_history = []
        self.episode_stats = {
            'ncuts': [],
            'ncuts_applied': [],
            'solving_time': [],
            'processed_nodes': [],
            'gap': [],
            'lp_rounds': [],
            'lp_iterations': [],
            'dualbound': []
        }
        self.stats_updated = False
        self.cut_generator = None
        self.dataset_name = 'trainset'  # or <easy/medium/hard>_<validset/testset>
        self.lp_iterations_limit = -1
        self.terminal_state = False
        self.setting = 'root_only'
        self.run_times = []

        # debugging stats
        self.training_n_random_actions = 0
        self.training_n_actions = 0

        # file system paths
        self.run_dir = hparams['run_dir']
        self.checkpoint_filepath = os.path.join(self.run_dir, 'learner_checkpoint.pt')
        # training logs
        self.training_stats = {'db_auc': [], 'db_auc_improvement': [], 'gap_auc': [], 'gap_auc_improvement': [], 'active_applied_ratio': [], 'applied_available_ratio': [], 'accuracy': [], 'f1_score': [], 'jaccard_similarity': []}
        self.last_training_episode_stats = {}
        # tmp buffer for holding cutting planes statistics
        self.sepa_stats = None

        # debug todo remove when finished
        self.debug_n_tracking_errors = 0
        self.debug_n_early_stop = 0
        self.debug_n_episodes_done = 0
        self.debug_n_buggy_episodes = 0

        # assign the validation instances according to worker_id and num_workers:
        # flatten all instances to a list of tuples of (dataset_name, inst_idx, seed_idx)
        datasets = hparams['datasets']
        flat_instances = []
        for dataset_name, dataset in datasets.items():
            if 'train' in dataset_name or 'test' in dataset_name:
                continue
            if hparams['overfit'] and dataset_name not in hparams['overfit']:
                continue
            for inst_idx in range(dataset['ngraphs']):
                for scip_seed in dataset['scip_seed']:
                    flat_instances.append((dataset_name, inst_idx, scip_seed))
        idx = worker_id-1
        self.eval_instances = []
        while idx < len(flat_instances):
            self.eval_instances.append(flat_instances[idx])
            idx += hparams['num_workers']

        # distributed system stuff
        self.worker_id = worker_id
        self.generate_demonstration_data = False
        self.print_prefix = f'[Worker {self.worker_id}] '
        # initialize zmq sockets
        # use socket.connect() instead of .bind() because workers are the least stable part in the system
        # (not supposed to but rather suspected to be)
        print(self.print_prefix, "initializing sockets..")
        # for receiving params from learner and requests from replay server
        context = zmq.Context()
        self.send_2_apex_socket = context.socket(zmq.PUSH)  # for sending logs
        self.sub_socket = context.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all topics
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)  # keep only last message received

        # connect to the main apex process
        self.send_2_apex_socket.connect(f'tcp://127.0.0.1:{hparams["com"]["apex_port"]}')
        self.print(f'connecting to apex_port: {hparams["com"]["apex_port"]}')

        # connect to learner pub socket
        self.sub_socket.connect(f'tcp://127.0.0.1:{hparams["com"]["learner_2_workers_pubsub_port"]}')
        self.print(f'connecting to learner_2_workers_pubsub_port: {hparams["com"]["learner_2_workers_pubsub_port"]}')

        # connect to replay_server pub socket
        self.sub_socket.connect(f'tcp://127.0.0.1:{hparams["com"]["replay_server_2_workers_pubsub_port"]}')
        self.print(f'connecting to replay_server_2_workers_pubsub_port: {hparams["com"]["replay_server_2_workers_pubsub_port"]}')


        # for sending replay data to buffer
        context = zmq.Context()
        self.worker_2_replay_server_socket = context.socket(zmq.PUSH)
        self.worker_2_replay_server_socket.connect(f'tcp://127.0.0.1:{hparams["com"]["workers_2_replay_server_port"]}')
        self.print(f'connecting to workers_2_replay_server_port: {hparams["com"]["workers_2_replay_server_port"]}')

        # save pid to run_dir
        pid = os.getpid()
        pid_file = os.path.join(hparams["run_dir"], f'{self.actor_name}_pid.txt')
        self.print(f'saving pid {pid} to {pid_file}')
        with open(pid_file, 'w') as f:
            f.writelines(str(pid) + '\n')

    @property
    def actor_name(self):
        return f"worker_{self.worker_id}"

    def synchronize_params(self, new_params_packet):
        """Synchronize worker's policy_net with learner's policy_net params """
        new_params, params_id = new_params_packet
        model = self.policy_net
        for param, new_param in zip(model.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)
        # synchronize the global step counter self.num_param_updates with the value arrived from learner.
        # this makes self.log_stats() robust to Worker failures, missed packets and in resumed training.
        assert self.num_param_updates < params_id, f"global step counter is not consistent between learner and worker: TestWorker.num_param_updates={self.num_param_updates}, ParamsID={params_id}"
        self.num_param_updates = params_id
        # test should evaluate model here and then log stats.
        # workers should log stats before synchronizing, to plot the statistics collected by the previous policy,
        # together with the previous policy's params_id.

    def send_replay_data(self, replay_data):
        replay_data_packet = self.pack_replay_data(replay_data)
        self.worker_2_replay_server_socket.send(replay_data_packet)

    def read_message(self, message):
        new_params_packet = None
        message = pa.deserialize(message)
        if message[0] == 'new_params':
            new_params_packet = message[1]
        elif message[0] == 'generate_demonstration_data':
            raise ValueError('scip tuning does not support demonstrations')
        elif message[0] == 'generate_agent_data':
            self.generate_demonstration_data = False
            print(self.print_prefix, 'collecting agent data')
        else:
            raise ValueError
        return new_params_packet

    def recv_messages(self, wait_for_new_params=False):
        """
        Subscribes to learner and replay_server messages.
        if topic == 'new_params' update model and return received_new_params.
           topic == 'generate_demonstration_data' set self.generate_demonstration_data True
           topic == 'generate_egent_data' set self.generate_demonstration_data False
        """
        new_params_packet = None
        # if wait_for_new_params:
        #     while new_params_packet is None:
        #         message = self.sub_socket.recv()
        #         new_params_packet = self.read_message(message)
        # else:
        try:
            while new_params_packet is None:
                if wait_for_new_params:
                    message = self.sub_socket.recv()
                else:
                    message = self.sub_socket.recv(zmq.DONTWAIT)
                new_params_packet = self.read_message(message)
        except zmq.Again:
            # no packets are waiting
            pass

        if new_params_packet is not None:
            self.synchronize_params(new_params_packet)
            received_new_params = True
        else:
            received_new_params = False
        return received_new_params

    def run(self):
        self.initialize_training()
        self.load_datasets()
        while True:
            received_new_params = self.recv_messages()
            if received_new_params:
                self.evaluate_and_send_logs()

            replay_data = self.collect_data()
            self.send_replay_data(replay_data)

    def evaluate_and_send_logs(self):
        self.print(f'evaluating param id = {self.num_param_updates}')
        global_step, validation_stats = self.evaluate()
        log_packet = ('log', f'worker_{self.worker_id}', global_step,
                      ([(k, v) for k, v in self.training_stats.items()],
                       validation_stats,
                       [(f'worker_{self.worker_id}_exploration', self.training_n_random_actions/self.training_n_actions), (f'worker_{self.worker_id}_env_steps', self.num_env_steps_done), (f'worker_{self.worker_id}_episodes_done', self.i_episode)],
                       [(k, v) for k, v in self.last_training_episode_stats.items()]))
        log_packet = pa.serialize(log_packet).to_buffer()
        self.send_2_apex_socket.send(log_packet)
        # reset training stats for the next round
        for k in self.training_stats.keys():
            self.training_stats[k] = []
        self.training_n_actions = 0
        self.training_n_random_actions = 0

    def collect_data(self):
        """ Fill local buffer until some stopping criterion is satisfied """
        self.set_training_mode()
        local_buffer = []
        trainset = self.trainset
        while len(local_buffer) < self.hparams.get('local_buffer_size'):
            # sample graph randomly
            graph_idx = self.graph_indices[(self.i_episode + 1) % len(self.graph_indices)]
            G, instance_info = trainset['instances'][graph_idx]
            if self.hparams.get('overfit', False):
                lp_iter_limit = trainset['overfit_lp_iter_limits'][graph_idx]
            else:
                lp_iter_limit = trainset['lp_iterations_limit']
            # fix training scip_seed for debug purpose
            if self.hparams['fix_training_scip_seed']:
                scip_seed = self.hparams['fix_training_scip_seed']
            else:
                # set random scip seed
                scip_seed = np.random.randint(1000000000)
            self.cur_graph = f'trainset graph {graph_idx} seed {scip_seed}'
            # execute episodes, collect experience and append to local_buffer
            trajectory, _ = self.execute_episode(G, instance_info, lp_iter_limit,
                                                 dataset_name=trainset['dataset_name'])

            local_buffer += trajectory
            if self.i_episode + 1 % len(self.graph_indices) == 0:
                self.graph_indices = torch.randperm(trainset['num_instances'])
        return local_buffer

    @staticmethod
    def pack_replay_data(replay_data):
        """
        Convert a list of (Transition, initial_weights) to list of (TransitionNumpyTuple, initial_priorities.numpy())
        :param replay_data: list of (Transition, float initial_priority)
        :return:
        """
        replay_data_packet = []
        for transition, initial_priority, is_demonstration in replay_data:
            size_gbyte = get_data_memory(transition, units='G')
            replay_data_packet.append((transition.to_numpy_tuple(), initial_priority, is_demonstration, size_gbyte))
        replay_data_packet = pa.serialize(replay_data_packet).to_buffer()
        return replay_data_packet

    # done
    def init_episode(self, G, x, lp_iterations_limit, cut_generator=None, instance_info=None, dataset_name='trainset25', scip_seed=None, setting='root_only'):
        self.G = G
        self.x = x
        # self.y = y
        self.instance_info = instance_info
        self.scip_seed = scip_seed
        self.action = None
        self.prev_action = None
        self.prev_state = None
        self.episode_history = []
        self.episode_stats = {
            'ncuts': [],
            'ncuts_applied': [],
            'solving_time': [],
            'processed_nodes': [],
            'gap': [],
            'lp_rounds': [],
            'lp_iterations': [],
            'dualbound': []
        }
        self.stats_updated = False
        self.cut_generator = cut_generator
        self.dataset_name = dataset_name
        self.lp_iterations_limit = lp_iterations_limit
        self.terminal_state = False
        self.setting = 'root_only'
        self.run_times = []

    # done
    def sepaexeclp(self):
        t0 = time()
        if self.hparams.get('debug_events', False):
            self.print('DEBUG MSG: cut_selection_dqn separator called')

        # finish with the previous step:
        # todo - in case of no cuts, we return here a second time without any new action. we shouldn't record stats twice.
        self._update_episode_stats()

        # if for some reason we terminated the episode (lp iterations limit reached / empty action etc.
        # we dont want to run any further cut_selection_dqn steps, and therefore we return immediately.
        if self.terminal_state:
            # discard all the cuts in the separation storage and return
            self.model.clearCuts()
            self.model.interruptSolve()
            result = {"result": SCIP_RESULT.DIDNOTRUN}

        elif self.model.getNLPIterations() < self.lp_iterations_limit:
            result = self._do_dqn_step()

        else:
            # stop optimization (implicitly), and don't add any more cuts
            if self.hparams.get('verbose', 0) == 2:
                self.print('LP_ITERATIONS_LIMIT reached. DIDNOTRUN!')
            self.terminal_state = 'LP_ITERATIONS_LIMIT_REACHED'
            # get stats of prev_action
            self.model.getState(query=self.prev_action)
            # clear cuts and terminate
            self.model.clearCuts()
            self.model.interruptSolve()
            result = {"result": SCIP_RESULT.DIDNOTRUN}

        # todo - what retcode should be returned here?
        #  currently: if selected cuts              -> SEPARATED
        #                discarded all or no cuts   -> DIDNOTFIND
        #                otherwise                  -> DIDNOTRUN
        self.run_times.append(time() - t0)
        return result

    # done
    def _do_dqn_step(self):
        """
        Here is the episode inner loop (DQN)
        We sequentially
        1. get state
        2. select action
        3. get the next state and stats for computing reward (in the next LP round, after the LP solver solved for our cuts)
        4. store transition in memory
        Offline, we optimize the policy on the replay data.
        When the instance is solved, the episode ends, and we start solving another instance,
        continuing with the latest policy parameters.
        This DQN agent should only be included as a separator in the next instance SCIP model.
        """
        info = {}
        # get the current state, a dictionary of available cuts (keyed by their names,
        # and query statistics related to the previous action (cut activeness etc.)
        cur_state, available_cuts = self.model.getState(state_format='tensor', get_available_cuts=True, query=self.prev_action)
        info['state_info'], info['action_info'] = cur_state, available_cuts

        # if there are available cuts, select action and continue to the next state
        if available_cuts['ncuts'] > 0:
            # select an action, and get q_values for PER
            assert not np.any(np.isnan(cur_state['C'])) and not np.any(np.isnan(cur_state['A'])), f'Nan values in state features\ncur_graph = {self.cur_graph}\nA = {cur_state["A"]}\nC = {cur_state["C"]}'
            action_info = self._select_action(cur_state)
            for k, v in action_info.items():
                info[k] = v

            # prob what the scip cut selection algorithm would do in this state given the deafult separating parameters
            self.reset_separating_params()
            cut_names_selected_by_scip = self.prob_scip_cut_selection()
            available_cuts['selected_by_scip'] = np.array([cut_name in cut_names_selected_by_scip for cut_name in available_cuts['cuts'].keys()])

            # apply the action
            selected_params = {k: self.hparams['action_set'][k][idx] for k, idx in action_info['selected'].items()}
            self.reset_separating_params(selected_params)
            cut_names_selected_by_agent = self.prob_scip_cut_selection()
            available_cuts['selected_by_agent'] = np.array([cut_name in cut_names_selected_by_agent for cut_name in available_cuts['cuts'].keys()])
            result = {"result": SCIP_RESULT.DIDNOTRUN}

            # SCIP will execute the action,
            # and return here in the next LP round -
            # unless the instance is solved and the episode is done.
            self.stats_updated = False  # mark false to record relevant stats after this action will make effect

            if self.setting == 'branch_and_cut':
                # return here and do not save episode history for saving memory
                return result

            # store the current state and action for
            # computing later the n-step rewards and the (s,a,r',s') transitions
            self.episode_history.append(info)
            self.prev_action = available_cuts
            self.prev_state = cur_state

        # If there are no available cuts we simply ignore this round.
        # The stats related to the previous action are already collected, and we are updated.
        # We don't store the current state-action pair, because there is no action at all, and the state is useless.
        # There is a case that after SCIP will apply heuristics we will return here again with new cuts,
        # and in this case the state will be different anyway.
        # If SCIP will decide to branch, we don't care, it is not related to us, and we won't consider improvements
        # in the dual bound caused by the branching.
        # The current gap can be either zero (OPTIMAL) or strictly positive.
        # model.getGap() can return gap > 0 even if the dual bound is optimal,
        # because SCIP stats will be updated only afterward.
        # So we temporarily set terminal_state to True (general description)
        # and we will accurately characterize it after the optimization terminates.
        elif available_cuts['ncuts'] == 0:
            self.prev_action = None
            # self.terminal_state = True
            # self.finished_episode_stats = True
            result = {"result": SCIP_RESULT.DIDNOTFIND}

        return result

    def reset_separating_params(self, params={}):
        self.model.setRealParam('separating/objparalfac', params.get('objparalfac', 0.1))
        self.model.setRealParam('separating/dircutoffdistfac', params.get('dircutoffdistfac', 0.5))
        self.model.setRealParam('separating/efficacyfac', params.get('efficacyfac', 1.))
        self.model.setRealParam('separating/intsupportfac', params.get('intsupportfac', 0.1))
        self.model.setIntParam('separating/maxcutsroot', params.get('maxcutsroot', 2000))
        self.model.setRealParam('separating/minorthoroot', params.get('minorthoroot', 0.9))

    def prob_scip_cut_selection(self):
        available_cuts = self.model.getCuts()
        lp_iter = self.model.getNLPIterations()
        self.model.startProbing()
        for cut in available_cuts:
            self.model.addCut(cut)
        self.model.applyCutsProbing()
        cut_names = self.model.getSelectedCutsNames()
        self.model.endProbing()
        if self.model.getNLPIterations() != lp_iter:
            # todo - investigate why with scip_seed = 562696653 probing increments lp_iter by one.
            #  it seems not to make any damage, however.
            print('Warning! SCIP probing mode changed num lp iterations.')
        # assert self.model.getNLPIterations() == lp_iter
        return cut_names

    # done
    def _select_action(self, scip_state):
        # TODO - move all models to return dict with everything needed.
        # transform scip_state into GNN data type
        batch = Transition.create(scip_state, tqnet_version='none').as_batch().to(self.device)

        # if self.training:
        #     # take epsilon-greedy action
        #     sample = random.random()
        #     eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
        #                     math.exp(-1. * self.num_env_steps_done / self.eps_decay)
        #     self.num_env_steps_done += 1
        #
        #     if sample > eps_threshold:
        #         random_action = None
        #     else:
        #         # randomize action
        #         random_action = {k: torch.randint(low=0, high=len(vals), size=(1,)).cpu() for k, vals in self.hparams['action_set'].items()}
        #         self.training_n_random_actions += 1
        #     self.training_n_actions += 1
        # else:
        #     random_action = None

        # take greedy action
        with torch.no_grad():
            # todo - move all architectures to output dict format
            q_values = self.policy_net(
                x_c=batch.x_c,
                x_v=batch.x_v,
                x_a=batch.x_a,
                edge_index_c2v=batch.edge_index_c2v,
                edge_index_a2v=batch.edge_index_a2v,
                edge_attr_c2v=batch.edge_attr_c2v,
                edge_attr_a2v=batch.edge_attr_a2v,
                edge_index_a2a=batch.edge_index_a2a,
                edge_attr_a2a=batch.edge_attr_a2a,
                x_c_batch=batch.x_c_batch,
                x_v_batch=batch.x_v_batch,
                x_a_batch=batch.x_a_batch
            )

        # output: qvals, params
        output = {'q_values': q_values, 'selected': {}}

        # select e-greedy action for each action dimension
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.num_env_steps_done / self.eps_decay)
        self.num_env_steps_done += self.training
        for k, vals in q_values.items():
            # take epsilon-greedy action
            sample = random.random()
            if sample < eps_threshold and self.training:
                output['selected'][k] = torch.randint(low=0, high=len(vals), size=(1,)).cpu()
                self.training_n_random_actions += 1
            else:
                # greedy
                output['selected'][k] = torch.argmax(vals)
            self.training_n_actions += self.training
        # if random_action is not None:
        #     output['selected'] = random_action
        # else:
        #     output['selected'] = {k: torch.argmax(vals) for k, vals in q_values.items()}
        output['selected_q_values'] = torch.tensor([q_values[k][0][idx] for k, idx in output['selected'].items()])
        return output

    # done
    def finish_episode(self):
        """
        Compute rewards, push transitions into memory and log stats
        INFO:
        SCIP can terminate an episode (due to, say, node_limit or lp_iterations_limit)
        after executing the LP without calling DQN.
        In this case we need to compute by hand the tightness of the last action taken,
        because the solver allows to access the information only in SCIP_STAGE.SOLVING
        We distinguish between 4 types of terminal states:
        OPTIMAL:
            Gap == 0. If the LP_ITERATIONS_LIMIT has been reached, we interpolate the final db/gap at the limit.
            Otherwise, the final statistics are taken as is.
            In this case the agent is rewarded also for all SCIP's side effects (e.g. heuristics)
            which potentially helped in solving the instance.
        LP_ITERATIONS_LIMIT_REACHED:
            Gap >= 0.
            We snapshot the current db/gap and interpolate the final db/gap at the limit.
        DIDNOTFIND:
            Gap > 0, ncuts == 0.
            We save the db/gap of the last round before ncuts == 0 occured as the final values,
            and specifically we do not consider improvements caused by branching.
        EMPTY_ACTION (deprecated):
            Gap > 0, ncuts > 0, nselected == 0.
            The treatment is the same as DIDNOTFIND.

        In practice, this function is called after SCIP.optimize() terminates.
        self.terminal_state is set to None at the beginning, and once one of the 4 cases above is detected,
        self.terminal_state is set to the appropriate value.

        """
        self.debug_n_episodes_done += 1         # todo debug remove when finished

        # classify the terminal state
        if self.terminal_state == 'TRAINING_BUG':
            assert self.training
            self.debug_n_buggy_episodes += 1
            self.print(f'discarded {self.debug_n_buggy_episodes} episodes')
            return [], None

        if self.terminal_state == 'LP_ITERATIONS_LIMIT_REACHED':
            pass
        # elif self.terminal_state and self.model.getGap() == 0:
        # todo: detect branching. if branching occured and stats are not updated, this is invalid episode.
        #  if branching occured but stats are updated, don't do anything and continue.
        #  if no branching occured, update stats and continue.

        # elif self.branching_event.branching_occured:
        #     self.terminal_state = 'NODE_LIMIT'
        #     if self.hparams.get('discard_bad_experience', True) and self.training:
        #         return []
        #     # if updated

        elif self.model.getGap() == 0:
            # assert self.stats_updated
            if self.model.getStatus() != 'nodelimit':
                self.terminal_state = 'OPTIMAL'
                # todo: correct gap, dual bound and LP iterations records:
                #  model.getGap() returns incorrect value in the last (redundant) LP round when we collect the
                #  last transition stats, so it is probably > 0 at this point.
                #  The dualbound might also be incorrect. For example, if primal heuristics improved, and conflict analysis
                #  or other heuristics improved the dualbound to its optimal value. As a consequence,
                #  the dual bound recorded at the last separation round is not optimal.
                #  In this case, the final LP iter might also be greater than the last value recorded,
                #  If they haven't reached the limit, then we take them as is.
                #  Otherwise, we truncate the curve as usual.
                #  By convention, we override the last records with the final values,
                #  and truncate the curves if necessary
                self.episode_stats['gap'][-1] = self.model.getGap()
                self.episode_stats['dualbound'][-1] = self.model.getDualbound()
                self.episode_stats['lp_iterations'][-1] = self.model.getNLPIterations()  # todo - subtract probing mode lp_iters if any
                self.truncate_to_lp_iterations_limit()
            else:
                self.terminal_state = 'NODE_LIMIT'
        else:
        # elif self.terminal_state and self.model.getGap() > 0:
        #     self.terminal_state = 'DIDNOTFIND'
            # todo
            #  The episode terminated unexpectedly, generating a bad reward,
            #  so terminate and discard trajectory.
            #  possible reasons: no available cuts (cycle inequalities are not sufficient for maxcut with K5 minors)
            #  In training we discard the episode to avoid extremely bad rewards.
            #  In testing we process it as is.
            assert self.model.getStatus() == 'nodelimit'
            self.terminal_state = 'NODE_LIMIT'

        # discard episodes which terminated early without optimal solution, to avoid extremely bad rewards.
        if self.terminal_state == 'NODE_LIMIT' and self.hparams.get('discard_bad_experience', False) \
                and self.training and self.model.getNLPIterations() < 0.90 * self.lp_iterations_limit:
            # todo remove printing-  debug
            self.debug_n_early_stop += 1
            self.print(f'discarded early stop {self.debug_n_early_stop}/{self.debug_n_episodes_done}')
            return [], None

        assert self.terminal_state in ['OPTIMAL', 'LP_ITERATIONS_LIMIT_REACHED', 'NODE_LIMIT']
        # in a case SCIP terminated without calling the agent,
        # we need to restore some information:
        # the normalized slack of the applied cuts, the selection order in demonstration episodes,
        # and to update the episode stats with the latest SCIP stats.
        if self.prev_action is not None and self.prev_action.get('normalized_slack', None) is None:
            # update stats for the last step
            self._update_episode_stats()
            ncuts = self.prev_action['ncuts']

            # todo:
            #  In rare cases self.model.getSelectedCutsNames() returns an empty list, although there were cuts applied.
            #  In such a case, we cannot restore the selection order, and therefore demonstration episodes will be discarded.
            #  If it is not demonstration episode, we can just assume that the cuts applied were those who selected by the agent.

            # try to restore the applied cuts from sepastore->selectedcutsnames
            selected_cuts_names = self.model.getSelectedCutsNames()
            # if failed:
            if len(selected_cuts_names) == 0 and self.demonstration_episode and self.training:
                # cannot restore the selection order. discarding episode.
                self.debug_n_tracking_errors += 1
                self.print(f'discarded tracking error {self.debug_n_tracking_errors}/{self.debug_n_episodes_done} ({self.cur_graph})')
                return [], None
            elif len(selected_cuts_names) == 0 and not self.demonstration_episode:
                # assert that the number of cuts selected by the agent is the number of cuts applied in the last round
                assert len(self.episode_stats['ncuts_applied'])-1 == len(self.episode_history)
                assert self.episode_stats['ncuts_applied'][-1] - self.episode_stats['ncuts_applied'][-2] == \
                       sum(self.episode_history[-1]['action_info']['selected_by_agent'])
                selected_cuts_names = []
                for cut_idx, cut_name in enumerate(self.episode_history[-1]['action_info']['cuts'].keys()):
                    if self.episode_history[-1]['action_info']['selected_by_agent'][cut_idx]:
                        selected_cuts_names.append(cut_name)
                assert len(selected_cuts_names) > 0

            # now compute the normalized slack etc.
            for i, cut_name in enumerate(selected_cuts_names):
                self.prev_action['cuts'][cut_name]['applied'] = True
                self.prev_action['cuts'][cut_name]['selection_order'] = i
            applied = np.zeros((ncuts,), dtype=np.bool)
            selection_order = np.full_like(applied, fill_value=ncuts, dtype=np.long)
            for i, cut in enumerate(self.prev_action['cuts'].values()):
                if i == ncuts:
                    break
                applied[i] = cut['applied']
                selection_order[i] = cut['selection_order']
            self.prev_action['applied'] = applied
            self.prev_action['selection_order'] = np.argsort(selection_order)[:len(selected_cuts_names)]  # todo verify bug fix

            # assert that the action taken by agent was actually applied
            selected_cuts = self.prev_action['selected_by_scip'] if self.demonstration_episode else self.prev_action['selected_by_agent']
            # todo
            #  assert all(selected_cuts == self.prev_action['applied'])
            #  for some reason this assertion fails because of self.model.getSelectedCutsNames() returns empty list,
            #  although there was at least one cut. the selected cuts names are important only for demonstrations,
            #  so if we are in training and in a demonstration episode then we just return here.
            if not all(selected_cuts == self.prev_action['applied']):
                # something gone wrong.
                # assert len(selection_order) == 0  # this is the known reason for this problem
                if self.training and self.demonstration_episode:
                    # todo remove printing-  debug
                    self.debug_n_tracking_errors += 1
                    self.print(f'discarded tracking error {self.debug_n_tracking_errors}/{self.debug_n_episodes_done} ({self.cur_graph})')
                    return [], None

            assert self.terminal_state in ['OPTIMAL', 'LP_ITERATIONS_LIMIT_REACHED', 'NODE_LIMIT']
            nvars = self.model.getNVars()

            cuts_nnz_vals = self.prev_state['cut_nzrcoef']['vals']
            cuts_nnz_rowidxs = self.prev_state['cut_nzrcoef']['rowidxs']
            cuts_nnz_colidxs = self.prev_state['cut_nzrcoef']['colidxs']
            cuts_matrix = sp.sparse.coo_matrix((cuts_nnz_vals, (cuts_nnz_rowidxs, cuts_nnz_colidxs)), shape=[ncuts, nvars]).toarray()
            final_solution = self.model.getBestSol()
            sol_vector = np.array([self.model.getSolVal(final_solution, x_i) for x_i in self.x.values()])
            # # sol_vector += [self.model.getSolVal(final_solution, y_ij) for y_ij in self.y.values()]
            # sol_vector = np.array(sol_vector)
            # sol_vector = np.array([x_i.getLPSol() for x_i in self.x.values()])
            # rhs slack of all cuts added at the previous round (including the discarded cuts)
            # we normalize the slack by the coefficients norm, to avoid different penalty to two same cuts,
            # with only constant factor between them
            cuts_norm = np.linalg.norm(cuts_matrix, axis=1)
            rhs_slack = self.prev_action['rhss'] - cuts_matrix @ sol_vector  # todo what about the cst and norm?
            normalized_slack = rhs_slack / cuts_norm
            # assign tightness penalty only to the selected cuts.
            self.prev_action['normalized_slack'] = np.zeros_like(self.prev_action['selected_by_agent'], dtype=np.float32)
            self.prev_action['normalized_slack'][self.prev_action['selected_by_agent']] = normalized_slack[self.prev_action['selected_by_agent']]

        # compute rewards and other stats for the whole episode,
        # and if in training session, push transitions into memory
        trajectory, stats = self._compute_rewards_and_stats()
        # increase the number of episodes done
        if self.training:
            self.i_episode += 1

        return trajectory, stats

    # done
    def _compute_rewards_and_stats(self):
        """
        Compute reward and store (s,a,r,s') transitions in memory
        By the way, compute some stats for logging, e.g.
        1. dualbound auc,
        2. gap auc,
        3. nactive/napplied,
        4. napplied/navailable
        """
        lp_iterations_limit = self.lp_iterations_limit
        gap = self.episode_stats['gap']
        dualbound = self.episode_stats['dualbound']
        lp_iterations = self.episode_stats['lp_iterations']

        # compute the area under the curve:
        if len(dualbound) <= 2:
            if self.training:
                # this episode is not informative. too easy. optimal on the beginning.
                return [], None
            # print(self.episode_stats)
            # print(self.episode_history)

        # todo - consider squaring the dualbound/gap before computing the AUC.
        dualbound_area, db_slope, db_diff = get_normalized_areas(t=lp_iterations, ft=dualbound, t_support=lp_iterations_limit, reference=self.instance_info['optimal_value'], return_slope_and_diff=True)
        gap_area, gap_slope, gap_diff = get_normalized_areas(t=lp_iterations, ft=gap, t_support=lp_iterations_limit, reference=0, return_slope_and_diff=True)  # optimal gap is always 0
        if self.hparams['reward_func'] == 'db_auc':
            immediate_rewards = dualbound_area
        elif self.hparams['reward_func'] == 'gap_auc':
            immediate_rewards = gap_area
        elif self.hparams['reward_func'] == 'db_aucXslope':
            immediate_rewards = dualbound_area * db_slope
        elif self.hparams['reward_func'] == 'db_slopeXdiff':
            immediate_rewards = db_slope * db_diff
        else:
            raise NotImplementedError

        if self.hparams.get('square_reward', False):
            immediate_rewards = immediate_rewards ** 2  # todo verification

        trajectory = []
        if self.training:
            # compute n-step returns for each state-action pair (s_t, a_t)
            # and store a transition (s_t, a_t, r_t, s_{t+n}
            # todo - in learning from demonstrations we used to compute both 1-step and n-step returns.
            n_transitions = len(self.episode_history)
            n_steps = self.nstep_learning
            gammas = self.gamma**np.arange(n_steps).reshape(-1, 1)  # [1, gamma, gamma^2, ... gamma^{n-1}]
            indices = np.arange(n_steps).reshape(1, -1) + np.arange(n_transitions).reshape(-1, 1)  # indices of sliding windows
            # in case of n_steps > 1, pad objective_area with zeros only for avoiding overflow
            max_index = np.max(indices)
            if max_index >= len(immediate_rewards):
                immediate_rewards = np.pad(immediate_rewards, (0, max_index+1-len(immediate_rewards)), 'constant', constant_values=0)
            # take sliding windows of width n_step from objective_area
            n_step_rewards = immediate_rewards[indices]
            # compute returns
            # R[t] = r[t] + gamma * r[t+1] + ... + gamma^(n-1) * r[t+n-1]
            R = n_step_rewards @ gammas
            bootstrapping_q = []
            discarded = False
            # assign rewards and store transitions (s,a,r,s')
            for step, (step_info, reward) in enumerate(zip(self.episode_history, R)):
                state, action_info, q_values = step_info['state_info'], step_info['action_info'], step_info['selected_q_values']

                # get the next n-step state and q values. if the next state is terminal
                # return 0 as q_values (by convention)
                next_step_info = self.episode_history[step + n_steps] if step + n_steps < n_transitions else {}
                next_state = next_step_info.get('state_info', None)
                next_action_info = next_step_info.get('action_info', None)
                next_q_values = next_step_info.get('selected_q_values', None)

                normalized_slack = action_info['normalized_slack']
                # todo: verify with Aleks - consider slack < 1e-10 as zero
                approximately_zero = np.abs(normalized_slack) < self.hparams['slack_tol']
                normalized_slack[approximately_zero] = 0
                # assert (normalized_slack >= 0).all(), f'rhs slack variable is negative,{normalized_slack}'
                if (normalized_slack < 0).any():
                    self.print(f'Warning: encountered negative RHS slack variable.\nnormalized_slack: {normalized_slack}\ndiscarding the rest of the episode\ncur_graph = {self.cur_graph}')
                    discarded = True
                    break
                # same reward for each param selection
                selected_action = np.array(list(step_info['selected'].values()))
                # reward = np.full_like(selected_action, joint_reward, dtype=np.float32)

                # TODO CONTINUE FROM HERE
                transition = Transition.create(scip_state=state,
                                               action=selected_action,
                                               reward=reward,
                                               scip_next_state=next_state,
                                               tqnet_version='none')

                if self.use_per:
                    # todo - compute initial priority for PER based on the policy q_values.
                    #        compute the TD error for each action in the current state as we do in sgd_step,
                    #        and then take the norm of the resulting cut-wise TD-errors as the initial priority
                    # selected_action = torch.from_numpy(action_info['selected_by_agent']).unsqueeze(1).long()  # cut-wise action
                    # q_values = q_values.gather(1, selected_action)  # gathering is done now in _select_action
                    if next_q_values is None:
                        # next state is terminal, and its q_values are 0 by convention
                        target_q_values = torch.from_numpy(reward)
                        bootstrapping_q.append(0)
                    else:
                        # todo - verify the next_q_values are the q values of the selected action, not the full set
                        if self.hparams.get('value_aggr', 'mean') == 'max':
                            max_next_q_values_aggr = next_q_values.max()
                        if self.hparams.get('value_aggr', 'mean') == 'mean':
                            max_next_q_values_aggr = next_q_values.mean()
                        bootstrapping_q.append(max_next_q_values_aggr)
                        max_next_q_values_broadcast = torch.full_like(q_values, fill_value=max_next_q_values_aggr)
                        target_q_values = torch.from_numpy(reward) + (self.gamma ** self.nstep_learning) * max_next_q_values_broadcast
                    td_error = torch.abs(q_values - target_q_values)
                    td_error = torch.clamp(td_error, min=1e-8)
                    initial_priority = torch.norm(td_error).item()  # default L2 norm
                    trajectory.append((transition, initial_priority, False))
                else:
                    trajectory.append(transition)

            if not discarded:
                bootstrapped_returns = R.flatten() + self.gamma**self.nstep_learning * np.array(bootstrapping_q).flatten()


        # compute some stats and store in buffer
        n_rewards = len(dualbound_area)
        discounted_rewards = [np.sum(dualbound_area[idx:] * self.gamma**np.arange(n_rewards-idx)) for idx in range(n_rewards)]
        selected_q_avg = [np.mean(info.get('selected_q_values', torch.zeros((1,))).numpy()) for info in self.episode_history]
        selected_q_std = [np.std(info.get('selected_q_values', torch.zeros((1,))).numpy()) for info in self.episode_history]

        active_applied_ratio = []
        applied_available_ratio = []
        accuracy_list, f1_score_list, jaccard_sim_list = [], [], []
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        q_avg, q_std = [], []
        for info in self.episode_history:
            action_info = info['action_info']
            normalized_slack = action_info['normalized_slack']
            # todo: verify with Aleks - consider slack < 1e-10 as zero
            approximately_zero = np.abs(normalized_slack) < self.hparams['slack_tol']
            normalized_slack[approximately_zero] = 0

            applied = action_info['applied']
            is_active = normalized_slack[applied] == 0
            active_applied_ratio.append(sum(is_active)/sum(applied) if sum(applied) > 0 else 0)
            applied_available_ratio.append(sum(applied)/len(applied) if len(applied) > 0 else 0)
            # if self.demonstration_episode: todo verification
            accuracy_list.append(np.mean(action_info['selected_by_scip'] == action_info['selected_by_agent']))
            f1_score_list.append(f1_score(action_info['selected_by_scip'], action_info['selected_by_agent']))
            intersection = len(set(action_info['selected_by_scip']).intersection(action_info['selected_by_agent']))
            jaccard_sim_list.append(intersection / (len(action_info['selected_by_scip']) + len(action_info['selected_by_agent']) - intersection))

            # store for plotting later
            scip_action = info['action_info']['selected_by_scip']
            agent_action = info['action_info']['selected_by_agent']
            true_pos += sum(scip_action[scip_action == 1] == agent_action[scip_action == 1])
            true_neg += sum(scip_action[scip_action == 0] == agent_action[scip_action == 0])
            false_pos += sum(scip_action[agent_action == 1] != agent_action[agent_action == 1])
            false_neg += sum(scip_action[agent_action == 0] != agent_action[agent_action == 0])
            # compute average and std of the selected cuts q values
            q_avg.append(info['selected_q_values'].mean())
            q_std.append(info['selected_q_values'].std())

        # store episode results in tmp_stats_buffer
        db_auc = sum(dualbound_area)
        gap_auc = sum(gap_area)
        # stats_folder = 'Demonstrations/' if self.demonstration_episode else ''
        if self.training:
            # todo - add here db auc improvement
            self.training_stats['db_auc'].append(db_auc)
            self.training_stats['db_auc_improvement'].append(db_auc / self.instance_info['baselines']['default'][223]['db_auc'])
            self.training_stats['gap_auc'].append(gap_auc)
            self.training_stats['gap_auc_improvement'].append(gap_auc / self.instance_info['baselines']['default'][223]['gap_auc'] if self.instance_info['baselines']['default'][223]['gap_auc'] > 0 else -1)
            self.training_stats['active_applied_ratio'] += active_applied_ratio  # .append(np.mean(active_applied_ratio))
            self.training_stats['applied_available_ratio'] += applied_available_ratio  # .append(np.mean(applied_available_ratio))
            self.training_stats['accuracy'] += accuracy_list
            self.training_stats['f1_score'] += f1_score_list
            self.training_stats['jaccard_similarity'] += jaccard_sim_list
            if not discarded:
                self.last_training_episode_stats['bootstrapped_returns'] = bootstrapped_returns
                self.last_training_episode_stats['discounted_rewards'] = discounted_rewards
                self.last_training_episode_stats['selected_q_avg'] = selected_q_avg
                self.last_training_episode_stats['selected_q_std'] = selected_q_std

            stats = None
        else:
            stats = {**self.episode_stats,
                     'db_auc': db_auc,
                     'db_auc_improvement': db_auc / self.instance_info['baselines']['default'][self.scip_seed]['db_auc'],
                     'gap_auc': gap_auc,
                     'gap_auc_improvement': gap_auc / self.instance_info['baselines']['default'][self.scip_seed]['gap_auc'] if self.instance_info['baselines']['default'][self.scip_seed]['gap_auc'] > 0 else -1,
                     'active_applied_ratio': np.mean(active_applied_ratio),
                     'applied_available_ratio': np.mean(applied_available_ratio),
                     'accuracy': np.mean(accuracy_list),
                     'f1_score': np.mean(f1_score_list),
                     'jaccard_similarity': np.mean(jaccard_sim_list),
                     'tot_solving_time': self.episode_stats['solving_time'][-1],
                     'tot_lp_iterations': self.episode_stats['lp_iterations'][-1],
                     'terminal_state': self.terminal_state,
                     'true_pos': true_pos,
                     'true_neg': true_neg,
                     'false_pos': false_pos,
                     'false_neg': false_neg,
                     'discounted_rewards': discounted_rewards,
                     'selected_q_avg': selected_q_avg,
                     'selected_q_std': selected_q_std,
                     }

        # # todo remove this and store instead test episode_stats, terminal_state, gap_auc, db_auc, and send to logger as is.
        # if self.baseline.get('rootonly_stats', None) is not None:
        #     # this is evaluation round.
        #     # test_stats_buffer uses for determining the best model performance.
        #     # if we ignore_test_early_stop, then we don't consider episodes which terminated due to branching
        #     if not (self.terminal_state == 'NODE_LIMIT' and self.hparams.get('ignore_test_early_stop', False)):
        #         self.test_stats_buffer[stats_folder + 'db_auc_imp'].append(db_auc/self.baseline['baselines']['default'][self.scip_seed]['db_auc'])
        #         self.test_stats_buffer[stats_folder + 'gap_auc_imp'].append(gap_auc/self.baseline['baselines']['default'][self.scip_seed]['gap_auc'])
        #         self.test_stats_buffer['db_auc'].append(db_auc)
        #         self.test_stats_buffer['gap_auc'].append(gap_auc)
        #
        # # if self.demonstration_episode:  todo verification
        # # store performance for tracking best models, ignoring bad outliers (e.g branching occured)
        # if not self.terminal_state == 'NODE_LIMIT' or self.hparams.get('ignore_test_early_stop', False):
        #     self.test_perf_list.append(db_auc if self.dqn_objective == 'db_auc' else gap_auc)
        return trajectory, stats

    # done
    def _update_episode_stats(self):
        """ Collect statistics related to the action taken at the previous round.
        This function is assumed to be called in the consequent separation round
        after the action was taken.
        A corner case is when choosing "EMPTY_ACTION" (shouldn't happen if we force selecting at least one cut)
        then the function is called immediately, and we need to add 1 to the number of lp_rounds.
        """
        if self.stats_updated:  # or self.prev_action is None:   <- todo: this was a bug. missed the initial stats
            return
        # todo verify recording initial state stats before taking any action
        self.episode_stats['ncuts'].append(0 if self.prev_action is None else self.prev_action['ncuts'] )
        self.episode_stats['ncuts_applied'].append(self.model.getNCutsApplied())
        self.episode_stats['solving_time'].append(self.model.getSolvingTime())
        self.episode_stats['processed_nodes'].append(self.model.getNNodes())
        self.episode_stats['gap'].append(self.model.getGap())
        self.episode_stats['lp_iterations'].append(self.model.getNLPIterations())
        self.episode_stats['dualbound'].append(self.model.getDualbound())
        # todo - we always store the stats referring to the previous lp round action, so we need to subtract 1 from the
        #  the current LP round counter
        if self.terminal_state and self.terminal_state == 'EMPTY_ACTION':
            self.episode_stats['lp_rounds'].append(self.model.getNLPs()+1)  # todo - check if needed to add 1 when EMPTY_ACTION
        else:
            self.episode_stats['lp_rounds'].append(self.model.getNLPs())
        self.truncate_to_lp_iterations_limit()
        self.stats_updated = True

    def truncate_to_lp_iterations_limit(self):
        # enforce the lp_iterations_limit on the last two records
        lp_iterations_limit = self.lp_iterations_limit
        if lp_iterations_limit > 0 and self.episode_stats['lp_iterations'][-1] > lp_iterations_limit:
            # interpolate the dualbound and gap at the limit
            if self.episode_stats['lp_iterations'][-2] >= lp_iterations_limit:
                warn(f'BUG IN STATS\ntraining={self.training}\nterminal_state={self.terminal_state}\nepisode_stats={self.episode_stats}\ncur_graph={self.cur_graph}')
            # assert self.episode_stats['lp_iterations'][-2] < lp_iterations_limit, f'terminal_state={self.terminal_state}\nepisode_stats={self.episode_stats}\nepisode_history={self.episode_history}'
            # assert self.episode_stats['lp_iterations'][-2] < lp_iterations_limit, f'terminal_state={self.terminal_state}\n'+'episode_stats={\n' + "\n".join([f"{k}:{v}," for k,v in self.episode_stats.items()]) + '\n}' + f'episode_history={self.episode_history}'
            t = self.episode_stats['lp_iterations'][-2:]
            for k in ['dualbound', 'gap']:
                ft = self.episode_stats[k][-2:]
                # compute ft slope in the last interval [t[-2], t[-1]]
                slope = (ft[-1] - ft[-2]) / (t[-1] - t[-2])
                # compute the linear interpolation of ft at the limit
                interpolated_ft = ft[-2] + slope * (lp_iterations_limit - t[-2])
                self.episode_stats[k][-1] = interpolated_ft
            # finally truncate the lp_iterations to the limit
            self.episode_stats['lp_iterations'][-1] = lp_iterations_limit

    # done
    def set_eval_mode(self):
        self.training = False
        self.policy_net.eval()

    # done
    def set_training_mode(self):
        self.training = True
        self.policy_net.train()

    # done
    def load_checkpoint(self, filepath=None):
        if filepath is None:
            filepath = self.checkpoint_filepath
        if not os.path.exists(filepath):
            print(self.print_prefix, 'Checkpoint file does not exist! starting from scratch.')
            return
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        # self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_env_steps_done = checkpoint['num_env_steps_done']
        self.num_sgd_steps_done = checkpoint['num_sgd_steps_done']
        self.num_param_updates = checkpoint['num_param_updates']
        self.walltime_offset = checkpoint['walltime_offset']
        # self.best_perf = checkpoint['best_perf']
        # self.n_step_loss_moving_avg = checkpoint['n_step_loss_moving_avg']
        self.policy_net.to(self.device)
        # self.target_net.to(self.device)
        print(self.print_prefix, 'Loaded checkpoint from: ', filepath)

    # done
    @staticmethod
    def load_data(hparams):
        # datasets and baselines
        datasets = deepcopy(hparams['datasets'])

        # load instances:
        with open(os.path.join(hparams['datadir'], hparams['problem'], 'data.pkl'), 'rb') as f:
            instances = pickle.load(f)
        for dataset_name, dataset in datasets.items():
            dataset.update(instances[dataset_name])

        # for the validation and test datasets compute average performance of all baselines:
        # this should be done in the logger process only
        for dataset_name, dataset in datasets.items():
            if dataset_name[:8] == 'trainset':
                continue
            dataset['stats'] = {}
            for bsl in ['default', '15_random', '15_most_violated']:
                db_auc_list = []
                gap_auc_list = []
                for (_, baseline) in dataset['instances']:
                    optimal_value = baseline['optimal_value']
                    for scip_seed in dataset['scip_seed']:
                        # align curves to lp_iterations_limit
                        tmp_stats = {}
                        for k, v in baseline['baselines'][bsl][scip_seed].items():
                            if k not in ['lp_iterations', 'db_auc', 'gap_auc'] and len(v) > 0:
                                aligned_lp_iterations, aligned_v = truncate(t=baseline['baselines'][bsl][scip_seed]['lp_iterations'],
                                                                            ft=v,
                                                                            support=dataset['lp_iterations_limit'],
                                                                            interpolate=type(v[0]) == float)
                                tmp_stats[k] = aligned_v
                                tmp_stats['lp_iterations'] = aligned_lp_iterations
                        # override with aligned stats
                        baseline['baselines'][bsl][scip_seed] = tmp_stats

                        dualbound = baseline['baselines'][bsl][scip_seed]['dualbound']
                        gap = baseline['baselines'][bsl][scip_seed]['gap']
                        lpiter = baseline['baselines'][bsl][scip_seed]['lp_iterations']
                        db_auc = sum(get_normalized_areas(t=lpiter, ft=dualbound, t_support=dataset['lp_iterations_limit'], reference=optimal_value))
                        gap_auc = sum(get_normalized_areas(t=lpiter, ft=gap, t_support=dataset['lp_iterations_limit'], reference=0))
                        baseline['baselines'][bsl][scip_seed]['db_auc'] = db_auc
                        baseline['baselines'][bsl][scip_seed]['gap_auc'] = gap_auc
                        db_auc_list.append(db_auc)
                        gap_auc_list.append(gap_auc)
                # compute stats for the whole dataset
                db_auc_avg = np.mean(db_auc)
                db_auc_std = np.std(db_auc)
                gap_auc_avg = np.mean(gap_auc)
                gap_auc_std = np.std(gap_auc)
                dataset['stats'][bsl] = {'db_auc_avg': db_auc_avg,
                                         'db_auc_std': db_auc_std,
                                         'gap_auc_avg': gap_auc_avg,
                                         'gap_auc_std': gap_auc_std}
        return datasets

    def load_datasets(self):
        """
        Load train/valid/test sets
        todo - overfit: load only test100[0] as trainset and validset
        """
        hparams = self.hparams
        self.datasets = datasets = self.load_data(hparams)
        self.trainset = [v for k, v in self.datasets.items() if 'trainset' in k][0]

        if hparams.get('overfit', False):
            instances = []
            overfit_lp_iter_limits = []
            trainset_name = 'trainset_overfit'
            for dataset_name in hparams['overfit']:
                instances += datasets[dataset_name]['instances']
                overfit_lp_iter_limits += [datasets[dataset_name]['lp_iterations_limit']]*len(datasets[dataset_name]['instances'])
                trainset_name += f'_{dataset_name}'
            self.trainset['instances'] = instances
            self.trainset['num_instances'] = len(instances)
            self.trainset['dataset_name'] = trainset_name
            self.trainset['overfit_lp_iter_limits'] = overfit_lp_iter_limits

        self.graph_indices = torch.randperm(self.trainset['num_instances'])
        return datasets

    # done
    def initialize_training(self):
        # fix random seed for all experiment
        if self.hparams.get('seed', None) is not None:
            np.random.seed(self.hparams['seed'] + int(self.worker_id))
            torch.manual_seed(self.hparams['seed'] + int(self.worker_id))

        # initialize agent
        self.set_training_mode()
        if self.hparams.get('resume', False):
            self.load_checkpoint()
            # # initialize prioritized replay buffer internal counters, to continue beta from the point it was
            # if self.use_per:
            #     self.memory.num_sgd_steps_done = self.num_sgd_steps_done

    # done
    def execute_episode(self, G, instance_info, lp_iterations_limit, dataset_name, scip_seed=None, setting='root_only'):
        # create a SCIP model for G
        hparams = self.hparams
        if hparams['problem'] == 'MAXCUT':
            model, x, cut_generator = maxcut_mccormic_model(G, hparams=hparams,
                                                            use_heuristics=(hparams['use_heuristics'] or setting == 'branch_and_cut'),
                                                            use_random_branching=(setting != 'branch_and_cut'),
                                                            allow_restarts=(setting == 'branch_and_cut'))  #, use_propagation=False)
        elif hparams['problem'] == 'MVC':
            model, x = mvc_model(G,
                                 use_heuristics=(hparams['use_heuristics'] or setting == 'branch_and_cut'),
                                 use_random_branching=(setting != 'branch_and_cut'),
                                 allow_restarts=(setting == 'branch_and_cut'))  #, use_propagation=False)
            cut_generator = None
        if hparams['aggressive_separation']:
            set_aggresive_separation(model)

        # reset new episode
        self.init_episode(G, x, lp_iterations_limit, cut_generator=cut_generator, instance_info=instance_info,
                          dataset_name=dataset_name, scip_seed=scip_seed, setting=setting)

        # include self, setting lower priority than the cycle inequalities separator
        model.includeSepa(self, '#CS_TuningDQN', 'Tuning agent', priority=-100000000, freq=1)
        # include reset separator for restting maxcutsroot every round
        reset_sepa = CSResetSepa(hparams)
        model.includeSepa(reset_sepa, '#CS_reset', 'reset maxcutsroot', priority=99999999, freq=1)
        # set some model parameters, to avoid early branching.
        # termination condition is either optimality or lp_iterations_limit.
        # since there is no way to limit lp_iterations explicitly,
        # it is enforced implicitly by the separators, which won't add any more cuts.
        if setting == 'root_only':
            model.setLongintParam('limits/nodes', 1)  # solve only at the root node
            model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever

        # set environment random seed
        if scip_seed is not None:
            model.setBoolParam('randomization/permutevars', True)
            model.setIntParam('randomization/permutationseed', scip_seed)
            model.setIntParam('randomization/randomseedshift', scip_seed)

        if self.hparams.get('hide_scip_output', True):
            model.hideOutput()

        if self.hparams.get('debug_events'):
            debug_eventhdlr = DebugEvents(self.hparams.get('debug_events'))
            model.includeEventhdlr(debug_eventhdlr, "DebugEvents", "Catches "+",".join(self.hparams.get('debug_events')))

        # gong! run episode
        model.optimize()

        # compute stats and finish episode
        if setting == 'branch_and_cut':
            # retutn episode stats only.
            self._update_episode_stats()
            return None, self.episode_stats

        trajectory, stats = self.finish_episode()
        return trajectory, stats

    # done
    def evaluate(self):
        start_time = time()
        datasets = self.datasets
        # evaluate the model on the validation and test sets
        if self.num_param_updates == 0:
            # wait until the model starts learning
            return None, None
        global_step = self.num_param_updates
        test_summary = []
        # for workers which hasn't been assigned eval instances
        if len(self.eval_instances) == 0:
            return global_step, test_summary

        self.set_eval_mode()
        avg_times = {k: [] for k in set([tup[0] for tup in self.eval_instances])}
        for dataset_name, inst_idx, scip_seed in self.eval_instances:
            t0 = time()
            dataset = datasets[dataset_name]
            G, instance_info = dataset['instances'][inst_idx]
            self.cur_graph = f'{dataset_name} graph {inst_idx} seed {scip_seed}'
            _, stats = self.execute_episode(G, instance_info, dataset['lp_iterations_limit'],
                                            dataset_name=dataset_name,
                                            scip_seed=scip_seed)
            stats['dataset_name'] = dataset_name
            stats['inst_idx'] = inst_idx
            stats['scip_seed'] = scip_seed
            test_summary.append([(k, v) for k, v in stats.items()])
            avg_times[dataset_name].append(time()-t0)

        self.set_training_mode()
        avg_times = {k: np.mean(v) for k, v in avg_times.items()}

        self.print(f'Eval no. {global_step}\t| Total time: {time()-start_time}\t| Time/Instance: {avg_times}')
        return global_step, test_summary

    def run_test(self):
        self.load_datasets()
        # evaluate 3 models x 6 datasets x 5 insts X 3 seeds x 2 settings (root only and B&C) : 3x6x5x3x2 = 360 evaluations
        # distributed over 72 workers it is 5 evaluations per worker
        datasets = self.datasets
        model_params_files = [f'best_{dataset_name}_params.pkl' for dataset_name in datasets.keys() if 'valid' in dataset_name]
        settings = ['root_only', 'branch_and_cut']
        flat_instances = []
        for model_params_file in model_params_files:
            for setting in settings:
                for dataset_name, dataset in datasets.items():
                    if 'train' in dataset_name:
                        continue
                    for inst_idx in range(dataset['ngraphs']):
                        for scip_seed in dataset['scip_seed']:
                            flat_instances.append((model_params_file, setting, dataset_name, inst_idx, scip_seed))
        idx = self.worker_id - 1
        eval_instances = []
        while idx < len(flat_instances):
            eval_instances.append(flat_instances[idx])
            idx += self.hparams['num_workers']
        test_results = []
        current_model = 'none'
        self.set_eval_mode()
        for model_params_file, setting, dataset_name, inst_idx, scip_seed in eval_instances:
            # set model params to evaluate
            if model_params_file != current_model:
                with open(os.path.join(self.hparams['run_dir'], model_params_file), 'rb') as f:
                    new_params = pickle.load(f)
                for param, new_param in zip(self.policy_net.parameters(), new_params):
                    new_param = torch.FloatTensor(new_param).to(self.device)
                    param.data.copy_(new_param)
                current_model = model_params_file

            dataset = datasets[dataset_name]
            lp_iterations_limit = dataset['lp_iterations_limit'] if setting == 'root_only' else 100000
            G, instance_info = dataset['instances'][inst_idx]
            self.cur_graph = f'{dataset_name} graph {inst_idx} seed {scip_seed}'
            _, stats = self.execute_episode(G, instance_info,
                                            lp_iterations_limit=lp_iterations_limit,
                                            dataset_name=dataset_name,
                                            scip_seed=scip_seed,
                                            setting=setting)
            stats['model'] = model_params_file[:-4]
            stats['setting'] = setting
            stats['dataset_name'] = dataset_name
            stats['inst_idx'] = inst_idx
            stats['scip_seed'] = scip_seed
            stats['run_times'] = self.run_times
            test_results.append([(k, v) for k, v in stats.items()])
        # send all results back to apex controller and terminate
        self.send_2_apex_socket.send(pa.serialize(('test_results', self.worker_id, test_results).to_buffer()))

    def print(self, expr):
        print(self.print_prefix, expr)

    @staticmethod
    def get_custom_wandb_logs(validation_stats, dataset_name, best=False):
        return {}
