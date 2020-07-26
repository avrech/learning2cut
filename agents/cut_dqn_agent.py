from sklearn.metrics import f1_score
from pyscipopt import Sepa, SCIP_RESULT
from time import time
import numpy as np
from utils.data import Transition
import os
import math
import random
from gnn.models import Qnet, TQnet, TransformerDecoderContext
import torch
import scipy as sp
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data.batch import Batch
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from tqdm import tqdm
from utils.functions import get_normalized_areas
from collections import namedtuple
import matplotlib as mpl
import pickle
from utils.scip_models import maxcut_mccormic_model
from utils.buffer import ReplayBuffer, PrioritizedReplayBuffer
from separators.mccormick_cycle_separator import MccormickCycleSeparator
from copy import deepcopy
mpl.rc('figure', max_open_warning=0)
# mpl.rcParams['text.antialiased'] = False
# mpl.use('agg')
import matplotlib.pyplot as plt
StateActionContext = namedtuple('StateActionQValuesContext', ('scip_state', 'action', 'q_values', 'transformer_context'))


class CutDQNAgent(Sepa):
    def __init__(self, name='DQN', hparams={}, use_gpu=True, gpu_id=None, **kwargs):
        """
        Sample scip.Model state every time self.sepaexeclp is invoked.
        Store the generated data object in
        """
        super(CutDQNAgent, self).__init__()
        self.name = name
        self.hparams = hparams

        # DQN stuff
        self.use_per = hparams.get('use_per', True)
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(config=hparams)
        else:
            self.memory = ReplayBuffer(hparams.get('replay_buffer_capacity', 2**16))
        cuda_id = 'cuda' if gpu_id is None else f'cuda:{gpu_id}'
        self.device = torch.device(cuda_id if use_gpu and torch.cuda.is_available() else "cpu")
        self.batch_size = hparams.get('batch_size', 64)
        self.gamma = hparams.get('gamma', 0.999)
        self.eps_start = hparams.get('eps_start', 0.9)
        self.eps_end = hparams.get('eps_end', 0.05)
        self.eps_decay = hparams.get('eps_decay', 200)
        if hparams.get('dqn_arch', 'TQNet'):
            # todo - consider support also mean value aggregation.
            assert hparams.get('value_aggr') == 'max', "TQNet v3 supports only value_aggr == max"
            assert hparams.get('tqnet_version', 'v3') == 'v3', 'v1 and v2 are no longer supported. need to adapt to new decoder context'
        self.policy_net = TQnet(hparams=hparams, use_gpu=use_gpu, gpu_id=gpu_id).to(self.device) if hparams.get('dqn_arch', 'TQNet') == 'TQNet' else Qnet(hparams=hparams).to(self.device)
        self.target_net = TQnet(hparams=hparams, use_gpu=use_gpu, gpu_id=gpu_id).to(self.device) if hparams.get('dqn_arch', 'TQNet') == 'TQNet' else Qnet(hparams=hparams).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.tqnet_version = hparams.get('tqnet_version', 'v3')
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=hparams.get('lr', 0.001), weight_decay=hparams.get('weight_decay', 0.0001))
        # value aggregation method for the target Q values
        if hparams.get('value_aggr', 'mean') == 'max':
            self.value_aggr = scatter_max
        elif hparams.get('value_aggr', 'mean') == 'mean':
            self.value_aggr = scatter_mean
        self.nstep_learning = hparams.get('nstep_learning', 1)
        self.dqn_objective = hparams.get('dqn_objective', 'db_auc')
        self.use_transformer = hparams.get('dqn_arch', 'TQNet') == 'TQNet'
        self.empty_action_penalty = self.hparams.get('empty_action_penalty', 0)
        self.select_at_least_one_cut = self.hparams.get('select_at_least_one_cut', True)

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

        # instance specific data needed to be reset every episode
        self.G = None
        self.x = None
        self.y = None
        self.baseline = None
        self.scip_seed = None
        self.action = None
        self.prev_action = None
        self.prev_state = None
        self.state_action_qvalues_context_list = []
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
        self.finished_episode_stats = False
        self.cut_generator = None
        self.nseparounds = 0
        self.dataset_name = 'trainset'  # or <easy/medium/hard>_<validset/testset>
        self.lp_iterations_limit = -1
        self.terminal_state = False
        # learning from demonstrations stuff
        self.demonstration_episode = False
        self.num_demonstrations_done = 0

        # logging
        self.is_tester = True  # todo set in distributed setting
        self.is_worker = True
        self.is_learner = True
        # file system paths
        # todo - set worker-specific logdir for distributed DQN
        self.logdir = hparams.get('logdir', 'results')
        self.checkpoint_filepath = os.path.join(self.logdir, 'checkpoint.pt')
        self.writer = SummaryWriter(log_dir=os.path.join(self.logdir, 'tensorboard'))
        self.print_prefix = ''

        # tmp buffer for holding each episode results until averaging and appending to experiment_stats
        self.tmp_stats_buffer = {'db_auc': [], 'gap_auc': [], 'active_applied_ratio': [], 'applied_available_ratio': []}
        self.test_stats_buffer = {'db_auc_imp': [], 'gap_auc_imp': []}

        # learning from demonstrations stats
        for k in list(self.tmp_stats_buffer.keys()):
            self.tmp_stats_buffer['Demonstrations/' + k] = []
        self.tmp_stats_buffer['Demonstrations/accuracy'] = []
        self.tmp_stats_buffer['Demonstrations/f1_score'] = []
        for k in list(self.test_stats_buffer.keys()):
            self.test_stats_buffer['Demonstrations/' + k] = []

        # best performance log for validation sets
        self.best_perf = {k: -1000000 for k in hparams['datasets'].keys() if k[:8] == 'validset'}
        self.n_step_loss_moving_avg = 0
        self.demonstration_loss_moving_avg = 0

        # self.figures = {'Dual_Bound_vs_LP_Iterations': [], 'Gap_vs_LP_Iterations': []}
        self.figures = {}

        # debug
        self.sanity_check_stats = {'n_duplicated_cuts': [],
                                   'n_weak_cuts': [],
                                   'n_original_cuts': []}

        # initialize (set seed and load checkpoint)
        self.initialize_training()

    # done
    def init_episode(self, G, x, y, lp_iterations_limit, cut_generator=None, baseline=None, dataset_name='trainset25', scip_seed=None, demonstration_episode=False):
        self.G = G
        self.x = x
        self.y = y
        self.baseline = baseline
        self.scip_seed = scip_seed
        self.action = None
        self.prev_action = None
        self.prev_state = None
        self.state_action_qvalues_context_list = []
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
        self.finished_episode_stats = False
        self.cut_generator = cut_generator
        self.nseparounds = 0
        self.dataset_name = dataset_name
        self.lp_iterations_limit = lp_iterations_limit
        self.terminal_state = False
        self.demonstration_episode = demonstration_episode

    # done
    def sepaexeclp(self):
        if self.hparams.get('debug', False):
            print(self.print_prefix, 'dqn')

        # assert proper behavior
        self.nseparounds += 1
        if self.cut_generator is not None:
            assert self.nseparounds == self.cut_generator.nseparounds
            # assert self.nseparounds == self.model.getNLPs() todo: is it really important?

        # sanity check only:
        # for each cut in the separation storage add an identical cut with only a different scale,
        # and a weak cut with right shifted rhs
        if self.hparams.get('sanity_check', False) and self.is_tester:
            self.add_identical_and_weak_cuts()

        # finish with the previous step:
        self._update_episode_stats()

        # if for some reason we terminated the episode (lp iterations limit reached / empty action etc.
        # we dont want to run any further dqn steps, and therefore we return immediately.
        if self.terminal_state:
            # discard potentially added cuts and return
            self.model.clearCuts()
            result = {"result": SCIP_RESULT.DIDNOTRUN}

        elif self.model.getNLPIterations() < self.lp_iterations_limit:

            result = self._do_dqn_step()

            # sanity check: for each cut in the separation storage
            # check if its weak version was applied
            # and if both the original and the identical versions were applied.
            if self.hparams.get('sanity_check', False) and self.is_tester:
                self.track_identical_and_weak_cuts()

        else:
            # stop optimization (implicitly), and don't add any more cuts
            if self.hparams.get('verbose', 0) == 2:
                print(self.print_prefix + 'LP_ITERATIONS_LIMIT reached. DIDNOTRUN!')
            self.terminal_state = 'LP_ITERATIONS_LIMIT_REACHED'
            # get stats of prev_action
            self.model.getState(query=self.prev_action)
            # finish collecting episode stats
            self.finished_episode_stats = True
            # clear cuts and terminate
            self.model.clearCuts()
            result = {"result": SCIP_RESULT.DIDNOTRUN}

        # todo - what retcode should be returned here?
        #  currently: if selected cuts              -> SEPARATED
        #                discarded all or no cuts   -> DIDNOTFIND
        #                otherwise                  -> DIDNOTRUN
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
        The priority of calling the DQN separator should be the lowest, so it will be able to
        see all the available cuts.

        Learning from demonstrations:
        When learning from demonstrations, the agent doesn't take any action, but rather let SCIP select cuts, and track
        SCIP's actions.
        After the episode is done, SCIP actions are analyzed and a decoder context is reconstructed following
        SCIP's policy.
        TODO demonstration data is stored with additional flag demonstration=True, to inform
         the learner to compute the expert loss J_E.
        """
        # get the current state, a dictionary of available cuts (keyed by their names,
        # and query statistics related to the previous action (cut activeness etc.)
        cur_state, available_cuts = self.model.getState(state_format='tensor', get_available_cuts=True,
                                                        query=self.prev_action)

        # validate the solver behavior
        if self.prev_action is not None and not self.demonstration_episode:
            # assert that all the selected cuts were actually applied
            # otherwise, there is a bug (or maybe safety/feasibility issue?)
            assert (self.prev_action['selected'] == self.prev_action['applied']).all()

        # if there are available cuts, select action and continue to the next state
        if available_cuts['ncuts'] > 0:

            # select an action, and get the decoder context for a case we use transformer and q_values for PER
            action, q_values, decoder_context = self._select_action(cur_state)
            available_cuts['selected'] = action

            if not self.demonstration_episode:
                # apply the action
                if any(action):
                    # force SCIP to take the selected cuts and discard the others
                    self.model.forceCuts(action)
                    # set SCIP maxcutsroot and maxcuts to the number of selected cuts,
                    # in order to prevent it from adding more or less cuts
                    self.model.setIntParam('separating/maxcuts', int(sum(action)))
                    self.model.setIntParam('separating/maxcutsroot', int(sum(action)))
                    # continue to the next state
                    result = {"result": SCIP_RESULT.SEPARATED}

                else:
                    # todo - This action leads to the terminal state.
                    #        SCIP may apply now heuristics which will further improve the dualbound/gap.
                    #        However, those improvements are not related to the currently taken action.
                    #        So we snapshot here the dualbound and gap and other related stats,
                    #        and set the terminal_state flag accordingly.
                    # force SCIP to "discard" all the available cuts by flushing the separation storage
                    self.model.clearCuts()
                    if self.hparams.get('verbose', 0) == 2:
                        print(self.print_prefix + 'discarded all cuts')
                    self.terminal_state = 'EMPTY_ACTION'
                    self._update_episode_stats()
                    self.finished_episode_stats = True
                    result = {"result": SCIP_RESULT.DIDNOTFIND}

                # SCIP will execute the action,
                # and return here in the next LP round -
                # unless the instance is solved and the episode is done.
            else:
                # when learning from demonstrations, we use SCIP's cut selection, so don't do anything.
                result = {"result": SCIP_RESULT.DIDNOTRUN}

            # store the current state and action for
            # computing later the n-step rewards and the (s,a,r',s') transitions
            self.state_action_qvalues_context_list.append(StateActionContext(cur_state, available_cuts, q_values, decoder_context))
            self.prev_action = available_cuts
            self.prev_state = cur_state

        # If there are no available cuts we terminate the episode.
        # The stats of the previous action are already collected,
        # so we can finish collecting stats.
        # We don't store the current state-action pair,
        # since the terminal state is not important.
        # The final gap in this state can be either zero (OPTIMAL) or strictly positive.
        # However, model.getGap() can potentially return gap > 0, as SCIP stats will be updated only afterward.
        # So we temporarily set terminal_state to True (general description)
        # and we will accurately characterize it after the optimization terminates.
        elif available_cuts['ncuts'] == 0:
            # todo: check if we ever get here, and verify behavior
            self.prev_action = None
            self.terminal_state = True
            self.finished_episode_stats = True
            result = {"result": SCIP_RESULT.DIDNOTFIND}

        return result

    # done
    def _select_action(self, scip_state):
        # todo - what should be the return types? action only, or maybe also q values and decoder context?
        #  for simple tracking return all types, for compatibility with non-transformer models return only action.
        # transform scip_state into GNN data type

        # batch = Batch.from_data_list([Transition.create(scip_state, tqnet_version=self.tqnet_version)],
        #                              follow_batch=['x_c', 'x_v', 'x_a', 'ns_x_c', 'ns_x_v', 'ns_x_a']).to(self.device)
        batch = Transition.create(scip_state, tqnet_version=self.tqnet_version).as_batch().to(self.device)

        if self.training:
            if self.demonstration_episode:
                # take only greedy actions to compute online policy stats
                # in demonstration mode, we don't increment num_env_steps_done,
                # since we want to start exploration from the beginning once the demonstration phase is completed.
                sample, eps_threshold = 1, 0
            else:
                # take epsilon-greedy action
                sample = random.random()
                eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                                math.exp(-1. * self.num_env_steps_done / self.eps_decay)
                self.num_env_steps_done += 1

            if sample > eps_threshold:
                # take greedy action
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is the index of where max element was found.
                    # we pick action with the larger expected reward.
                    # todo - move all architectures to output dict format
                    output = self.policy_net(
                        x_c=batch.x_c,
                        x_v=batch.x_v,
                        x_a=batch.x_a,
                        edge_index_c2v=batch.edge_index_c2v,
                        edge_index_a2v=batch.edge_index_a2v,
                        edge_attr_c2v=batch.edge_attr_c2v,
                        edge_attr_a2v=batch.edge_attr_a2v,
                        edge_index_a2a=batch.edge_index_a2a,
                        edge_attr_a2a=batch.edge_attr_a2a
                    )
                    if self.use_transformer:
                        # todo- verification v3
                        # the action is not necessarily q_values.argmax(dim=1).
                        # take the action constructed internally in the transformer, and the corresponding q_values
                        greedy_q_values = output['q_values'].gather(1, output['action'].long().unsqueeze(1)).detach().cpu()  # todo - verification return the relevant q values only
                        # greedy_q_values = q_values.gather(1, self.policy_net.decoder_greedy_action.long().unsqueeze(1)).detach().cpu()  # todo - verification return the relevant q values only
                        greedy_action = output['action'].numpy()  # self.policy_net.decoder_greedy_action.numpy()
                        # return also the decoder context to store for backprop
                        greedy_action_decoder_context = output['decoder_context']
                        # greedy_action_decoder_context = self.policy_net.decoder_context
                    else:
                        greedy_q_values, greedy_action = output['q_values'].max(1)  # todo - verification
                        greedy_action = greedy_action.cpu().numpy().astype(np.bool)  # todo - verify detach()
                        greedy_q_values = greedy_q_values.cpu()  # todo - detach() is not necessary due to torch.no_grad()
                        greedy_action_decoder_context = None

                    return greedy_action, greedy_q_values, greedy_action_decoder_context

            else:
                # randomize action
                random_action = torch.randint_like(batch.a, low=0, high=2, dtype=torch.float32).cpu()
                if self.select_at_least_one_cut and random_action.sum() == 0:
                    # select a cut arbitrarily
                    random_action[torch.randint(low=0, high=len(random_action), size=(1,))] = 1

                # for prioritized experience replay we need the q_values to compute the initial priorities
                # whether we take a random action or not.
                # For transformer, we compute the random action q_values based on the random decoder context,
                # and we do it in parallel like we do in sgd_step()
                # For non-transformer model, it doesn't affect anything
                output = self.policy_net(
                    x_c=batch.x_c,
                    x_v=batch.x_v,
                    x_a=batch.x_a,
                    edge_index_c2v=batch.edge_index_c2v,
                    edge_index_a2v=batch.edge_index_a2v,
                    edge_attr_c2v=batch.edge_attr_c2v,
                    edge_attr_a2v=batch.edge_attr_a2v,
                    edge_index_a2a=batch.edge_index_a2a,
                    edge_attr_a2a=batch.edge_attr_a2a,
                    random_action=random_action  # for transformer to set context
                )
                random_action_decoder_context = output['decoder_context'] if self.use_transformer else None
                random_action_q_values = output['q_values'].detach().cpu().gather(1, random_action.long().unsqueeze(1))  # todo - verification. take the relevant q_values only.
                random_action = random_action.numpy().astype(np.bool)
                return random_action, random_action_q_values, random_action_decoder_context
        else:
            # in test time, take greedy action
            with torch.no_grad():
                output = self.policy_net(
                    x_c=batch.x_c,
                    x_v=batch.x_v,
                    x_a=batch.x_a,
                    edge_index_c2v=batch.edge_index_c2v,
                    edge_index_a2v=batch.edge_index_a2v,
                    edge_attr_c2v=batch.edge_attr_c2v,
                    edge_attr_a2v=batch.edge_attr_a2v,
                    edge_index_a2a=batch.edge_index_a2a,
                    edge_attr_a2a=batch.edge_attr_a2a
                )
                # todo enforce select_at_least_one_cut.
                #  in tqnet v2 it is enforced internally, so that the decoder_greedy_action is valid.
                if self.use_transformer:
                    greedy_action = output['action'].numpy()
                else:
                    greedy_action = output['q_values'].max(1)[1].cpu().numpy().astype(np.bool)
                assert not self.select_at_least_one_cut or any(greedy_action)
                # return None, None for q_values and decoder context,
                # since they are used only while generating experience
                return greedy_action, None, None

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
            Gap == 0. If the LP_ITERATIONS_LIMIT was reached, we interpolate the final db/gap at the limit.
            Otherwise, the final statistics are taken as is.
            In this case the agent is rewarded also for all SCIP's side effects (e.g. heuristics)
            which potentially helped in solving the instance.
        LP_ITERATIONS_LIMIT_REACHED:
            Gap >= 0. This state refers to the case in which the last action was not empty.
            We snapshot the current db/gap and interpolate the final db/gap at the limit.
        DIDNOTFIND:
            Gap > 0, ncuts == 0.
            We save the current db/gap as the final values,
            and do not consider any more db improvements (e.g. by heuristics)
        EMPTY_ACTION:
            Gap > 0, ncuts > 0, nselected == 0.
            The treatment is the same as DIDNOTFIND.

        In practice, this function is called after SCIP.optimize() terminates.
        self.terminal_state is set to None at the beginning, and once one of the 4 cases above is detected,
        self.terminal_state is set to the appropriate value.

        """
        if self.cut_generator is not None:
            assert self.nseparounds == self.cut_generator.nseparounds

        # classify the terminal state
        if self.terminal_state == 'EMPTY_ACTION':
            # all the available cuts were discarded.
            # the dualbound / gap might have been changed due to heuristics etc.
            # however, this improvement is not related to the empty action.
            # we extend the dualbound curve with constant and the LP iterations to the LP_ITERATIONS_LIMIT.
            # the discarded cuts slack is not relevant anyway, since we penalize this action with constant.
            # we set it to zero.
            self.prev_action['normalized_slack'] = np.zeros_like(self.prev_action['selected'], dtype=np.float32)
            self.prev_action['applied'] = np.zeros_like(self.prev_action['selected'], dtype=np.bool)

        elif self.terminal_state == 'LP_ITERATIONS_LIMIT_REACHED':
            pass
        elif self.terminal_state and self.model.getGap() == 0:
            self.terminal_state = 'OPTIMAL'
        elif self.terminal_state and self.model.getGap() > 0:
            self.terminal_state = 'DIDNOTFIND'

        assert self.terminal_state in ['OPTIMAL', 'LP_ITERATIONS_LIMIT_REACHED', 'DIDNOTFIND', 'EMPTY_ACTION']
        assert not (self.select_at_least_one_cut and self.terminal_state == 'EMPTY_ACTION')
        # in a case SCIP terminated without calling the agent,
        # we need to complete some feedback manually.
        # (it can happen only in terminal_state = OPTIMAL/LP_ITERATIONS_LIMIT_REACHED).
        # we need to evaluate the normalized slack of the applied cuts,
        # and to update the episode stats with the latest SCIP stats.
        if self.prev_action is not None and self.prev_action.get('normalized_slack', None) is None:
            ncuts = self.prev_action['ncuts']
            # todo not verified.
            #  restore the applied cuts from sepastore->selectedcutsnames
            selected_cuts_names = self.model.getSelectedCutsNames()
            for i, cut_name in enumerate(selected_cuts_names):
                self.prev_action[cut_name]['applied'] = True
                self.prev_action[cut_name]['selection_order'] = i
            applied = np.zeros((ncuts,), dtype=np.bool)
            selection_order = np.full_like(applied, fill_value=ncuts, dtype=np.long)
            for i, cut in enumerate(self.prev_action.values()):
                if i == ncuts:
                    break
                applied[i] = cut['applied']
                selection_order[i] = cut['selection_order']
            self.prev_action['applied'] = applied
            self.prev_action['selection_order'] = np.argsort(selection_order)[len(selected_cuts_names)]
            if not self.demonstration_episode:
                # assert that the action taken by agent was actually applied
                assert all(self.prev_action['selected'] == self.prev_action['applied'])

            assert self.terminal_state in ['OPTIMAL', 'LP_ITERATIONS_LIMIT_REACHED']
            nvars = self.model.getNVars()

            cuts_nnz_vals = self.prev_state['cut_nzrcoef']['vals']
            cuts_nnz_rowidxs = self.prev_state['cut_nzrcoef']['rowidxs']
            cuts_nnz_colidxs = self.prev_state['cut_nzrcoef']['colidxs']
            cuts_matrix = sp.sparse.coo_matrix((cuts_nnz_vals, (cuts_nnz_rowidxs, cuts_nnz_colidxs)), shape=[ncuts, nvars]).toarray()
            final_solution = self.model.getBestSol()
            sol_vector = [self.model.getSolVal(final_solution, x_i) for x_i in self.x.values()]
            sol_vector += [self.model.getSolVal(final_solution, y_ij) for y_ij in self.y.values()]
            sol_vector = np.array(sol_vector)
            # rhs slack of all cuts added at the previous round (including the discarded cuts)
            # generally, LP rows look like
            # lhs <= coef @ vars + cst <= rhs
            # here, self.prev_action['rhss'] = rhs - cst,
            # so cst is already subtracted.
            # in addition, we normalize the slack by the coefficients norm, to avoid different penalty to two same cuts,
            # with only constant factor between them
            cuts_norm = np.linalg.norm(cuts_matrix, axis=1)
            rhs_slack = self.prev_action['rhss'] - cuts_matrix @ sol_vector  # todo what about the cst and norm?
            normalized_slack = rhs_slack / cuts_norm
            # assign tightness penalty only to the selected cuts.
            self.prev_action['normalized_slack'] = np.zeros_like(self.prev_action['selected'], dtype=np.float32)
            self.prev_action['normalized_slack'][self.prev_action['selected']] = normalized_slack[self.prev_action['selected']]

            # update the rest of statistics needed to compute rewards
            self._update_episode_stats()

        # compute rewards and other stats for the whole episode,
        # and if in training session, push transitions into memory
        trajectory = self._compute_rewards_and_stats()
        # increase the number of episodes done
        if self.training:
            self.i_episode += 1

        return trajectory

    # done
    def _compute_rewards_and_stats(self):
        """
        Compute action-wise reward and store (s,a,r,s') transitions in memory
        By the way, compute so metrics for plotting, e.g.
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
            print(self.episode_stats)
            print(self.state_action_qvalues_context_list)

        # todo - consider squaring the dualbound/gap before computing the AUC.
        dualbound_area = get_normalized_areas(t=lp_iterations, ft=dualbound, t_support=lp_iterations_limit, reference=self.baseline['optimal_value'])
        gap_area = get_normalized_areas(t=lp_iterations, ft=gap, t_support=lp_iterations_limit, reference=0)  # optimal gap is always 0
        if self.dqn_objective == 'db_auc':
            objective_area = dualbound_area
        elif self.dqn_objective == 'gap_auc':
            objective_area = gap_area
        else:
            raise NotImplementedError

        trajectory = []
        if self.training:
            # compute n-step returns for each state-action pair (s_t, a_t)
            # and store a transition (s_t, a_t, r_t, s_{t+n}
            # todo - in learning from demonstrations we used to compute both 1-step and n-step returns.
            n_transitions = len(self.state_action_qvalues_context_list)
            n_steps = self.nstep_learning
            gammas = self.gamma**np.arange(n_steps).reshape(-1, 1)  # [1, gamma, gamma^2, ... gamma^{n-1}]
            indices = np.arange(n_steps).reshape(1, -1) + np.arange(n_transitions).reshape(-1, 1)  # indices of sliding windows
            # in case of n_steps > 1, pad objective_area with zeros only for avoiding overflow
            max_index = np.max(indices)
            if max_index >= len(objective_area):
                objective_area = np.pad(objective_area, (0, max_index+1-len(objective_area)), 'constant', constant_values=0)
            # take sliding windows of width n_step from objective_area
            n_step_rewards = objective_area[indices]
            # compute returns
            # R[t] = r[t] + gamma * r[t+1] + ... + gamma^(n-1) * r[t+n-1]
            R = n_step_rewards @ gammas
            # assign rewards and store transitions (s,a,r,s')
            for step, ((state, action, q_values, transformer_decoder_context), joint_reward) in enumerate(zip(self.state_action_qvalues_context_list, R)):
                if self.demonstration_episode:
                    # todo - create a decoder context that imitates SCIP cut selection
                    #  a. get initial_edge_index_a2a and initial_edge_attr_a2a
                    initial_edge_index_a2a, initial_edge_attr_a2a = Transition.get_initial_decoder_context(scip_state=state, tqnet_version=self.tqnet_version)
                    #  b. create context
                    transformer_decoder_context = self.policy_net.get_complete_context(
                        torch.from_numpy(action['applied']), initial_edge_index_a2a, initial_edge_attr_a2a,
                        selection_order=action['selection_order'])

                # get the next n-step state and q values. if the next state is terminal
                # return 0 as q_values (by convention)
                next_state, next_action, next_q_values, _ = self.state_action_qvalues_context_list[step + n_steps] if step + n_steps < n_transitions else (None, None, None, None)

                # credit assignment:
                # R is a joint reward for all cuts applied at each step.
                # now, assign to each cut its reward according to its slack
                # slack == 0 if the cut is tight, and > 0 otherwise. (<0 means violated and should happen)
                # so we punish inactive cuts by decreasing their reward to
                # R * (1 - slack)
                # The slack is normalized by the cut's norm, to fairly penalizing similar cuts of different norms.
                normalized_slack = action['normalized_slack']
                if self.hparams.get('credit_assignment', True):
                    credit = 1 - normalized_slack
                    reward = joint_reward * credit
                else:
                    reward = joint_reward * np.ones_like(normalized_slack)

                # penalize "empty" action
                is_empty_action = np.logical_not(action['selected']).all()
                if self.empty_action_penalty is not None and is_empty_action:
                    reward = np.full_like(normalized_slack, fill_value=self.empty_action_penalty)

                transition = Transition.create(scip_state=state,
                                               action=action['applied'],
                                               transformer_decoder_context=transformer_decoder_context,
                                               reward=reward,
                                               scip_next_state=next_state,
                                               tqnet_version=self.tqnet_version)

                if self.use_per:
                    # todo - compute initial priority for PER based on the policy q_values.
                    #        compute the TD error for each action in the current state as we do in sgd_step,
                    #        and then take the norm of the resulting cut-wise TD-errors as the initial priority
                    selected_action = torch.from_numpy(action['selected']).unsqueeze(1).long()  # cut-wise action
                    # q_values = q_values.gather(1, selected_action)  # gathering is done now in _select_action
                    if next_q_values is None:
                        # next state is terminal, and its q_values are 0 by convention
                        target_q_values = torch.from_numpy(reward)
                    else:
                        # todo - tqnet v2 & v3:
                        #  take only the max q value over the "select" entries
                        if self.use_transformer:
                            max_next_q_values_aggr = next_q_values[next_action['applied'] == 1].max()  # todo - verification
                        else:
                            # todo - verify the next_q_values are the q values of the selected action, not the full set
                            if self.hparams.get('value_aggr', 'mean') == 'max':
                                max_next_q_values_aggr = next_q_values.max()
                            if self.hparams.get('value_aggr', 'mean') == 'mean':
                                max_next_q_values_aggr = next_q_values.mean()

                        max_next_q_values_broadcast = torch.full_like(q_values, fill_value=max_next_q_values_aggr)
                        target_q_values = torch.from_numpy(reward) + (self.gamma ** self.nstep_learning) * max_next_q_values_broadcast
                    td_error = torch.abs(q_values - target_q_values)
                    td_error = torch.clamp(td_error, min=1e-8)
                    initial_priority = torch.norm(td_error).item()  # default L2 norm
                    trajectory.append((transition, initial_priority, self.demonstration_episode))
                else:
                    trajectory.append(transition)

        # compute some stats and store in buffer
        active_applied_ratio = []
        applied_available_ratio = []
        accuracy_list, f1_score_list = [], []
        for _, action, _, _ in self.state_action_qvalues_context_list:
            normalized_slack = action['normalized_slack']
            # because of numerical errors, we consider as zero |value| < 1e-6
            approximately_zero = np.abs(normalized_slack) < 1e-6
            normalized_slack[approximately_zero] = 0

            applied = action['applied']
            is_active = normalized_slack[applied] == 0
            active_applied_ratio.append(sum(is_active)/sum(applied) if sum(applied) > 0 else 0)
            applied_available_ratio.append(sum(applied)/len(applied) if len(applied) > 0 else 0)
            if self.demonstration_episode:
                accuracy_list.append(np.mean(action['applied'] == action['selected']))
                f1_score_list.append(f1_score(action['applied'], action['selected']))

        # store episode results in tmp_stats_buffer
        db_auc = sum(dualbound_area)
        gap_auc = sum(gap_area)
        stats_folder = 'Demonstrations/' if self.demonstration_episode else ''
        self.tmp_stats_buffer[stats_folder + 'db_auc'].append(db_auc)
        self.tmp_stats_buffer[stats_folder + 'gap_auc'].append(gap_auc)
        self.tmp_stats_buffer[stats_folder + 'active_applied_ratio'] += active_applied_ratio  # .append(np.mean(active_applied_ratio))
        self.tmp_stats_buffer[stats_folder + 'applied_available_ratio'] += applied_available_ratio  # .append(np.mean(applied_available_ratio))
        if self.baseline.get('rootonly_stats', None) is not None:
            # this is evaluation round.
            self.test_stats_buffer[stats_folder + 'db_auc_imp'].append(db_auc/self.baseline['rootonly_stats'][self.scip_seed]['db_auc'])
            self.test_stats_buffer[stats_folder + 'gap_auc_imp'].append(gap_auc/self.baseline['rootonly_stats'][self.scip_seed]['gap_auc'])
        if self.demonstration_episode:
            self.tmp_stats_buffer['Demonstrations/accuracy'] += accuracy_list # .append(np.mean(action['applied'] == action['selected']))
            self.tmp_stats_buffer['Demonstrations/f1_score'] += f1_score_list # .append(f1_score(action['applied'], action['selected']))

        return trajectory

    # done
    def optimize_model(self):
        """
        Single threaded DQN policy update function.
        Sample uniformly a batch from the memory and execute one SGD pass
        todo - support PER
               importance sampling +
               when recovering from failures wait like in Ape-X paper appendix
        """
        if len(self.memory) < self.hparams.get('replay_buffer_minimum_size', self.batch_size*10):
            return
        if self.use_per:
            # todo sample with beta

            transitions, weights, idxes, data_ids = self.memory.sample(self.batch_size)
            is_demonstration = idxes < self.hparams.get('replay_buffer_n_demonstrations', 0)
            new_priorities = self.sgd_step(transitions=transitions, importance_sampling_correction_weights=torch.from_numpy(weights), is_demonstration=is_demonstration)
            # update priorities
            self.memory.update_priorities(idxes, new_priorities, data_ids)

        else:
            # learning from demonstration is disabled with simple replay buffer.
            transitions = self.memory.sample(self.batch_size)
            self.sgd_step(transitions)
        self.num_param_updates += 1
        if self.num_sgd_steps_done % self.hparams.get('target_update_interval', 1000) == 0:
            self.update_target()

    def sgd_step(self, transitions, importance_sampling_correction_weights=None, is_demonstration=None):
        """ implement the basic DQN optimization step """

        # old replay buffer returned transitions as separated Transition objects
        batch = Transition.create_batch(transitions).to(self.device)  #, follow_batch=['x_c', 'x_v', 'x_a', 'ns_x_c', 'ns_x_v', 'ns_x_a']).to(self.device)

        action_batch = batch.a

        # Compute Q(s, a):
        # the model computes Q(s,:), then we select the columns of the actions taken, i.e action_batch.
        # TODO: transformer & learning from demonstrations:
        #  for each cut, we need to compute the supervising expert loss
        #  J_E = max( Q(s,a) + l(a,a_E) ) - Q(s,a_E).
        #  in order to compute the max term
        #  we need to compute the whole sequence of q_values as done in inference,
        #  and at each decoder iteration, to compute J_E.
        #  How can we do it efficiently?
        #  The cuts encoding are similar for all those cut-decisions, and can be computed only once.
        #  In order to have the full set of Q values,
        #  we can replicate the cut encodings |selected|+1 times and
        #  expand the decoder context to the full sequence of edge attributes accordingly.
        #  Each encoding replication with its corresponding context, can define a separate graph,
        #  batched together, and fed in parallel into the decoder.
        #  For the first |selected| replications we need to compute a single J_E for the corresponding
        #  selected cut, and for the last replication representing the q values of all the discarded cuts,
        #  we need to compute the |discarded| J_E losses for all the discarded cuts.
        #  Note: each one of those J_E losses should consider only the "remaining" cuts.
        policy_output = self.policy_net(
            x_c=batch.x_c,
            x_v=batch.x_v,
            x_a=batch.x_a,
            edge_index_c2v=batch.edge_index_c2v,
            edge_index_a2v=batch.edge_index_a2v,
            edge_attr_c2v=batch.edge_attr_c2v,
            edge_attr_a2v=batch.edge_attr_a2v,
            edge_index_a2a=batch.edge_index_a2a,
            edge_attr_a2a=batch.edge_attr_a2a,
            mode='batch'
        )
        q_values = policy_output['q_values'].gather(1, action_batch.unsqueeze(1))

        # demonstration loss:
        # TODO - currently implemented only for tqnet v3
        if is_demonstration.any():
            demonstration_loss = self.compute_demonstration_loss(policy_output['cut_encoding'], batch.x_a_batch, transitions, is_demonstration, importance_sampling_correction_weights)
        else:
            demonstration_loss = 0

        # Compute the Bellman target for all next states.
        # Expected values of actions for non_terminal_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0] for DQN, or based on the policy_net
        # prediction for DDQN.
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was terminal.
        # The value of a state is computed as in BQN paper https://arxiv.org/pdf/1711.08946.pdf

        # for each graph in the next state batch, and for each cut, compute the q_values
        # using the target network.
        target_next_output = self.target_net(
            x_c=batch.ns_x_c,
            x_v=batch.ns_x_v,
            x_a=batch.ns_x_a,
            edge_index_c2v=batch.ns_edge_index_c2v,
            edge_index_a2v=batch.ns_edge_index_a2v,
            edge_attr_c2v=batch.ns_edge_attr_c2v,
            edge_attr_a2v=batch.ns_edge_attr_a2v,
            edge_index_a2a=batch.ns_edge_index_a2a,
            edge_attr_a2a=batch.ns_edge_attr_a2a,
            mode='batch'
        )
        target_next_q_values = target_next_output['q_values']
        # compute the target using either DQN or DDQN formula
        # todo: TQNet v2:
        #  compute the argmax across the "select" q values only.
        if self.use_transformer:
            assert self.tqnet_version == 'v3', 'v1 and v2 are no longer supported'
            if self.hparams.get('update_rule', 'DQN') == 'DQN':
                # y = r + gamma max_a' target_net(s', a')
                max_target_next_q_values_aggr, _ = scatter_max(target_next_q_values[:, 1],  # find max across the "select" q values only
                                                               batch.ns_x_a_batch,
                                                               # target index of each element in source
                                                               dim=0,  # scattering dimension
                                                               dim_size=self.batch_size)
            elif self.hparams.get('update_rule', 'DQN') == 'DDQN':
                # y = r + gamma target_net(s', argmax_a' policy_net(s', a'))
                policy_next_output = self.policy_net(
                    x_c=batch.ns_x_c,
                    x_v=batch.ns_x_v,
                    x_a=batch.ns_x_a,
                    edge_index_c2v=batch.ns_edge_index_c2v,
                    edge_index_a2v=batch.ns_edge_index_a2v,
                    edge_attr_c2v=batch.ns_edge_attr_c2v,
                    edge_attr_a2v=batch.ns_edge_attr_a2v,
                    edge_index_a2a=batch.ns_edge_index_a2a,
                    edge_attr_a2a=batch.ns_edge_attr_a2a,
                    mode='batch'
                )
                policy_next_q_values = policy_next_output['q_values']
                # compute the argmax over 'select' for each graph in batch
                _, argmax_policy_next_q_values = scatter_max(policy_next_q_values[:, 1], # find max across the "select" q values only
                                                             batch.ns_x_a_batch, # target index of each element in source
                                                             dim=0,  # scattering dimension
                                                             dim_size=self.batch_size)
                max_target_next_q_values_aggr = target_next_q_values[argmax_policy_next_q_values, 1].detach()

        else:
            if self.hparams.get('update_rule', 'DQN') == 'DQN':
                # y = r + gamma max_a' target_net(s', a')
                max_target_next_q_values = target_next_q_values.max(1)[0].detach()
            elif self.hparams.get('update_rule', 'DQN') == 'DDQN':
                # y = r + gamma target_net(s', argmax_a' policy_net(s', a'))
                policy_next_output = self.policy_net(
                    x_c=batch.ns_x_c,
                    x_v=batch.ns_x_v,
                    x_a=batch.ns_x_a,
                    edge_index_c2v=batch.ns_edge_index_c2v,
                    edge_index_a2v=batch.ns_edge_index_a2v,
                    edge_attr_c2v=batch.ns_edge_attr_c2v,
                    edge_attr_a2v=batch.ns_edge_attr_a2v,
                    edge_index_a2a=batch.ns_edge_index_a2a,
                    edge_attr_a2a=batch.ns_edge_attr_a2a,
                )
                policy_next_q_values = policy_next_output['q_values']
                argmax_policy_next_q_values = policy_next_q_values.max(1)[1].detach()
                max_target_next_q_values = target_next_q_values.gather(1, argmax_policy_next_q_values).detach()

            # aggregate the action-wise values using mean or max,
            # and generate for each graph in the batch a single value
            max_target_next_q_values_aggr = self.value_aggr(max_target_next_q_values,  # source vector
                                                            batch.ns_x_a_batch,  # target index of each element in source
                                                            dim=0,  # scattering dimension
                                                            dim_size=self.batch_size)   # output tensor size in dim after scattering

        # override with zeros the values of terminal states which are zero by convention
        max_target_next_q_values_aggr[batch.ns_terminal] = 0
        # broadcast the next state q_values graph-wise to update all action-wise rewards
        target_next_q_values_broadcast = max_target_next_q_values_aggr[batch.x_a_batch]

        # now compute the target Q values - action-wise
        reward_batch = batch.r
        target_q_values = reward_batch + (self.gamma ** self.nstep_learning) * target_next_q_values_broadcast

        # Compute Huber loss
        # todo - support importance sampling correction - double check
        if self.use_per:
            # broadcast each transition importance sampling weight to all its related losses
            importance_sampling_correction_weights = importance_sampling_correction_weights.to(self.device)[batch.x_a_batch]
            # multiply each action loss by its importance sampling correction weight and average
            n_step_loss = (importance_sampling_correction_weights * F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1), reduction='none')).mean()
        else:
            # generate equal weights for all losses
            n_step_loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))

        # loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1)) - original pytorch example
        self.n_step_loss_moving_avg = 0.95 * self.n_step_loss_moving_avg + 0.05 * n_step_loss.detach().cpu().numpy()

        # sum all losses
        loss = n_step_loss + self.hparams.get('demonstration_loss_coef', 0.5) * demonstration_loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.num_sgd_steps_done += 1
        # todo - for distributed learning return losses to update priorities - double check
        if self.use_per:
            # for transition, compute the norm of its TD-error
            td_error = torch.abs(q_values - target_q_values.unsqueeze(1)).detach()
            td_error = torch.clamp(td_error, min=1e-8)
            # to compute p,q norm take power p and compute sqrt q.
            td_error_l2_norm = torch.sqrt(scatter_add(td_error ** 2, batch.x_a_batch, # target index of each element in source
                                                      dim=0,                          # scattering dimension
                                                      dim_size=self.batch_size))      # output tensor size in dim after scattering

            new_priorities = td_error_l2_norm.squeeze().cpu().numpy()

            return new_priorities
        else:
            return None

    def compute_demonstration_loss(self, cut_encoding_batch, cut_encoding_transition_id, transitions, is_demonstration, weights):
        """
        break batch cut encoding to graph level.
        expand decoder context
        mask irrelevant entries
        create l(a,a_E)
        J_E scatter_max q(s,a) + l(a,a_E) - q(s, a_E)

        :param cut_encoding_transition_id:
        :param transitions:
        :param is_demonstration: indicate on deomnstration data
        :param weights: importance sampling weights
        :return: loss (scalar or array)?
        """
        # expand graph-level encoding
        # todo - consider removing all edges whose target nodes will be masked anyway, and save computations.
        encoding_list, edge_index_list, edge_attr_list, q_mask_list, large_margin_list, a_E_list, weights_list = [], [], [], [], [], [], []
        for idx, (transition, demonstration_data, weight) in enumerate(zip(transitions, is_demonstration, weights)):
            if not demonstration_data:
                continue
            encoding = cut_encoding_batch[cut_encoding_transition_id == idx]

            # expand decoder input to full context
            n = encoding.shape[0]
            dense_attr = torch.sparse.FloatTensor(transition.edge_index_a2a, transition.edge_attr_a2a).to_dense()  # n x n x d
            expanded_o_ij = dense_attr[:, :, :1].unsqueeze(0).expand(n, -1, -1, -1)
            expanded_i_features = dense_attr[:, :, 1:].transpose(0, 1).unsqueeze(2).expand(-1, -1, n, -1)
            expanded_context = torch.cat([expanded_o_ij, expanded_i_features], dim=-1)

            # create Data object for each context slice, and the corresponding masks and margin

            selected_cuts = transition.a.nonzero().view(-1).tolist() if transition.a.bool().any() else []
            for i in selected_cuts:
                context = expanded_context[i, :, :, :]
                edge_index = (context[:, :, 0] + 1).nonzero().t()  # get matrix indices sorted
                edge_attr = context[edge_index[0], edge_index[1], :]
                q_mask = torch.zeros((n, 2), dtype=torch.bool)
                q_mask[context[:, i, -1] == 1] = True  # mask the already selected cuts
                a_E = torch.zeros((n, 2), dtype=torch.bool)
                a_E[i, 1] = True
                large_margin = torch.full_like(q_mask, fill_value=self.hparams.get('demonstration_large_margin', 0.1), dtype=torch.float32)
                large_margin[a_E] = 0
                edge_index_list.append(edge_index)
                edge_attr_list.append(edge_attr)
                encoding_list.append(encoding)
                q_mask_list.append(q_mask)
                large_margin_list.append(large_margin)
                a_E_list.append(a_E)
            # expand context for the first discarded cut only
            # todo - consider penalizing for all discarded cuts, not only the first one.
            #  it can be straightforwardly done by repeating the same computation for all discarded cuts.
            if transition.a.logical_not().any():
                i = transition.a.logical_not().nonzero().view(-1)[0]
                context = expanded_context[i, :, :, :]
                edge_index = (context[:, :, 0] + 1).nonzero().t()  # get matrix indices sorted
                edge_attr = context[edge_index[0], edge_index[1], :]
                q_mask = torch.zeros((n, 2), dtype=torch.bool)
                q_mask[context[:, i, -1] == 1] = True  # mask the already selected cuts
                q_mask[:, 0] = True   # mask the discard options
                q_mask[i, 0] = False  # except of discard cut i
                a_E = torch.zeros((n, 2), dtype=torch.bool)
                a_E[i, 0] = True
                large_margin = torch.full_like(q_mask, fill_value=self.hparams.get('demonstration_large_margin', 0.1), dtype=torch.float32)
                large_margin[a_E] = 0
                edge_index_list.append(edge_index)
                edge_attr_list.append(edge_attr)
                encoding_list.append(encoding)
                q_mask_list.append(q_mask)
                large_margin_list.append(large_margin)
                a_E_list.append(a_E)
            # broadcast the transition weight to all its related losses
            # the number of losses is the number of selected cuts plus (1 if there are discarded cuts else 0)
            weights_list.append(torch.full((len(selected_cuts)+transition.a.logical_not().any(), ), fill_value=weight, dtype=torch.float32))

        # build helper graphs to compute the demonstration loss in parallel
        expanded_data_list = [Data(x=x, edge_index=edge_index, edge_attr=edge_attr) for x, edge_index, edge_attr in zip(encoding_list, edge_index_list, edge_attr_list)]
        expanded_batch = Batch.from_data_list(expanded_data_list).to(self.device)
        batch_q_mask = torch.cat(q_mask_list, dim=0).to(self.device)
        batch_large_margin = torch.cat(large_margin_list, dim=0).to(self.device)
        batch_a_E = torch.cat(a_E_list, dim=0).to(self.device)
        batch_weights = torch.cat(weights_list, dim=0).to(self.device)

        # decode the full set of q values
        x, _, _ = self.policy_net.decoder_conv((expanded_batch.x, expanded_batch.edge_index, expanded_batch.edge_attr))
        q = self.policy_net.q(x)
        q_a_E = q[batch_a_E]                    # expert action q values
        q_plus_l = q + batch_large_margin       # add large margin and
        q_plus_l[batch_q_mask] = -float('Inf')  # mask the non-relevant entries

        # compute max graph-wise
        max_q_plus_l, _ = scatter_max(q_plus_l, expanded_batch.batch, dim=0, dim_size=expanded_batch.num_graphs)
        max_q_plus_l, _ = max_q_plus_l.max(dim=1)

        # compute demonstration loss J_E = max [Q(s,a) + large margin] - Q(s, a_E)
        losses = max_q_plus_l - q_a_E
        loss = (losses * batch_weights).mean()

        # log demonstration loss moving average
        self.demonstration_loss_moving_avg = 0.95 * self.demonstration_loss_moving_avg + 0.05 * loss.detach().cpu().numpy()

        # todo - compute accuracy

        return loss

    # done
    def update_target(self):
        # Update the target network, copying all weights and biases in DQN
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def add_identical_and_weak_cuts(self):
        """
        For each cut in the separation storage
        add:
            - identical cut with only different scale
            - weak cut with rhs shifted right such that the weak cut is still efficacious.
        This function is written to maximize clarity, and not necessarily efficiency.
        """
        model = self.model
        cuts = self.model.getCuts()
        n_original_cuts = len(cuts)
        cut_names = set()
        for cut in cuts:
            cut_name = cut.name
            cut_names.add(cut_name)
            cut_rhs = cut.getRhs()
            cut_lhs = cut.getLhs()
            cut_cst = cut.getConstant()
            assert cut_cst == 0
            cut_cols = cut.getCols()
            cut_coef = cut.getVals()
            cut_vars = [col.getVar() for col in cut_cols]

            # add an identical cut by multiplying the original cut by a constant
            scaling_factor = np.random.randint(low=2, high=5)
            identical_cut = self.model.createEmptyRowSepa(self,
                                                          cut_name + 'iden',
                                                          rhs=cut_rhs * scaling_factor,
                                                          lhs=cut_lhs * scaling_factor,
                                                          local=cut.isLocal(),
                                                          removable=cut.isRemovable())
            model.cacheRowExtensions(identical_cut)
            for v, c in zip(cut_vars, cut_coef):
                model.addVarToRow(identical_cut, v, c * scaling_factor)
            # flush all changes before adding the cut
            model.flushRowExtensions(identical_cut)
            infeasible = model.addCut(identical_cut)
            assert not infeasible
            model.releaseRow(identical_cut)

            # add a weak cut by shifting right the rhs such that the cut is still efficacious
            cut_activity = model.getRowLPActivity(cut)
            # the activity of a violated cut should be higher than the rhs.
            # we generate a weak cut by shifting the rhs to random value sampled uniformly
            # from (cut_activity, cut_rhs)
            weak_rhs = 0
            while weak_rhs >= cut_activity or weak_rhs <= cut_rhs:
                weak_rhs = np.random.uniform(low=cut_rhs, high=cut_activity)
            assert weak_rhs < cut_activity and weak_rhs > cut_rhs, f'weak_rhs: {weak_rhs}, cut_activity: {cut_activity}, cut_rhs: {cut_rhs}'
            weak_cut = self.model.createEmptyRowSepa(self,
                                                     cut_name + 'weak',
                                                     rhs=weak_rhs,
                                                     lhs=cut_lhs,
                                                     local=cut.isLocal(),
                                                     removable=cut.isRemovable())
            model.cacheRowExtensions(weak_cut)
            for v, c in zip(cut_vars, cut_coef):
                model.addVarToRow(weak_cut, v, c)
            # flush all changes before adding the cut
            model.flushRowExtensions(weak_cut)
            infeasible = model.addCut(weak_cut)
            assert not infeasible
            model.releaseRow(weak_cut)

        self.sanity_check_stats['cur_cut_names'] = cut_names
        self.sanity_check_stats['cur_n_original_cuts'] = n_original_cuts

    def track_identical_and_weak_cuts(self):
        cur_n_original_cuts = self.sanity_check_stats['cur_n_original_cuts']
        if cur_n_original_cuts == 0:
            return
        cut_names = list(self.prev_action.keys())[:cur_n_original_cuts*3]
        cur_n_duplicated_cuts = 0
        cur_n_weak_cuts = 0
        added_cuts = set()

        for cut_name, selected in zip(cut_names, self.prev_action['selected']):
            if selected:
                if cut_name[-4:] == 'weak':
                    cur_n_weak_cuts += 1
                    continue
                basename = cut_name[:-4] if cut_name[-4:] == 'iden' else cut_name
                if basename in added_cuts:
                    cur_n_duplicated_cuts += 1
                else:
                    added_cuts.add(basename)
        self.sanity_check_stats['n_duplicated_cuts'].append(cur_n_duplicated_cuts)
        self.sanity_check_stats['n_weak_cuts'].append(cur_n_weak_cuts)
        self.sanity_check_stats['n_original_cuts'].append(cur_n_original_cuts)

    # done
    def _update_episode_stats(self):
        """ Collect statistics related to the action taken at the previous round.
        This function is assumed to be called in the consequent separation round
        after the action was taken.
        A corner case is when choosing "EMPTY_ACTION" (shouldn't happen if we force selecting at least one cut)
        then the function is called immediately, and we need to add 1 to the number of lp_rounds.
        """
        if self.finished_episode_stats or self.prev_action is None:
            return
        self.episode_stats['ncuts'].append(self.prev_action['ncuts'])
        self.episode_stats['ncuts_applied'].append(self.model.getNCutsApplied())
        self.episode_stats['solving_time'].append(self.model.getSolvingTime())
        self.episode_stats['processed_nodes'].append(self.model.getNNodes())
        self.episode_stats['gap'].append(self.model.getGap())
        self.episode_stats['lp_iterations'].append(self.model.getNLPIterations())
        self.episode_stats['dualbound'].append(self.model.getDualbound())

        if self.terminal_state and self.terminal_state == 'EMPTY_ACTION':
            self.episode_stats['lp_rounds'].append(self.model.getNLPs()+1)  # todo - check if needed to add 1 when EMPTY_ACTION
        else:
            self.episode_stats['lp_rounds'].append(self.model.getNLPs())


        # enforce the lp_iterations_limit
        lp_iterations_limit = self.lp_iterations_limit
        if lp_iterations_limit > 0 and self.episode_stats['lp_iterations'][-1] > lp_iterations_limit:
            # interpolate the dualbound and gap at the limit
            assert self.episode_stats['lp_iterations'][-2] < lp_iterations_limit
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
    def log_stats(self, save_best=False, plot_figures=False, global_step=None, info={}):
        """
        Average tmp_stats_buffer values, log to tensorboard dir,
        and reset tmp_stats_buffer for the next round.
        This function should be called periodically during training,
        and at the end of every validation/test set evaluation
        save_best should be set to the best model according to the agent performnace on the validation set.
        If the model has shown its best so far, we save the model parameters and the dualbound/gap curves
        The global_step in plots is by default the number of policy updates.
        This global_step measure holds both for single threaded DQN and distributed DQN,
        where self.num_policy_updates is updated in
            single threaded DQN - every time optimize_model() is executed
            distributed DQN - every time the workers' policy is updated
        Tracking the global_step is essential for "resume-training".
        TODO adapt function to run on learner and workers separately.
            learner - plot loss
            worker - plot auc etc.
            test_worker - plot valid/test auc frac and figures
            in single thread - these are all true.
            need to separate workers' logdir in the distributed main script
        """
        if global_step is None:
            global_step = self.num_param_updates

        print(self.print_prefix, f'Global step: {global_step} | {self.dataset_name}\t|', end='')
        cur_time_sec = time() - self.start_time + self.walltime_offset

        if self.is_tester:
            if plot_figures:
                self.decorate_figures()

            if save_best:
                perf = np.mean(self.tmp_stats_buffer[self.dqn_objective])  # todo bug with (-) ?
                if perf > self.best_perf[self.dataset_name]:
                    self.best_perf[self.dataset_name] = perf
                    self.save_checkpoint(filepath=os.path.join(self.logdir, f'best_{self.dataset_name}_checkpoint.pt'))
                    self.save_figures(filename_prefix='best')

            # add episode figures (for validation and test sets only)
            if plot_figures:
                for figname in self.figures['fignames']:
                    self.writer.add_figure(figname + '/' + self.dataset_name, self.figures[figname]['fig'],
                                           global_step=global_step, walltime=cur_time_sec)

            # plot dualbound and gap auc improvement over the baseline (for validation and test sets only)
            for k, vals in self.test_stats_buffer.items():
                if len(vals) > 0:
                    avg = np.mean(vals)
                    std = np.std(vals)
                    print('{}: {:.4f} | '.format(k, avg), end='')
                    self.writer.add_scalar(k + '/' + self.dataset_name, avg, global_step=global_step, walltime=cur_time_sec)
                    self.writer.add_scalar(k+'_std' + '/' + self.dataset_name, std, global_step=global_step, walltime=cur_time_sec)
                    self.test_stats_buffer[k] = []

            # sanity check
            if self.hparams.get('sanity_check', False):
                n_original_cuts = np.array(self.sanity_check_stats['n_original_cuts'])
                n_duplicated_cuts = np.array(self.sanity_check_stats['n_duplicated_cuts'])
                n_weak_cuts = np.array(self.sanity_check_stats['n_weak_cuts'])
                sanity_check_stats = {'dup_frac': n_duplicated_cuts / n_original_cuts,
                                      'weak_frac': n_weak_cuts / n_original_cuts}
                for k, vals in sanity_check_stats.items():
                    avg = np.mean(vals)
                    std = np.std(vals)
                    print('{}: {:.4f} | '.format(k, avg), end='')
                    self.writer.add_scalar(k + '/' + self.dataset_name, avg, global_step=global_step, walltime=cur_time_sec)
                    self.writer.add_scalar(k + '_std' + '/' + self.dataset_name, std, global_step=global_step,
                                           walltime=cur_time_sec)
                self.sanity_check_stats['n_original_cuts'] = []
                self.sanity_check_stats['n_duplicated_cuts'] = []
                self.sanity_check_stats['n_weak_cuts'] = []

        if self.is_worker or self.is_tester:
            # plot normalized dualbound and gap auc
            for k, vals in self.tmp_stats_buffer.items():
                if len(vals) == 0:
                    continue
                avg = np.mean(vals)
                std = np.std(vals)
                print('{}: {:.4f} | '.format(k, avg), end='')
                self.writer.add_scalar(k + '/' + self.dataset_name, avg, global_step=global_step, walltime=cur_time_sec)
                self.writer.add_scalar(k + '_std' + '/' + self.dataset_name, std, global_step=global_step, walltime=cur_time_sec)
                self.tmp_stats_buffer[k] = []

        if self.is_learner:
            # log the average loss of the last training session
            print('{}-step Loss: {:.4f} | '.format(self.nstep_learning, self.n_step_loss_moving_avg), end='')
            print('Demonstration Loss: {:.4f} | '.format(self.demonstration_loss_moving_avg), end='')
            self.writer.add_scalar('Nstep_Loss', self.n_step_loss_moving_avg, global_step=global_step, walltime=cur_time_sec)
            self.writer.add_scalar('Demonstration_Loss', self.n_step_loss_moving_avg, global_step=global_step, walltime=cur_time_sec)
            print(f'SGD Step: {self.num_sgd_steps_done} | ', end='')

        # print the additional info
        for k, v in info.items():
            print(k + ': ' + v + ' | ', end='')

        # print times
        d = int(np.floor(cur_time_sec/(3600*24)))
        h = int(np.floor(cur_time_sec/3600) - 24*d)
        m = int(np.floor(cur_time_sec/60) - 60*(24*d + h))
        s = int(cur_time_sec) % 60
        print('Iteration Time: {:.1f}[sec]| '.format(cur_time_sec - self.last_time_sec), end='')
        print('Total Time: {}-{:02d}:{:02d}:{:02d}'.format(d, h, m, s))
        self.last_time_sec = cur_time_sec

    # done
    def init_figures(self, fignames, nrows=10, ncols=3, col_labels=['seed_i']*3, row_labels=['graph_i']*10):
        for figname in fignames:
            fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
            fig.set_size_inches(w=8, h=10)
            fig.set_tight_layout(True)
            self.figures[figname] = {'fig': fig, 'axes': axes}
        self.figures['nrows'] = nrows
        self.figures['ncols'] = ncols
        self.figures['col_labels'] = col_labels
        self.figures['row_labels'] = row_labels
        self.figures['fignames'] = fignames

    # done
    def add_episode_subplot(self, row, col):
        """
        plot the last episode curves to subplot in position (row, col)
        plot dqn agent dualbound/gap curves together with the baseline curves.
        should be called after each validation/test episode with row=graph_idx, col=seed_idx
        """
        if 'Dual_Bound_vs_LP_Iterations' in self.figures.keys():
            bsl = self.baseline['rootonly_stats'][self.scip_seed]
            bsl_lpiter, bsl_db, bsl_gap = bsl['lp_iterations'], bsl['dualbound'], bsl['gap']
            dqn_lpiter, dqn_db, dqn_gap = self.episode_stats['lp_iterations'], self.episode_stats['dualbound'], self.episode_stats['gap']
            if dqn_lpiter[-1] < self.lp_iterations_limit:
                # extend curve to the limit
                dqn_lpiter = dqn_lpiter + [self.lp_iterations_limit]
                dqn_db = dqn_db + dqn_db[-1:]
                dqn_gap = dqn_gap + dqn_gap[-1:]
            if bsl_lpiter[-1] < self.lp_iterations_limit:
                # extend curve to the limit
                bsl_lpiter = bsl_lpiter + [self.lp_iterations_limit]
                bsl_db = bsl_db + bsl_db[-1:]
                bsl_gap = bsl_gap + bsl_gap[-1:]
            assert dqn_lpiter[-1] == self.lp_iterations_limit
            assert bsl_lpiter[-1] == self.lp_iterations_limit
            # plot dual bound
            ax = self.figures['Dual_Bound_vs_LP_Iterations']['axes'][row, col]
            ax.plot(dqn_lpiter, dqn_db, 'b', label='DQN')
            ax.plot(bsl_lpiter, bsl_db, 'r', label='SCIP default')
            ax.plot([0, self.baseline['lp_iterations_limit']], [self.baseline['optimal_value']]*2, 'k', label='optimal value')
            # plot gap
            ax = self.figures['Gap_vs_LP_Iterations']['axes'][row, col]
            ax.plot(dqn_lpiter, dqn_gap, 'b', label='DQN')
            ax.plot(bsl_lpiter, bsl_gap, 'r', label='SCIP default')
            ax.plot([0, self.baseline['lp_iterations_limit']], [0, 0], 'k', label='optimal gap')

        # plot imitation performance bars
        if 'Imitation_Performance' in self.figures.keys():
            true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
            for saqc in self.state_action_qvalues_context_list:
                scip_action = saqc[1]['applied']
                agent_action = saqc[1]['selected']
                true_pos += sum(scip_action[scip_action == 1] == agent_action[scip_action == 1])
                true_neg += sum(scip_action[scip_action == 0] == agent_action[scip_action == 0])
                false_pos += sum(scip_action[agent_action == 1] != agent_action[agent_action == 1])
                false_neg += sum(scip_action[agent_action == 0] != agent_action[agent_action == 0])
            # lp_rounds = np.arange(1, len(self.state_action_qvalues_context_list)+1)
            ax = self.figures['Imitation_Performance']['axes'][row, col]
            ax.bar(-0.3, true_pos, width=0.2, label='true pos')
            ax.bar(-0.1, true_neg, width=0.2, label='true neg')
            ax.bar(+0.1, false_pos, width=0.2, label='false pos')
            ax.bar(+0.3, false_neg, width=0.2, label='false neg')

    # done
    def decorate_figures(self, legend=True, col_labels=True, row_labels=True):
        """ save figures to png file """
        # decorate (title, labels etc.)
        nrows, ncols = self.figures['nrows'], self.figures['ncols']
        for figname in self.figures['fignames']:
            if col_labels:
                # add col labels at the first row only
                for col in range(ncols):
                    ax = self.figures[figname]['axes'][0, col]
                    ax.set_title(self.figures['col_labels'][col])
            if row_labels:
                # add row labels at the first col only
                for row in range(nrows):
                    ax = self.figures[figname]['axes'][row, 0]
                    ax.set_ylabel(self.figures['row_labels'][row])
            if legend:
                # add legend to the bottom-left subplot only
                ax = self.figures[figname]['axes'][-1, 0]
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=1, borderaxespad=0.)

    # done
    def save_figures(self, filename_prefix=None):
        for figname in ['Dual_Bound_vs_LP_Iterations', 'Gap_vs_LP_Iterations']:
            # save png
            fname = f'{self.dataset_name}_{figname}.png'
            if filename_prefix is not None:
                fname = filename_prefix + '_' + fname
            fpath = os.path.join(self.logdir, fname)
            self.figures[figname]['fig'].savefig(fpath)

    # done
    def save_checkpoint(self, filepath=None):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_env_steps_done': self.num_env_steps_done,
            'num_sgd_steps_done': self.num_sgd_steps_done,
            'num_param_updates': self.num_param_updates,
            'i_episode': self.i_episode,
            'walltime_offset': time() - self.start_time + self.walltime_offset,
            'best_perf': self.best_perf,
            'n_step_loss_moving_avg': self.n_step_loss_moving_avg,
        }, filepath if filepath is not None else self.checkpoint_filepath)
        if self.hparams.get('verbose', 1) > 1:
            print(self.print_prefix, 'Saved checkpoint to: ', filepath if filepath is not None else self.checkpoint_filepath)

    # done
    def _save_if_best(self):
        """Save the model if show the best performance on the validation set.
        The performance is the -(dualbound/gap auc),
        according to the DQN objective"""
        perf = -np.mean(self.tmp_stats_buffer[self.dqn_objective])
        if perf > self.best_perf[self.dataset_name]:
            self.best_perf[self.dataset_name] = perf
            self.save_checkpoint(filepath=os.path.join(self.logdir, f'best_{self.dataset_name}_checkpoint.pt'))

    # done
    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_filepath):
            print(self.print_prefix, 'Checkpoint file does not exist! starting from scratch.')
            return
        checkpoint = torch.load(self.checkpoint_filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_env_steps_done = checkpoint['num_env_steps_done']
        self.num_sgd_steps_done = checkpoint['num_sgd_steps_done']
        self.num_param_updates = checkpoint['num_param_updates']
        self.i_episode = checkpoint['i_episode']
        self.walltime_offset = checkpoint['walltime_offset']
        self.best_perf = checkpoint['best_perf']
        self.n_step_loss_moving_avg = checkpoint['n_step_loss_moving_avg']
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        print(self.print_prefix, 'Loaded checkpoint from: ', self.checkpoint_filepath)

    # done
    def load_datasets(self):
        """
        Load train/valid/test sets
        todo - overfit: load only test100[0] as trainset and validset
        """
        hparams = self.hparams

        # datasets and baselines
        datasets = deepcopy(hparams['datasets'])

        # todo - in overfitting sanity check consider only the first instance of the overfitted dataset
        overfit_dataset_name = self.hparams.get('overfit', False)
        if overfit_dataset_name in datasets.keys():
            for dataset_name in hparams['datasets'].keys():
                if dataset_name != overfit_dataset_name:
                    datasets.pop(dataset_name)

        for dataset_name, dataset in datasets.items():
            datasets[dataset_name]['datadir'] = os.path.join(hparams['datadir'], dataset_name,
                                                       "barabasi-albert-n{}-m{}-weights-{}-seed{}".format(
                                                           dataset['graph_size'], dataset['barabasi_albert_m'],
                                                           dataset['weights'], dataset['dataset_generation_seed']))

            # read all graphs with their baselines from disk
            dataset['instances'] = []
            for filename in tqdm(os.listdir(datasets[dataset_name]['datadir']), desc=f'Loading {dataset_name}'):
                # todo - overfitting sanity check consider only graph_0_0.pkl
                if overfit_dataset_name and filename != 'graph_0_0.pkl':
                    continue

                with open(os.path.join(datasets[dataset_name]['datadir'], filename), 'rb') as f:
                    G, baseline = pickle.load(f)
                    if baseline['is_optimal']:
                        dataset['instances'].append((G, baseline))
                    else:
                        print(filename, ' is not solved to optimality')
            dataset['num_instances'] = len(dataset['instances'])

        # for the validation and test datasets compute some metrics:
        for dataset_name, dataset in datasets.items():
            if dataset_name[:8] == 'trainset':
                continue
            db_auc_list = []
            gap_auc_list = []
            for (_, baseline) in dataset['instances']:
                optimal_value = baseline['optimal_value']
                for scip_seed in dataset['scip_seed']:
                    dualbound = baseline['rootonly_stats'][scip_seed]['dualbound']
                    gap = baseline['rootonly_stats'][scip_seed]['gap']
                    lpiter = baseline['rootonly_stats'][scip_seed]['lp_iterations']
                    db_auc = sum(get_normalized_areas(t=lpiter, ft=dualbound, t_support=dataset['lp_iterations_limit'],
                                                      reference=optimal_value))
                    gap_auc = sum(
                        get_normalized_areas(t=lpiter, ft=gap, t_support=dataset['lp_iterations_limit'], reference=0))
                    baseline['rootonly_stats'][scip_seed]['db_auc'] = db_auc
                    baseline['rootonly_stats'][scip_seed]['gap_auc'] = gap_auc
                    db_auc_list.append(db_auc)
                    gap_auc_list.append(gap_auc)
            # compute stats for the whole dataset
            db_auc_avg = np.mean(db_auc)
            db_auc_std = np.std(db_auc)
            gap_auc_avg = np.mean(gap_auc)
            gap_auc_std = np.std(gap_auc)
            dataset['stats'] = {}
            dataset['stats']['db_auc_avg'] = db_auc_avg
            dataset['stats']['db_auc_std'] = db_auc_std
            dataset['stats']['gap_auc_avg'] = gap_auc_avg
            dataset['stats']['gap_auc_std'] = gap_auc_std

        self.datasets = datasets
        # todo - overfitting sanity check -
        #  change 'testset100' to 'validset100' to enable logging stats collected only for validation sets.
        #  set trainset and validset100
        #  remove all the other datasets from database
        if overfit_dataset_name:
            self.trainset = deepcopy(self.datasets[overfit_dataset_name])
            self.trainset['dataset_name'] = 'trainset-' + self.trainset['dataset_name'] + '[0]'
            self.trainset['instances'][0][1].pop('rootonly_stats')
        else:
            self.trainset = self.datasets['trainset25']
        self.graph_indices = torch.randperm(self.trainset['num_instances'])
        return datasets

    # done
    def initialize_training(self):
        # fix random seed for all experiment
        if self.hparams.get('seed', None) is not None:
            np.random.seed(self.hparams['seed'])
            torch.manual_seed(self.hparams['seed'])

        # initialize agent
        self.set_training_mode()
        if self.hparams.get('resume_training', False):
            self.load_checkpoint()
            # initialize prioritized replay buffer internal counters, to continue beta from the point it was
            if self.use_per:
                self.memory.num_sgd_steps_done = self.num_sgd_steps_done

    # done
    def execute_episode(self, G, baseline, lp_iterations_limit, dataset_name, scip_seed=None, demonstration_episode=False):
        # create SCIP model for G
        hparams = self.hparams
        model, x, y = maxcut_mccormic_model(G, use_general_cuts=hparams.get('use_general_cuts',
                                                                            False))  # disable default cuts

        # include cycle inequalities separator with high priority
        cycle_sepa = MccormickCycleSeparator(G=G, x=x, y=y, name='MLCycles', hparams=hparams)
        model.includeSepa(cycle_sepa, 'MLCycles',
                          "Generate cycle inequalities for the MaxCut McCormick formulation",
                          priority=1000000, freq=1)

        # reset new episode
        self.init_episode(G, x, y, lp_iterations_limit, cut_generator=cycle_sepa, baseline=baseline,
                          dataset_name=dataset_name, scip_seed=scip_seed,
                          demonstration_episode=demonstration_episode)

        # include self, setting lower priority than the cycle inequalities separator
        model.includeSepa(self, 'DQN', 'Cut selection agent',
                          priority=-100000000, freq=1)

        # set some model parameters, to avoid early branching.
        # termination condition is either optimality or lp_iterations_limit.
        # since there is no way to limit lp_iterations explicitly,
        # it is enforced implicitly by the separators, which won't add any more cuts.
        model.setLongintParam('limits/nodes', 1)  # solve only at the root node
        model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever

        # set environment random seed
        if scip_seed is None:
            # set random scip seed
            scip_seed = np.random.randint(1000000000)
        model.setBoolParam('randomization/permutevars', True)
        model.setIntParam('randomization/permutationseed', scip_seed)
        model.setIntParam('randomization/randomseedshift', scip_seed)

        if self.hparams.get('hide_scip_output', True):
            model.hideOutput()

        # gong! run episode
        model.optimize()

        # once episode is done
        trajectory = self.finish_episode()
        return trajectory

    # done
    def evaluate(self, datasets=None, ignore_eval_interval=False, eval_demonstration=False):
        if datasets is None:
            datasets = self.datasets
        # evaluate the model on the validation and test sets
        if self.num_param_updates == 0:
            # wait until the model starts learning
            return
        global_step = self.num_param_updates
        self.set_eval_mode()
        for dataset_name, dataset in datasets.items():
            if 'trainset' in dataset_name:
                continue
            if ignore_eval_interval or global_step % dataset['eval_interval'] == 0:
                fignames = ['Dual_Bound_vs_LP_Iterations', 'Gap_vs_LP_Iterations'] if not eval_demonstration else ['Imitation_Performance']
                self.init_figures(fignames,
                                  nrows=dataset['num_instances'],
                                  ncols=len(dataset['scip_seed']),
                                  col_labels=[f'Seed={seed}' for seed in dataset['scip_seed']],
                                  row_labels=[f'inst {inst_idx}' for inst_idx in
                                              range(dataset['num_instances'])])
                for inst_idx, (G, baseline) in enumerate(dataset['instances']):
                    for seed_idx, scip_seed in enumerate(dataset['scip_seed']):
                        if self.hparams.get('verbose', 0) == 2:
                            print('##################################################################################')
                            print(f'dataset: {dataset_name}, inst: {inst_idx}, seed: {scip_seed}')
                            print('##################################################################################')
                        self.execute_episode(G, baseline, dataset['lp_iterations_limit'], dataset_name=dataset_name,
                                             scip_seed=scip_seed, demonstration_episode=eval_demonstration)
                        self.add_episode_subplot(inst_idx, seed_idx)

                self.log_stats(save_best=('validset' in dataset_name and not eval_demonstration), plot_figures=True)

        self.set_training_mode()

    # done
    def train(self):
        datasets = self.load_datasets()
        trainset = self.trainset
        graph_indices = self.graph_indices
        hparams = self.hparams

        # training infinite loop
        for i_episode in range(self.i_episode + 1, hparams['num_episodes']):
            # sample graph randomly
            graph_idx = graph_indices[i_episode % len(graph_indices)]
            G, baseline = trainset['instances'][graph_idx]
            if hparams.get('debug', False):
                filename = os.listdir(trainset['datadir'])[graph_idx]
                filename = os.path.join(trainset['datadir'], filename)
                print(f'instance no. {graph_idx}, filename: {filename}')

            # execute episode and collect experience
            trajectory = self.execute_episode(G, baseline, trainset['lp_iterations_limit'], dataset_name=trainset['dataset_name'],
                                              demonstration_episode=(self.num_demonstrations_done < self.hparams.get('replay_buffer_n_demonstrations', 0)))

            # increment the counter of demonstrations done
            if self.num_demonstrations_done < self.hparams.get('replay_buffer_n_demonstrations', 0):
                self.num_demonstrations_done += len(trajectory)

            if i_episode % len(graph_indices) == 0:
                graph_indices = torch.randperm(trainset['num_instances'])

            # push experience into memory
            self.memory.add_data_list(trajectory)

            # perform 1 optimization step
            self.optimize_model()

            global_step = self.num_param_updates
            if global_step > 0 and global_step % hparams.get('log_interval', 100) == 0:
                self.log_stats()

            if global_step > 0 and global_step % hparams.get('checkpoint_interval', 100) == 0:
                self.save_checkpoint()

            # evaluate periodically
            self.evaluate()

        return 0

    # done
    def get_model(self):
        """ useful for distributed actors """
        return self.policy_net
