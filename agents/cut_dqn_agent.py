from sklearn.metrics import f1_score
from pyscipopt import Sepa, SCIP_RESULT, SCIP_STAGE
from time import time
import numpy as np
from utils.data import Transition
from utils.misc import get_img_from_fig
from utils.event_hdlrs import DebugEvents, BranchingEventHdlr
import os
import math
import random
from gnn.models import Qnet, TQnet, TransformerDecoderContext
import torch
import scipy as sp
import torch.optim as optim
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_add
from tqdm import tqdm
from utils.functions import get_normalized_areas, truncate
from collections import namedtuple
import matplotlib as mpl
import pickle
from utils.scip_models import maxcut_mccormic_model
from utils.buffer import ReplayBuffer, PrioritizedReplayBuffer
from separators.mccormick_cycle_separator import MccormickCycleSeparator
from copy import deepcopy
mpl.rc('figure', max_open_warning=0)
import matplotlib.pyplot as plt
import wandb


StateActionContext = namedtuple('StateActionQValuesContext', ('scip_state', 'action', 'q_values', 'transformer_context'))
DemonstrationBatch = namedtuple('DemonstrationBatch', (
    'context_edge_index',
    'context_edge_attr',
    'action',
    'idx',
    'conv_aggr_out_idx',
    'encoding_broadcast',
    'action_batch',
))


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
        self.cur_graph = None

        # instance specific data needed to be reset every episode
        self.G = None
        self.x = None
        self.y = None
        self.baseline = None
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
        self.node_limit_reached = False
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
        self.run_dir = hparams['run_dir']
        self.checkpoint_filepath = os.path.join(self.run_dir, 'checkpoint.pt')
        # self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, 'tensorboard'))  # todo remove after wandb works
        self.print_prefix = ''

        # tmp buffer for holding each episode results until averaging and appending to experiment_stats
        self.tmp_stats_buffer = {'db_auc': [], 'gap_auc': [], 'active_applied_ratio': [], 'applied_available_ratio': []}
        self.test_stats_buffer = {'db_auc_imp': [], 'gap_auc_imp': [], 'db_auc': [], 'gap_auc': []}
        self.test_perf_list = []

        # tmp buffer for holding cycle-statistics
        self.cycle_stats = None

        # learning from demonstrations stats
        for k in list(self.tmp_stats_buffer.keys()):
            self.tmp_stats_buffer['Demonstrations/' + k] = []
        self.tmp_stats_buffer['Demonstrations/accuracy'] = []
        self.tmp_stats_buffer['Demonstrations/f1_score'] = []
        # for k in list(self.test_stats_buffer.keys()):
        #     self.test_stats_buffer['Demonstrations/' + k] = []

        self.test_stats_dict = {}  # store testset stats by graph_id and seed

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

        # debug todo remove when finished
        self.debug_n_tracking_errors = 0
        self.debug_n_early_stop = 0
        self.debug_n_episodes_done = 0

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
        self.nseparounds = 0
        self.dataset_name = dataset_name
        self.lp_iterations_limit = lp_iterations_limit
        self.terminal_state = False
        self.demonstration_episode = demonstration_episode
        self.node_limit_reached = False

    # done
    def sepaexeclp(self):
        if self.hparams.get('debug_events', False):
            self.print('DEBUG MSG: dqn separator called')

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
        # todo - in case of no cuts, we return here a second time without any new action. we shouldn't record stats twice.
        self._update_episode_stats()

        # if for some reason we terminated the episode (lp iterations limit reached / empty action etc.
        # we dont want to run any further dqn steps, and therefore we return immediately.
        if self.terminal_state:
            # discard all the cuts in the separation storage and return
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
                self.print('LP_ITERATIONS_LIMIT reached. DIDNOTRUN!')
            self.terminal_state = 'LP_ITERATIONS_LIMIT_REACHED'
            # get stats of prev_action
            self.model.getState(query=self.prev_action)
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
        info = {}
        # get the current state, a dictionary of available cuts (keyed by their names,
        # and query statistics related to the previous action (cut activeness etc.)
        cur_state, available_cuts = self.model.getState(state_format='tensor', get_available_cuts=True,
                                                        query=self.prev_action)
        info['state_info'], info['action_info'] = cur_state, available_cuts

        # validate the solver behavior
        if self.prev_action is not None:
            # assert that all the selected cuts were actually applied
            # otherwise, there is a bug (or maybe safety/feasibility issue?)
            selected_cuts = self.prev_action['selected_by_scip'] if self.demonstration_episode else self.prev_action['selected_by_agent']
            assert (selected_cuts == self.prev_action['applied']).all()

        # if there are available cuts, select action and continue to the next state
        if available_cuts['ncuts'] > 0:

            # select an action, and get the decoder context for a case we use transformer and q_values for PER
            # todo old: action, q_values, decoder_context = self._select_action(cur_state)
            action_info = self._select_action(cur_state)
            selected = action_info['selected_by_agent']
            available_cuts['selected_by_agent'] = action_info['selected_by_agent'].numpy()
            for k, v in action_info.items():
                info[k] = v

            # prob what scip cut selection algorithm would do in this state
            # todo - check overhead, and if too large then enable only in evaluate()
            cut_names_selected_by_scip = self.prob_scip_cut_selection()
            available_cuts['selected_by_scip'] = np.array([cut_name in cut_names_selected_by_scip for cut_name in available_cuts['cuts'].keys()])
            if self.demonstration_episode:
                # use SCIP's cut selection (don't do anything)
                result = {"result": SCIP_RESULT.DIDNOTRUN}

            else:
                # apply the action
                if any(selected):
                    # force SCIP to take the selected cuts and discard the others
                    self.model.forceCuts(selected.numpy())
                    # set SCIP maxcutsroot and maxcuts to the number of selected cuts,
                    # in order to prevent it from adding more or less cuts
                    self.model.setIntParam('separating/maxcuts', int(sum(selected)))
                    self.model.setIntParam('separating/maxcutsroot', int(sum(selected)))
                    # continue to the next state
                    result = {"result": SCIP_RESULT.SEPARATED}

                else:
                    raise Exception('this case is not valid anymore. use hparam select_at_least_one_cut=True')
                    # todo - This action leads to the terminal state.
                    #        SCIP may apply now heuristics which will further improve the dualbound/gap.
                    #        However, this improvement is not related to the currently taken action.
                    #        So we snapshot here the dualbound and gap and other related stats,
                    #        and set the terminal_state flag accordingly.
                    #        NOTE - with self.select_at_least_one_cut=True this shouldn't happen
                    # force SCIP to "discard" all the available cuts by flushing the separation storage
                    self.model.clearCuts()
                    if self.hparams.get('verbose', 0) == 2:
                        self.print('discarded all cuts')
                    self.terminal_state = 'EMPTY_ACTION'
                    self._update_episode_stats()
                    result = {"result": SCIP_RESULT.DIDNOTFIND}

                # SCIP will execute the action,
                # and return here in the next LP round -
                # unless the instance is solved and the episode is done.

            # store the current state and action for
            # computing later the n-step rewards and the (s,a,r',s') transitions
            self.episode_history.append(info)
            # todo - old: self.episode_history.append(StateActionContext(cur_state, available_cuts, q_values, decoder_context))
            self.prev_action = available_cuts
            self.prev_state = cur_state
            self.stats_updated = False  # mark false to record relevant stats after this action will make effect

        # todo
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
                random_action = None
            else:
                # randomize action
                random_action = torch.randint_like(batch.a, low=0, high=2).cpu().bool()
                if self.select_at_least_one_cut and random_action.sum() == 0:
                    # select a cut arbitrarily
                    random_action[torch.randint(low=0, high=len(random_action), size=(1,))] = True
        else:
            random_action = None

        # take greedy action
        with torch.no_grad():
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
                edge_attr_a2a=batch.edge_attr_a2a,
                mode='inference',
                query_action=random_action
            )
        assert not self.select_at_least_one_cut or output['selected_by_agent'].any()
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

        if self.cut_generator is not None:
            assert self.nseparounds == self.cut_generator.nseparounds

        # classify the terminal state
        if self.terminal_state == 'EMPTY_ACTION':
            raise ValueError('invalid state. set argument select_at_least_one_cut=True')
            # all the available cuts were discarded.
            # the dualbound / gap might have been changed due to heuristics etc.
            # however, this improvement is not related to the empty action.
            # we extend the dualbound curve with constant and the LP iterations to the LP_ITERATIONS_LIMIT.
            # the discarded cuts slack is not relevant anyway, since we penalize this action with constant.
            # we set it to zero.
            self.prev_action['normalized_slack'] = np.zeros_like(self.prev_action['selected_by_agent'], dtype=np.float32)
            self.prev_action['applied'] = np.zeros_like(self.prev_action['selected_by_agent'], dtype=np.bool)

        elif self.terminal_state == 'LP_ITERATIONS_LIMIT_REACHED':
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
                self.node_limit_reached = True
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
            self.node_limit_reached = True  # todo - why event handler doesn't catch all branching events?
            self.terminal_state = 'NODE_LIMIT'

        # discard episodes which terminated early without optimal solution, to avoid extremely bad rewards.
        if self.terminal_state == 'NODE_LIMIT' and self.hparams.get('discard_bad_experience', False) \
                and self.training and self.model.getNLPIterations() < 0.90 * self.lp_iterations_limit:
            # todo remove printing-  debug
            self.debug_n_early_stop += 1
            self.print(f'discarded early stop {self.debug_n_early_stop}/{self.debug_n_episodes_done}')
            return []

        assert self.terminal_state in ['OPTIMAL', 'LP_ITERATIONS_LIMIT_REACHED', 'NODE_LIMIT', 'EMPTY_ACTION']
        assert not (self.select_at_least_one_cut and self.terminal_state == 'EMPTY_ACTION')
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
                return []
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
                    return []

            assert self.terminal_state in ['OPTIMAL', 'LP_ITERATIONS_LIMIT_REACHED', 'NODE_LIMIT']
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
            self.prev_action['normalized_slack'] = np.zeros_like(self.prev_action['selected_by_agent'], dtype=np.float32)
            self.prev_action['normalized_slack'][self.prev_action['selected_by_agent']] = normalized_slack[self.prev_action['selected_by_agent']]

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
            print(self.episode_stats)
            print(self.episode_history)

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
            n_transitions = len(self.episode_history)
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
            for step, (step_info, joint_reward) in enumerate(zip(self.episode_history, R)):
                state, action, q_values = step_info['state_info'], step_info['action_info'], step_info['selected_q_values']
                if self.demonstration_episode:
                    # create a decoder context corresponding to SCIP cut selection order
                    # a. get initial_edge_index_a2a and initial_edge_attr_a2a
                    initial_edge_index_a2a, initial_edge_attr_a2a = Transition.get_initial_decoder_context(scip_state=state, tqnet_version=self.tqnet_version)
                    # b. create context
                    transformer_decoder_context = self.policy_net.get_context(
                        torch.from_numpy(action['applied']), initial_edge_index_a2a, initial_edge_attr_a2a,
                        selection_order=action['selection_order'])
                    for k, v in transformer_decoder_context.items():
                        step_info[k] = v

                # get the next n-step state and q values. if the next state is terminal
                # return 0 as q_values (by convention)
                next_step_info = self.episode_history[step + n_steps] if step + n_steps < n_transitions else {}
                next_state = next_step_info.get('state_info', None)
                next_action = next_step_info.get('action_info', None)
                next_q_values = next_step_info.get('selected_q_values', None)

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
                is_empty_action = np.logical_not(action['selected_by_agent']).all()
                if self.empty_action_penalty is not None and is_empty_action:
                    reward = np.full_like(normalized_slack, fill_value=self.empty_action_penalty)

                transition = Transition.create(scip_state=state,
                                               action=action['applied'],
                                               info=step_info,
                                               reward=reward,
                                               scip_next_state=next_state,
                                               tqnet_version=self.tqnet_version
                                               )

                if self.use_per:
                    # todo - compute initial priority for PER based on the policy q_values.
                    #        compute the TD error for each action in the current state as we do in sgd_step,
                    #        and then take the norm of the resulting cut-wise TD-errors as the initial priority
                    selected_action = torch.from_numpy(action['selected_by_agent']).unsqueeze(1).long()  # cut-wise action
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
        for info in self.episode_history:
            action = info['action_info']
            normalized_slack = action['normalized_slack']
            # todo: verify with Aleks - consider slack < 1e-10 as zero
            approximately_zero = np.abs(normalized_slack) < 1e-10
            normalized_slack[approximately_zero] = 0

            applied = action['applied']
            is_active = normalized_slack[applied] == 0
            active_applied_ratio.append(sum(is_active)/sum(applied) if sum(applied) > 0 else 0)
            applied_available_ratio.append(sum(applied)/len(applied) if len(applied) > 0 else 0)
            # if self.demonstration_episode: todo verification
            accuracy_list.append(np.mean(action['selected_by_scip'] == action['selected_by_agent']))
            f1_score_list.append(f1_score(action['selected_by_scip'], action['selected_by_agent']))

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
            # test_stats_buffer uses for determining the best model performance.
            # if we ignore_test_early_stop, then we don't consider episodes which terminated due to branching
            if not (self.node_limit_reached and self.hparams.get('ignore_test_early_stop', False)):
                self.test_stats_buffer[stats_folder + 'db_auc_imp'].append(db_auc/self.baseline['rootonly_stats'][self.scip_seed]['db_auc'])
                self.test_stats_buffer[stats_folder + 'gap_auc_imp'].append(gap_auc/self.baseline['rootonly_stats'][self.scip_seed]['gap_auc'])
                self.test_stats_buffer['db_auc'].append(db_auc)
                self.test_stats_buffer['gap_auc'].append(gap_auc)

        # if self.demonstration_episode:  todo verification
        self.tmp_stats_buffer['Demonstrations/accuracy'] += accuracy_list
        self.tmp_stats_buffer['Demonstrations/f1_score'] += f1_score_list
        # store performance for tracking best models, ignoring bad outliers (e.g branching occured)
        if not self.node_limit_reached or self.hparams.get('ignore_test_early_stop', False):
            self.test_perf_list.append(db_auc if self.dqn_objective == 'db_auc' else gap_auc)
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
            # is_demonstration = np.array([t.is_demonstration for t in transitions], dtype=np.bool)
            # # is_demonstration = idxes < self.hparams.get('replay_buffer_n_demonstrations', 0)
            # # sort demonstration transitions first:
            # argsort_demonstrations_first = is_demonstration.argsort()[::-1]
            # transitions = [transitions[idx] for idx in argsort_demonstrations_first]
            # weights, idxes, data_ids = weights[argsort_demonstrations_first], idxes[argsort_demonstrations_first], data_ids[argsort_demonstrations_first]
            # is_demonstration = is_demonstration[argsort_demonstrations_first]
            batch, weights, idxes, data_ids, is_demonstration, demonstration_batch = self.preprocess_batch(transitions, weights, idxes, data_ids)
            new_priorities = self.sgd_step(batch=batch,
                                           importance_sampling_correction_weights=weights,
                                           is_demonstration=is_demonstration,
                                           demonstration_batch=demonstration_batch)
            # update priorities
            self.memory.update_priorities(idxes, new_priorities, data_ids)

        else:
            # learning from demonstration is disabled with simple replay buffer.
            # todo - update and batch before sgd_step
            transitions = self.memory.sample(self.batch_size)
            batch = Transition.create_batch(transitions)
            self.sgd_step(batch)

        self.num_param_updates += 1
        if self.num_sgd_steps_done % self.hparams.get('target_update_interval', 1000) == 0:
            self.update_target()

    def preprocess_batch(self, transitions, weights, idxes, data_ids):
        # sort demonstration transitions first:
        is_demonstration = np.array([t.is_demonstration for t in transitions], dtype=np.bool)
        argsort_demonstrations_first = is_demonstration.argsort()[::-1]
        transitions = [transitions[idx] for idx in argsort_demonstrations_first]
        weights, idxes, data_ids = weights[argsort_demonstrations_first], idxes[argsort_demonstrations_first], data_ids[argsort_demonstrations_first]
        is_demonstration = is_demonstration[argsort_demonstrations_first]

        # create pytorch-geometric batch
        batch = Transition.create_batch(transitions)

        # prepare demonstration data if any
        if is_demonstration.any():
            # filter non demonstration data
            n_demonstrations = sum(is_demonstration)  # number of demonstration transitions in batch
            # remove from edge_index (and corresponding edge_attr) edges which reference cuts of index greater than
            max_demonstration_edge_index = sum(batch.x_a_batch < n_demonstrations) - 1
            demonstration_edges = batch.demonstration_context_edge_index[0, :] < max_demonstration_edge_index

            # demonstration_context_edge_index = batch.demonstration_context_edge_index[:, demonstration_edges]
            # demonstration_context_edge_attr = batch.demonstration_context_edge_attr[demonstration_edges, :]
            # demonstration_action = batch.demonstration_action[batch.demonstration_action_batch < n_demonstrations]
            # demonstration_idx = batch.demonstration_idx[batch.demonstration_idx_batch < n_demonstrations]
            # demonstration_conv_aggr_out_idx = batch.demonstration_conv_aggr_out_idx[demonstration_edges]
            # demonstration_encoding_broadcast = batch.demonstration_encoding_broadcast[batch.demonstration_idx_batch < n_demonstrations]
            # demonstration_action_batch = batch.demonstration_action_batch[batch.demonstration_action_batch < n_demonstrations]

            demonstration_batch = DemonstrationBatch(
                batch.demonstration_context_edge_index[:, demonstration_edges],
                batch.demonstration_context_edge_attr[demonstration_edges, :],
                batch.demonstration_action[batch.demonstration_action_batch < n_demonstrations],
                batch.demonstration_idx[batch.demonstration_idx_batch < n_demonstrations],
                batch.demonstration_conv_aggr_out_idx[demonstration_edges],
                batch.demonstration_encoding_broadcast[batch.demonstration_idx_batch < n_demonstrations],
                batch.demonstration_action_batch[batch.demonstration_action_batch < n_demonstrations])
        else:
            demonstration_batch = None
        return batch, torch.from_numpy(weights), idxes, data_ids, is_demonstration, demonstration_batch

    def sgd_step(self, batch, importance_sampling_correction_weights=None, is_demonstration=None, demonstration_batch=None):
        """ implement the basic DQN optimization step """

        # # old replay buffer returned transitions as separated Transition objects
        # # todo - batch before calling sgd_step to save GPU time on distributed learner
        # batch = Transition.create_batch(transitions).to(self.device)  #, follow_batch=['x_c', 'x_v', 'x_a', 'ns_x_c', 'ns_x_v', 'ns_x_a']).to(self.device)
        batch = batch.to(self.device)
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
            demonstration_loss = self.compute_demonstration_loss(batch, policy_output['cut_encoding'], is_demonstration, demonstration_batch, importance_sampling_correction_weights)
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
        # todo: TQNet v3:
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
        # broadcast the aggregated next state q_values to cut-level graph wise, to bootstrap the cut-level rewards
        target_next_q_values_broadcast = max_target_next_q_values_aggr[batch.x_a_batch]

        # now compute the cut-level target
        reward_batch = batch.r
        target_q_values = reward_batch + (self.gamma ** self.nstep_learning) * target_next_q_values_broadcast

        # Compute Huber loss
        # todo - support importance sampling correction - double check
        if self.use_per:
            # broadcast each transition importance sampling weight to all its cut-level losses
            importance_sampling_correction_weights = importance_sampling_correction_weights.to(self.device)[batch.x_a_batch]
            # multiply cut-level loss and importance sampling correction weight, and average
            n_step_loss = (importance_sampling_correction_weights * F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1), reduction='none')).mean()
        else:
            # generate equal weights for all losses
            n_step_loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))

        # loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1)) - original pytorch example
        self.n_step_loss_moving_avg = 0.95 * self.n_step_loss_moving_avg + 0.05 * n_step_loss.detach().cpu().numpy()

        # combine all losses (DQN and demonstration loss)
        loss = self.hparams.get('n_step_loss_coef', 1.0) * n_step_loss + self.hparams.get('demonstration_loss_coef', 0.5) * demonstration_loss

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
            # (to compute p,q norm take power p and compute sqrt q)
            td_error_l2_norm = torch.sqrt(scatter_add(td_error ** 2, batch.x_a_batch, # target index of each element in source
                                                      dim=0,                          # scattering dimension
                                                      dim_size=self.batch_size))      # output tensor size in dim after scattering

            new_priorities = td_error_l2_norm.squeeze().cpu().numpy()

            return new_priorities
        else:
            return None

    def compute_demonstration_loss(self, batch, cut_encoding_batch, is_demonstration, demonstration_batch, importance_sampling_correction_weights):
        # todo - continue here. verify batching and everything.
        # filter non demonstration data
        n_demonstrations = sum(is_demonstration)  # number of demonstration transitions in batch
        cut_encoding_batch = cut_encoding_batch[batch.x_a_batch < n_demonstrations]
        # remove from edge_index (and corresponding edge_attr) edges which reference cuts of index greater than
        # max_demonstration_edge_index = cut_encoding_batch.shape[0] - 1
        assert demonstration_batch.context_edge_index.max() == cut_encoding_batch.shape[0] - 1
        # demonstration_edges = batch.demonstration_context_edge_index[0, :] < max_demonstration_edge_index
        # demonstration_context_edge_index = batch.demonstration_context_edge_index[:, demonstration_edges]
        # demonstration_context_edge_attr = batch.demonstration_context_edge_attr[demonstration_edges, :]
        # demonstration_action = batch.demonstration_action[batch.demonstration_action_batch < n_demonstrations]
        # demonstration_idx = batch.demonstration_idx[batch.demonstration_idx_batch < n_demonstrations]
        # demonstration_conv_aggr_out_idx = batch.demonstration_conv_aggr_out_idx[demonstration_edges]
        # demonstration_encoding_broadcast = batch.demonstration_encoding_broadcast[batch.demonstration_idx_batch < n_demonstrations]
        # demonstration_action_batch = batch.demonstration_action_batch[batch.demonstration_action_batch < n_demonstrations]
        #
        # demonstration_batch = DemonstrationBatch(
        #     demonstration_context_edge_index,
        #     demonstration_context_edge_attr,
        #     demonstration_action,
        #     demonstration_idx,
        #     demonstration_conv_aggr_out_idx,
        #     demonstration_encoding_broadcast,
        #     demonstration_action_batch)

        demonstration_context_edge_index = demonstration_batch[0].to(self.device)
        demonstration_context_edge_attr = demonstration_batch[1].to(self.device)
        demonstration_action = demonstration_batch[2].to(self.device)
        demonstration_idx = demonstration_batch[3].to(self.device)
        demonstration_conv_aggr_out_idx = demonstration_batch[4].to(self.device)
        demonstration_encoding_broadcast = demonstration_batch[5].to(self.device)
        demonstration_action_batch = demonstration_batch[6].to(self.device)

        # predict cut-level q values
        cut_decoding = self.policy_net.decoder_conv.conv_demonstration(
            x=cut_encoding_batch,
            edge_index=demonstration_context_edge_index,
            edge_attr=demonstration_context_edge_attr,
            aggr_idx=demonstration_conv_aggr_out_idx,
            encoding_broadcast=demonstration_encoding_broadcast
        )
        q_values = self.policy_net.q(cut_decoding)
        # average "discard" q_values
        discard_q_values = scatter_mean(q_values[:, 0], demonstration_idx, dim=0)
        assert discard_q_values.shape[0] == demonstration_idx[-1] + 1

        # interleave the "select" and "discard" q values
        pooled_q_values = torch.empty(size=(q_values.shape[0] + discard_q_values.shape[0],), dtype=torch.float32, device=self.device)
        pooled_q_values[torch.arange(q_values.shape[0], device=self.device) + demonstration_idx] = q_values[:, 1]  # "select" q_values
        discard_idxes = scatter_add(torch.ones_like(demonstration_idx, device=self.device), demonstration_idx, dim=0).cumsum(0) + torch.arange(discard_q_values.shape[0], device=self.device)
        pooled_q_values[discard_idxes] = discard_q_values

        # add large margin
        large_margin = torch.full_like(pooled_q_values,
                                       fill_value=self.hparams.get('demonstration_large_margin', 0.1),
                                       device=self.device)
        large_margin[demonstration_action] = 0
        q_plus_l = pooled_q_values + large_margin

        # compute demonstration loss J_E = max [Q(s,a) + large margin] - Q(s, a_E)
        q_a_E = pooled_q_values[demonstration_action]
        # todo - extend demonstration_idx with the "discard" entries, then take max
        demonstration_idx_ext = torch.empty_like(q_plus_l, dtype=torch.long, device=self.device)
        demonstration_idx_ext[torch.arange(q_values.shape[0], device=self.device) + demonstration_idx] = demonstration_idx
        demonstration_idx_ext[discard_idxes] = torch.arange(demonstration_idx[-1] + 1, device=self.device)

        max_q_plus_l, _ = scatter_max(q_plus_l, demonstration_idx_ext, dim=0)
        losses = max_q_plus_l - q_a_E

        # broadcast transition-level importance sampling correction weights to actions
        weights_broadcasted = importance_sampling_correction_weights[demonstration_action_batch].to(self.device)
        loss = (losses * weights_broadcasted).mean()

        # log demonstration loss moving average
        self.demonstration_loss_moving_avg = 0.95 * self.demonstration_loss_moving_avg + 0.05 * loss.detach().cpu().numpy()
        return loss

    # done
    def update_target(self):
        # Update the target network, copying all weights and biases in DQN
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.print(f'target net updated (sgd step = {self.num_sgd_steps_done})')

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

        for cut_name, selected in zip(cut_names, self.prev_action['selected_by_agent']):
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
    def log_stats(self, save_best=False, plot_figures=False, global_step=None, info={}, log_directly=True):
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
        Tracking the global_step is essential for "resume".
        TODO adapt function to run on learner and workers separately.
            learner - plot loss
            worker - plot auc etc.
            test_worker - plot valid/test auc frac and figures
            in single thread - these are all true.
            need to separate workers' logdir in the distributed main script
        """
        # dictionary of all metrics to log
        log_dict = {}
        actor_name = '_' + self.print_prefix.replace('[', '').replace(']', '').replace(' ', '_') if len(self.print_prefix) > 0 else ''
        # actor_name = self.print_prefix.replace('[', '').replace(']', '').replace(' ', '') + '_'
        if global_step is None:
            global_step = self.num_param_updates

        print(self.print_prefix, f'Global step: {global_step} | {self.dataset_name}\t|', end='')
        cur_time_sec = time() - self.start_time + self.walltime_offset

        if self.is_tester:
            if plot_figures:
                self.decorate_figures()  # todo replace with wandb plot line

            if save_best:
                # todo check if ignoring outliers is a good idea
                # perf = np.mean(self.tmp_stats_buffer[self.dqn_objective])
                perf = np.mean(self.test_perf_list)
                if perf > self.best_perf[self.dataset_name]:
                    self.best_perf[self.dataset_name] = perf
                    self.save_checkpoint(filepath=os.path.join(self.run_dir, f'best_{self.dataset_name}_checkpoint.pt'))
                    self.save_figures(filename_suffix=f'best_{self.num_param_updates}')
                    # save full test stats dict
                    with open(os.path.join(self.run_dir, f'best_{self.dataset_name}_test_stats.pkl'), 'wb') as f:
                        pickle.dump(self.test_stats_dict[self.dataset_name], f)

            # add episode figures (for validation and test sets only)
            if plot_figures:
                for figname in self.figures['fignames']: # todo replace with wandb plot line. in the meanwhile use wandb Image
                    if log_directly:
                        log_dict[figname + '/' + self.dataset_name] = self.figures[figname]['fig']
                    else:
                        # in order to send to apex logger, we should serialize the image as numpy array.
                        # so convert first to numpy array
                        fig_rgb = get_img_from_fig(self.figures[figname]['fig'], dpi=300)
                        # and store with a label 'fig' to decode on the logger side
                        log_dict[figname + '/' + self.dataset_name] = ('fig', fig_rgb)

            # plot dualbound and gap auc improvement over the baseline (for validation and test sets only)
            for k, vals in self.test_stats_buffer.items():
                if len(vals) > 0:
                    avg = np.mean(vals)
                    # std = np.std(vals)
                    print('{}: {:.4f} | '.format(k, avg), end='')
                    self.test_stats_buffer[k] = []
                    log_dict[self.dataset_name + '/' + k + actor_name] = avg
                    # log_dict[self.dataset_name + '/' + k + '_std' + actor_name] = std

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
                    log_dict[self.dataset_name + '/' + k + actor_name] = avg
                    log_dict[self.dataset_name + '/' + k + '_std' + actor_name] = std

                self.sanity_check_stats['n_original_cuts'] = []
                self.sanity_check_stats['n_duplicated_cuts'] = []
                self.sanity_check_stats['n_weak_cuts'] = []

        if self.is_worker or self.is_tester:
            # plot normalized dualbound and gap auc
            for k, vals in self.tmp_stats_buffer.items():
                if len(vals) == 0:
                    continue
                avg = np.mean(vals)
                # std = np.std(vals)
                print('{}: {:.4f} | '.format(k, avg), end='')
                log_dict[self.dataset_name + '/' + k + actor_name] = avg
                # log_dict[self.dataset_name + '/' + k + '_std' + actor_name] = std
                self.tmp_stats_buffer[k] = []

        if self.is_learner:
            # log the average loss of the last training session
            print('{}-step Loss: {:.4f} | '.format(self.nstep_learning, self.n_step_loss_moving_avg), end='')
            print('Demonstration Loss: {:.4f} | '.format(self.demonstration_loss_moving_avg), end='')
            print(f'SGD Step: {self.num_sgd_steps_done} | ', end='')
            # todo wandb
            log_dict['Nstep_Loss'] = self.n_step_loss_moving_avg
            log_dict['Demonstration_Loss'] = self.demonstration_loss_moving_avg

        if log_directly:
            # todo wandb modify log dict keys with actor_name, or maybe agging is better?
            wandb.log(log_dict, step=global_step)

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

        self.test_perf_list = []  # reset for the next testset

        return global_step, log_dict

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
            dqn = self.episode_stats
            bsl_0 = self.baseline['rootonly_stats'][self.scip_seed]
            bsl_1 = self.baseline['10_random'][self.scip_seed]
            bsl_2 = self.baseline['10_most_violated'][self.scip_seed]
            bsl_stats = self.datasets[self.dataset_name]['stats']
            # bsl_lpiter, bsl_db, bsl_gap = bsl_0['lp_iterations'], bsl_0['dualbound'], bsl_0['gap']
            # dqn_lpiter, dqn_db, dqn_gap = self.episode_stats['lp_iterations'], self.episode_stats['dualbound'], self.episode_stats['gap']

            # set labels for the last subplot
            if row == self.figures['nrows'] - 1 and col == self.figures['ncols'] - 1:
                dqn_db_auc_avg_without_early_stop = np.mean(self.test_stats_buffer['db_auc'])
                dqn_gap_auc_avg_without_early_stop = np.mean(self.test_stats_buffer['gap_auc'])
                dqn_db_auc_avg = np.mean(self.tmp_stats_buffer['db_auc'])
                dqn_gap_auc_avg = np.mean(self.tmp_stats_buffer['gap_auc'])
                db_labels = ['DQN {:.4f}({:.4f})'.format(dqn_db_auc_avg, dqn_db_auc_avg_without_early_stop),
                             'SCIP {:.4f}'.format(self.datasets[self.dataset_name]['stats']['rootonly_stats']['db_auc_avg']),
                             '10 RANDOM {:.4f}'.format(self.datasets[self.dataset_name]['stats']['10_random']['db_auc_avg']),
                             '10 MOST VIOLATED {:.4f}'.format(self.datasets[self.dataset_name]['stats']['10_most_violated']['db_auc_avg']),
                             'OPTIMAL'
                             ]
                gap_labels = ['DQN {:.4f}({:.4f})'.format(dqn_gap_auc_avg, dqn_gap_auc_avg_without_early_stop),
                              'SCIP {:.4f}'.format(self.datasets[self.dataset_name]['stats']['rootonly_stats']['gap_auc_avg']),
                              '10 RANDOM {:.4f}'.format(self.datasets[self.dataset_name]['stats']['10_random']['gap_auc_avg']),
                              '10 MOST VIOLATED {:.4f}'.format(self.datasets[self.dataset_name]['stats']['10_most_violated']['gap_auc_avg']),
                              'OPTIMAL'
                              ]
            else:
                db_labels = [None] * 5
                gap_labels = [None] * 5

            for db_label, gap_label, color, lpiter, db, gap in zip(db_labels, gap_labels,
                                                                   ['b', 'g', 'y', 'c', 'k'],
                                                                   [dqn['lp_iterations'], bsl_0['lp_iterations'], bsl_1['lp_iterations'], bsl_2['lp_iterations'], [0, self.lp_iterations_limit]],
                                                                   [dqn['dualbound'], bsl_0['dualbound'], bsl_1['dualbound'], bsl_2['dualbound'], [self.baseline['optimal_value']]*2],
                                                                   [dqn['gap'], bsl_0['gap'], bsl_1['gap'], bsl_2['gap'], [0, 0]]
                                                                   ):
                if lpiter[-1] < self.lp_iterations_limit:
                    # extend curve to the limit
                    lpiter = lpiter + [self.lp_iterations_limit]
                    db = db + db[-1:]
                    gap = gap + gap[-1:]
                assert lpiter[-1] == self.lp_iterations_limit
                # plot dual bound and gap, marking early stops with red borders
                ax = self.figures['Dual_Bound_vs_LP_Iterations']['axes'][row, col]
                ax.plot(lpiter, db, color, label=db_label)
                if self.terminal_state == 'NODE_LIMIT':
                    for spine in ax.spines.values():
                        spine.set_edgecolor('red')
                ax = self.figures['Gap_vs_LP_Iterations']['axes'][row, col]
                ax.plot(lpiter, gap, color, label=gap_label)
                if self.terminal_state == 'NODE_LIMIT':
                    for spine in ax.spines.values():
                        spine.set_edgecolor('red')

            # if dqn_lpiter[-1] < self.lp_iterations_limit:
            #     # extend curve to the limit
            #     dqn_lpiter = dqn_lpiter + [self.lp_iterations_limit]
            #     dqn_db = dqn_db + dqn_db[-1:]
            #     dqn_gap = dqn_gap + dqn_gap[-1:]
            # if bsl_lpiter[-1] < self.lp_iterations_limit:
            #     # extend curve to the limit
            #     bsl_lpiter = bsl_lpiter + [self.lp_iterations_limit]
            #     bsl_db = bsl_db + bsl_db[-1:]
            #     bsl_gap = bsl_gap + bsl_gap[-1:]
            # assert dqn_lpiter[-1] == self.lp_iterations_limit
            # assert bsl_lpiter[-1] == self.lp_iterations_limit
            # # plot dual bound
            # ax = self.figures['Dual_Bound_vs_LP_Iterations']['axes'][row, col]
            # ax.plot(dqn_lpiter, dqn_db, 'b', label='DQN')
            # ax.plot(bsl_lpiter, bsl_db, 'r', label='SCIP default')
            # ax.plot([0, self.lp_iterations_limit], [self.baseline['optimal_value']]*2, 'k', label='optimal value')
            # # plot gap
            # ax = self.figures['Gap_vs_LP_Iterations']['axes'][row, col]
            # ax.plot(dqn_lpiter, dqn_gap, 'b', label='DQN')
            # ax.plot(bsl_lpiter, bsl_gap, 'r', label='SCIP default')
            # ax.plot([0, self.lp_iterations_limit], [0, 0], 'k', label='optimal gap')

        # plot imitation performance bars
        if 'Similarity_to_SCIP' in self.figures.keys():
            true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
            for info in self.episode_history:
                scip_action = info['action_info']['selected_by_scip']
                agent_action = info['action_info']['selected_by_agent']
                true_pos += sum(scip_action[scip_action == 1] == agent_action[scip_action == 1])
                true_neg += sum(scip_action[scip_action == 0] == agent_action[scip_action == 0])
                false_pos += sum(scip_action[agent_action == 1] != agent_action[agent_action == 1])
                false_neg += sum(scip_action[agent_action == 0] != agent_action[agent_action == 0])
            total_ncuts = true_pos + true_neg + false_pos + false_neg
            rects = []
            ax = self.figures['Similarity_to_SCIP']['axes'][row, col]
            rects += ax.bar(-0.3, true_pos / total_ncuts, width=0.2, label='true pos')
            rects += ax.bar(-0.1, true_neg / total_ncuts, width=0.2, label='true neg')
            rects += ax.bar(+0.1, false_pos / total_ncuts, width=0.2, label='false pos')
            rects += ax.bar(+0.3, false_neg / total_ncuts, width=0.2, label='false neg')

            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.2f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            ax.set_xticks([], [])  # disable x ticks

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
                ax = self.figures[figname]['axes'][-1, -1]
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=1, borderaxespad=0.)

    # done
    def save_figures(self, filename_suffix=None):
        for figname in ['Dual_Bound_vs_LP_Iterations', 'Gap_vs_LP_Iterations']:
            # save png
            fname = f'{self.dataset_name}_{figname}'
            if filename_suffix is not None:
                fname = fname + '_' + filename_suffix
            fname += '.png'
            fpath = os.path.join(self.run_dir, fname)
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
            self.save_checkpoint(filepath=os.path.join(self.run_dir, f'best_{self.dataset_name}_checkpoint.pt'))

    # done
    def load_checkpoint(self, filepath=None):
        if filepath is None:
            filepath = self.checkpoint_filepath
        if not os.path.exists(filepath):
            print(self.print_prefix, 'Checkpoint file does not exist! starting from scratch.')
            return
        checkpoint = torch.load(filepath, map_location=self.device)
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
        print(self.print_prefix, 'Loaded checkpoint from: ', filepath)

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
            datasets[dataset_name]['datadir'] = os.path.join(
                hparams['datadir'], dataset['dataset_name'],
                f"barabasi-albert-nmin{dataset['graph_size']['min']}-nmax{dataset['graph_size']['max']}-m{dataset['barabasi_albert_m']}-weights-{dataset['weights']}-seed{dataset['seed']}")

            # read all graphs with their baselines from disk
            dataset['instances'] = []
            for filename in tqdm(os.listdir(datasets[dataset_name]['datadir']), desc=f'{self.print_prefix}Loading {dataset_name}'):
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

        # for the validation and test datasets compute average performance of all baselines:
        for dataset_name, dataset in datasets.items():
            if dataset_name[:8] == 'trainset':
                continue
            dataset['stats'] = {}
            for bsl in ['rootonly_stats', '10_random', '10_most_violated']:
                db_auc_list = []
                gap_auc_list = []
                for (_, baseline) in dataset['instances']:
                    optimal_value = baseline['optimal_value']
                    for scip_seed in dataset['scip_seed']:
                        # align curves to lp_iterations_limit
                        tmp_stats = {}
                        for k, v in baseline[bsl][scip_seed].items():
                            if k != 'lp_iterations' and len(v) > 0:
                                aligned_lp_iterations, aligned_v = truncate(t=baseline[bsl][scip_seed]['lp_iterations'], ft=v,
                                                                            support=dataset['lp_iterations_limit'], interpolate=type(v[0]) == float)
                                tmp_stats[k] = aligned_v
                                tmp_stats['lp_iterations'] = aligned_lp_iterations
                        # override with aligned stats
                        baseline[bsl][scip_seed] = tmp_stats

                        dualbound = baseline[bsl][scip_seed]['dualbound']
                        gap = baseline[bsl][scip_seed]['gap']
                        lpiter = baseline[bsl][scip_seed]['lp_iterations']
                        db_auc = sum(get_normalized_areas(t=lpiter, ft=dualbound, t_support=dataset['lp_iterations_limit'], reference=optimal_value))
                        gap_auc = sum(get_normalized_areas(t=lpiter, ft=gap, t_support=dataset['lp_iterations_limit'], reference=0))
                        baseline[bsl][scip_seed]['db_auc'] = db_auc
                        baseline[bsl][scip_seed]['gap_auc'] = gap_auc
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
            self.trainset = self.datasets['trainset_20_30']
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
        if self.hparams.get('resume', False):
            self.load_checkpoint()
            # initialize prioritized replay buffer internal counters, to continue beta from the point it was
            if self.use_per:
                self.memory.num_sgd_steps_done = self.num_sgd_steps_done

    def on_nodebranched_event(self):
        # self.print('BRANCHING EVENT')
        self.node_limit_reached = True

    def on_lpsolved_event(self):
        # todo verification
        # self.print('LPSOLVED EVENT')
        # if self.prev_action is not None and self.prev_action.get('normalized_slack', None) is None and self.model.getStage() == SCIP_STAGE.SOLVING:
        #     self.model.getState(query=self.prev_action)
        # self._update_episode_stats()
        # pass
        # todo - for some reason we get at some point
        # ): /home/avrech/scipoptsuite-6.0.2-avrech/scip/src/scip/lp.c:3899: SCIPcolGetRedcost: Assertion `lp->validsollp == stat->lpcount' failed.
        pass

    # done
    def execute_episode(self, G, baseline, lp_iterations_limit, dataset_name, scip_seed=None, demonstration_episode=False):
        # fix training scip_seed for debug purpose
        if self.training and self.hparams.get('fix_training_scip_seed'):
            scip_seed = self.hparams['fix_training_scip_seed']

        # create a SCIP model for G, and disable default cuts
        hparams = self.hparams
        model, x, y = maxcut_mccormic_model(G, use_general_cuts=hparams.get('use_general_cuts', False))

        # include cycle inequalities separator with high priority
        cycle_sepa = MccormickCycleSeparator(G=G, x=x, y=y, name='MLCycles', hparams=hparams)
        model.includeSepa(cycle_sepa, 'MLCycles',
                          "Generate cycle inequalities for the MaxCut McCormick formulation",
                          priority=1000000, freq=1)

        # # include branching event handler
        branching_event = BranchingEventHdlr(on_nodebranched_event=self.on_nodebranched_event,
                                             on_lpsolved_event=self.on_lpsolved_event)
        model.includeEventhdlr(branching_event, "BranchingEventhdlr", "Catches NODEBRANCHED event")

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

        if self.hparams.get('debug_events'):
            debug_eventhdlr = DebugEvents()
            model.includeEventhdlr(debug_eventhdlr, "DebugEvents", "Catches LPSOLVED and ROWADDEDSEPA events")

        # gong! run episode
        model.optimize()

        # once episode is done
        trajectory = self.finish_episode()
        return trajectory

    # done
    def evaluate(self, datasets=None, ignore_eval_interval=False, log_directly=True, eval_testset=False):
        if datasets is None:
            datasets = self.datasets
        # evaluate the model on the validation and test sets
        if self.num_param_updates == 0:
            # wait until the model starts learning
            return None, None
        global_step = self.num_param_updates
        log_dict = {}
        # initialize cycle_stats first time
        if self.hparams.get('record_cycles', False) and self.cycle_stats is None:
            self.cycle_stats = {dataset_name: {inst_idx: {seed_idx: {}
                                                          for seed_idx in dataset['scip_seed']}
                                               for inst_idx in range(dataset['num_instances'])}
                                for dataset_name, dataset in datasets.items() if 'trainset' not in dataset_name}

        self.set_eval_mode()
        for dataset_name, dataset in datasets.items():
            if 'trainset' in dataset_name or (not eval_testset and 'testset' in dataset_name):
                continue
            if ignore_eval_interval or global_step % dataset['eval_interval'] == 0:
                fignames = ['Dual_Bound_vs_LP_Iterations', 'Gap_vs_LP_Iterations', 'Similarity_to_SCIP']
                self.init_figures(fignames,
                                  nrows=dataset['num_instances'],
                                  ncols=len(dataset['scip_seed']),
                                  col_labels=[f'Seed={seed}' for seed in dataset['scip_seed']],
                                  row_labels=[f'inst {inst_idx}' for inst_idx in
                                              range(dataset['num_instances'])])
                self.test_stats_dict[dataset_name] = {}
                for inst_idx, (G, baseline) in enumerate(dataset['instances']):
                    self.test_stats_dict[dataset_name][inst_idx] = {}
                    for seed_idx, scip_seed in enumerate(dataset['scip_seed']):
                        self.cur_graph = f'{dataset_name} graph {inst_idx} seed {scip_seed}'
                        if self.hparams.get('verbose', 0) == 2:
                            print('##################################################################################')
                            print(f'dataset: {dataset_name}, inst: {inst_idx}, seed: {scip_seed}')
                            print('##################################################################################')
                        self.execute_episode(G, baseline, dataset['lp_iterations_limit'], dataset_name=dataset_name,
                                             scip_seed=scip_seed)
                        self.add_episode_subplot(inst_idx, seed_idx)
                        self.test_stats_dict[dataset_name][inst_idx][scip_seed] = {'stats': self.episode_stats,
                                                                                   'node_limit_reached': self.node_limit_reached}

                        # record cycles statistics
                        if self.hparams.get('record_cycles', False):
                            cycle_stats = {}
                            cycle_stats['recorded_cycles'] = self.cut_generator.recorded_cycles
                            cycle_stats['episode_stats'] = self.episode_stats
                            cycle_stats['dualbound_area'] = get_normalized_areas(t=self.episode_stats['lp_iterations'], ft=self.episode_stats['dualbound'], t_support=self.lp_iterations_limit, reference=self.baseline['optimal_value'])
                            cycle_stats['gap_area'] = get_normalized_areas(t=self.episode_stats['lp_iterations'], ft=self.episode_stats['gap'], t_support=self.lp_iterations_limit, reference=0)  # optimal gap is always 0
                            self.cycle_stats[dataset_name][inst_idx][scip_seed][global_step] = cycle_stats
                            self.cycle_stats[dataset_name][inst_idx]['G'] = G
                            self.cycle_stats[dataset_name][inst_idx]['baseline'] = baseline

                step, logs = self.log_stats(save_best='validset' in dataset_name, plot_figures=True, log_directly=log_directly)
                log_dict.update(logs)

        # if len(list(self.cycle_stats['validset_20_30'][0].values())[0]) >= 2:
        #     with open(os.path.join(self.logdir, f'global_step-{global_step}_last_100_evaluations_cycle_stats.pkl'), 'wb') as f:
        #         pickle.dump(self.cycle_stats, f)
        #     self.cycle_stats = None

        self.set_training_mode()
        return step, log_dict

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
            self.cur_graph = f'trainset graph {graph_idx}'
            if self.hparams.get('verbose', 0) == 2:
                print(f'############################# starting episode {i_episode} (graph index {graph_idx}) #############################')
            G, baseline = trainset['instances'][graph_idx]
            if hparams.get('debug', False):
                filename = os.listdir(trainset['datadir'])[graph_idx]
                filename = os.path.join(trainset['datadir'], filename)
                print(f'instance no. {graph_idx}, filename: {filename}')

            # execute episode and collect experience
            # todo scip_seed=562696653 - probing scip action increments lp iterations by 1, from 564 to 565 after probing ends. investigate
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

    def test(self):
        """ playground for testing """
        self.load_datasets()
        self.load_checkpoint(filepath='/home/avrech/learning2cut/experiments/dqn/results/exp5/24jo87jy/best_validset_90_100_checkpoint.pt')
        # focus test on
        dataset = self.datasets['validset_90_100']
        dataset['instances'] = [dataset['instances'][idx] for idx in [3, 6]]
        dataset['scip_seed'] = [176]

        datasets = {'validset_90_100': dataset}
        stat = self.evaluate(datasets=datasets, ignore_eval_interval=True, log_directly=False)

    # done
    def get_model(self):
        """ useful for distributed actors """
        return self.policy_net

    def print(self, expr):
        print(self.print_prefix, expr)

