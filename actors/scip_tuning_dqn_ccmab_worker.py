""" Worker class copied and modified from https://github.com/cyoon1729/distributedRL """
from sklearn.metrics import f1_score
from pyscipopt import Sepa, SCIP_RESULT
import numpy as np
from utils.data import Transition
import torch
from utils.functions import get_normalized_areas
from collections import namedtuple
import matplotlib as mpl
from actors.scip_tuning_dqn_worker import SCIPTuningDQNWorker
mpl.rc('figure', max_open_warning=0)


StateActionContext = namedtuple('StateActionQValuesContext', ('scip_state', 'action', 'q_values', 'transformer_context'))


class SCIPTuningDQNCCMABWorker(Sepa, SCIPTuningDQNWorker):
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
        super(SCIPTuningDQNCCMABWorker, self).__init__(worker_id, hparams, use_gpu, gpu_id, **kwargs)
        self.name = 'SCIP Tuning DQN CCMAB Worker'
        self.first_round_action_info = None
        self.first_round = True

    def init_episode(self, G, x, lp_iterations_limit, cut_generator=None, instance_info=None, dataset_name='trainset25', scip_seed=None):
        super().init_episode(G, x, lp_iterations_limit, cut_generator, instance_info, dataset_name, scip_seed)
        self.first_round_action_info = None
        self.first_round = True

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

        if available_cuts['ncuts'] > 0:
            # select an action, and get q_values for PER
            assert not np.any(np.isnan(cur_state['C'])) and not np.any(np.isnan(cur_state['A'])), f'Nan values in state features\ncur_graph = {self.cur_graph}\nA = {cur_state["A"]}\nC = {cur_state["C"]}'
            if self.first_round:
                self.first_round_action_info = action_info = self._select_action(cur_state)
                self.first_round = False
            else:
                action_info = self.first_round_action_info

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

            # store the current state and action for
            # computing later the n-step rewards and the (s,a,r',s') transitions
            self.episode_history.append(info)
            self.prev_action = available_cuts
            self.prev_state = cur_state
            self.stats_updated = False  # mark false to record relevant stats after this action will make effect

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
        dualbound_area = get_normalized_areas(t=lp_iterations, ft=dualbound, t_support=lp_iterations_limit, reference=self.instance_info['optimal_value'])
        gap_area = get_normalized_areas(t=lp_iterations, ft=gap, t_support=lp_iterations_limit, reference=0)  # optimal gap is always 0
        if self.dqn_objective == 'db_auc':
            objective_area = dualbound_area
        elif self.dqn_objective == 'gap_auc':
            objective_area = gap_area
        else:
            raise NotImplementedError

        trajectory = []
        if self.training:
            # # compute reward and store a transition (s_0, a_0, r_0, s_T)
            # n_transitions = 1  # len(self.episode_history)
            # n_steps = self.nstep_learning
            # # gammas = self.gamma**np.arange(n_steps).reshape(-1, 1)  # [1, gamma, gamma^2, ... gamma^{n-1}]
            # # indices = np.arange(n_steps).reshape(1, -1) + np.arange(n_transitions).reshape(-1, 1)  # indices of sliding windows
            # # # in case of n_steps > 1, pad objective_area with zeros only for avoiding overflow
            # # max_index = np.max(indices)
            # # if max_index >= len(objective_area):
            # #     objective_area = np.pad(objective_area, (0, max_index+1-len(objective_area)), 'constant', constant_values=0)
            # # take sliding windows of width n_step from objective_area
            # n_step_rewards = objective_area[indices]
            # # compute returns
            # # R[t] = r[t] + gamma * r[t+1] + ... + gamma^(n-1) * r[t+n-1]
            reward = sum(objective_area)  # n_step_rewards @ gammas
            if self.hparams.get('dqn_objective_norm', False) and self.hparams['fix_training_scip_seed'] == 223:
                reward /= self.instance_info['baselines']['default'][223][self.dqn_objective]

            bootstrapping_q = []
            discarded = False
            # assign rewards and store transitions (s,a,r,s')

            step_info = self.episode_history[0]
            state, action_info, q_values = step_info['state_info'], step_info['action_info'], step_info['selected_q_values']

            # in CCMAB the next state is always the terminal state
            # return 0 as q_values (by convention)
            next_state = None

            normalized_slack = action_info['normalized_slack']
            # todo: verify with Aleks - consider slack < 1e-10 as zero
            approximately_zero = np.abs(normalized_slack) < self.hparams['slack_tol']
            normalized_slack[approximately_zero] = 0
            # assert (normalized_slack >= 0).all(), f'rhs slack variable is negative,{normalized_slack}'
            if (normalized_slack < 0).any():
                self.print(f'Warning: encountered negative RHS slack variable.\nnormalized_slack: {normalized_slack}\ndiscarding the rest of the episode\ncur_graph = {self.cur_graph}')
                discarded = True
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
                # next state is terminal, and its q_values are 0 by convention
                target_q_values = torch.from_numpy(reward)
                bootstrapping_q.append(0)
                td_error = torch.abs(q_values - target_q_values)
                td_error = torch.clamp(td_error, min=1e-8)
                initial_priority = torch.norm(td_error).item()  # default L2 norm
                trajectory.append((transition, initial_priority, False))
            else:
                trajectory.append(transition)

        # compute some stats and store in buffer
        discounted_rewards = [np.sum(dualbound_area)]
        selected_q_avg = [np.mean(self.episode_history[0].get('selected_q_values', torch.zeros((1,))).numpy())]
        selected_q_std = [np.std(self.episode_history[0].get('selected_q_values', torch.zeros((1,))).numpy())]

        active_applied_ratio = []
        applied_available_ratio = []
        accuracy_list, f1_score_list = [], []
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        q_avg, q_std = [], []
        for step_idx, info in enumerate(self.episode_history):
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
            # store for plotting later
            scip_action = info['action_info']['selected_by_scip']
            agent_action = info['action_info']['selected_by_agent']
            true_pos += sum(scip_action[scip_action == 1] == agent_action[scip_action == 1])
            true_neg += sum(scip_action[scip_action == 0] == agent_action[scip_action == 0])
            false_pos += sum(scip_action[agent_action == 1] != agent_action[agent_action == 1])
            false_neg += sum(scip_action[agent_action == 0] != agent_action[agent_action == 0])
            if step_idx == 0:
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
            self.training_stats['gap_auc_improvement'].append(gap_auc / self.instance_info['baselines']['default'][223]['gap_auc'])
            self.training_stats['active_applied_ratio'] += active_applied_ratio  # .append(np.mean(active_applied_ratio))
            self.training_stats['applied_available_ratio'] += applied_available_ratio  # .append(np.mean(applied_available_ratio))
            self.training_stats['accuracy'] += accuracy_list
            self.training_stats['f1_score'] += f1_score_list
            if not discarded:
                self.last_training_episode_stats['discounted_rewards'] = discounted_rewards
                self.last_training_episode_stats['selected_q_avg'] = selected_q_avg
                self.last_training_episode_stats['selected_q_std'] = selected_q_std

            stats = None
        else:
            stats = {**self.episode_stats,
                     'db_auc': db_auc,
                     'db_auc_improvement': db_auc / self.instance_info['baselines']['default'][self.scip_seed]['db_auc'],
                     'gap_auc': gap_auc,
                     'gap_auc_improvement': gap_auc / self.instance_info['baselines']['default'][self.scip_seed]['gap_auc'],
                     'active_applied_ratio': np.mean(active_applied_ratio),
                     'applied_available_ratio': np.mean(applied_available_ratio),
                     'accuracy': np.mean(accuracy_list),
                     'f1_score': np.mean(f1_score_list),
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

        return trajectory, stats
