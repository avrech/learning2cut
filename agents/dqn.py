from pyscipopt import  Sepa, SCIP_RESULT
from time import time
import numpy as np
from utils.data import get_transition
import os
import math
import random
from gnn.models import Qnet
import torch
import scipy as sp
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data.batch import Batch
from torch_scatter import scatter_mean, scatter_max
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from utils.functions import get_normalized_areas


class ReplayMemory(object):
    """
    Stores transitions of type utils.data.Transition in a single huge list
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            # extend the memory for storing the new transition
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(Sepa):
    def __init__(self, name='DQN', hparams={}):
        """
        Sample scip.Model state every time self.sepaexeclp is invoked.
        Store the generated data object in
        """
        self.name = name
        self.hparams = hparams

        # DQN stuff
        self.memory = ReplayMemory(hparams.get('memory_capacity', 1000000))
        self.device = torch.device("cuda" if torch.cuda.is_available() and hparams.get('use_gpu', False) else "cpu")
        self.batch_size = hparams.get('batch_size', 64)
        self.mini_batch_size = hparams.get('mini_batch_size', 8)
        self.n_sgd_epochs = hparams.get('n_sgd_epochs', 10)
        self.gamma = hparams.get('gamma', 0.999)
        self.eps_start = hparams.get('eps_start', 0.9)
        self.eps_end = hparams.get('eps_end', 0.05)
        self.eps_decay = hparams.get('eps_decay', 200)
        self.policy_net = Qnet(hparams=hparams).to(self.device)
        self.target_net = Qnet(hparams=hparams).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=hparams.get('lr', 0.001), weight_decay=hparams.get('weight_decay', 0.0001))
        # value aggregation method for the target Q values
        if hparams.get('value_aggr', 'mean') == 'max':
            self.aggr_func = scatter_max
        elif hparams.get('value_aggr', 'mean') == 'mean':
            self.aggr_func = scatter_mean
        self.nstep_learning = hparams.get('nstep_learning', 1)
        self.dqn_objective = hparams.get('dqn_objective', 'dualbound_integral')

        # training stuff
        self.steps_done = 0
        self.i_episode = 0
        self.training = True
        self.walltime_offset = 0
        self.start_time = time()
        self.last_time_sec = self.walltime_offset

        # file system paths
        self.logdir = hparams.get('logdir', 'results')
        self.checkpoint_filepath = os.path.join(self.logdir, 'checkpoint.pt')

        # instance specific data needed to be reset every episode
        self.G = None
        self.x = None
        self.y = None
        self.action = None
        self.prev_action = None
        self.prev_state = None
        self.state_action_pairs = []
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
        self.cut_generator = None
        self.nseparounds = 0
        self.dataset_name = 'trainset'  # or <easy/medium/hard>_<validset/testset>
        self.lp_iterations_limit = -1

        # logging
        self.writer = SummaryWriter(log_dir=os.path.join(self.logdir, 'tensorboard'))
        # todo compute fscore of p = nactive/napplied and q = nactive / (napplied + nstillviolated)
        # tmp buffer for holding each episode results until averaging and appending to experiment_stats
        self.tmp_stats_buffer = {'dualbound_integral': [], 'gap_integral': [], 'active_applied_ratio': [], 'applied_available_ratio': []}
        self.test_stats_buffer = {'db_int_improvement': [], 'gap_int_improvement': []}
        self.best_perf = {'easy_validset': -1000000, 'medium_validset': -1000000, 'hard_validset': -1000000}
        self.loss_moving_avg = 0

    # done
    def init_episode(self, G, x, y, lp_iterations_limit, cut_generator=None, baseline=None, dataset_name='trainset'):
        self.G = G
        self.x = x
        self.y = y
        self.baseline = baseline
        self.action = None
        self.prev_action = None
        self.prev_state = None
        self.state_action_pairs = []
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
        self.cut_generator = cut_generator
        self.nseparounds = 0
        self.dataset_name = dataset_name
        self.lp_iterations_limit = lp_iterations_limit

    # done
    def _select_action(self, scip_state):
        # transform scip_state into GNN data type
        batch = Batch().from_data_list([get_transition(scip_state)],
                                       follow_batch=['x_c', 'x_v', 'x_a', 'ns_x_c', 'ns_x_v', 'ns_x_a']).to(self.device)

        if self.training:
            # take epsilon-greedy action
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                            math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
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
                        x_a_batch=batch.x_a_batch
                    )
                    return q_values.max(1)[1]
            else:
                random_action = torch.randint_like(batch.a, low=0, high=2, dtype=torch.long)
                return random_action
        else:
            # in test time, take greedy action
            with torch.no_grad():
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
                    x_a_batch=batch.x_a_batch
                )
                return q_values.max(1)[1]

    # done
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # todo consider all features to parse correctly
        loader = DataLoader(
            transitions,
            batch_size=self.mini_batch_size,
            shuffle=True,
            follow_batch=['x_c', 'x_v', 'x_a', 'ns_x_c', 'ns_x_v', 'ns_x_a']
        )
        # batch = Batch().from_data_list(transitions, follow_batch=['x_c', 'x_v', 'x_a', 'nx_c', 'nx_v', 'nx_a']).to(self.device)
        for epoch in range(self.n_sgd_epochs):
            for batch in loader:
                batch = batch.to(self.device)
                action_batch = batch.a

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = self.policy_net(
                    x_c=batch.x_c,
                    x_v=batch.x_v,
                    x_a=batch.x_a,
                    edge_index_c2v=batch.edge_index_c2v,
                    edge_index_a2v=batch.edge_index_a2v,
                    edge_attr_c2v=batch.edge_attr_c2v,
                    edge_attr_a2v=batch.edge_attr_a2v,
                    edge_index_a2a=batch.edge_index_a2a,
                    edge_attr_a2a=batch.edge_attr_a2a,
                    x_a_batch=batch.x_a_batch
                )
                state_action_values = state_action_values.gather(1, action_batch.unsqueeze(1))
                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_terminal_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1)[0].
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was terminal.
                # The value of a state is computed as in BQN paper https://arxiv.org/pdf/1711.08946.pdf
                # next-state action-wise values
                next_state_action_wise_values = self.target_net(
                    x_c=batch.ns_x_c,
                    x_v=batch.ns_x_v,
                    x_a=batch.ns_x_a,
                    edge_index_c2v=batch.ns_edge_index_c2v,
                    edge_index_a2v=batch.ns_edge_index_a2v,
                    edge_attr_c2v=batch.ns_edge_attr_c2v,
                    edge_attr_a2v=batch.ns_edge_attr_a2v,
                    edge_index_a2a=batch.ns_edge_index_a2a,
                    edge_attr_a2a=batch.ns_edge_attr_a2a,
                    x_a_batch=batch.ns_x_a_batch
                ).max(1)[0].detach()
                # aggregate the action-wise values using mean or max,
                # and generate for each graph in the batch a single value
                next_state_values = self.aggr_func(next_state_action_wise_values,   # source vector
                                                   batch.ns_x_a_batch,              # target index of each element in source
                                                   dim=0,                           # scattering dimension
                                                   dim_size=self.mini_batch_size)   # output tensor size in dim after scattering
                # override with zeros the values of terminal states which are zero by convention
                next_state_values[batch.ns_terminal] = 0
                # scatter the next state values graph-wise to update all action-wise rewards
                next_state_values = next_state_values[batch.x_a_batch]

                # now compute the expected Q values for each action separately
                reward_batch = batch.r
                expected_state_action_values = (next_state_values * self.gamma ** self.nstep_learning) + reward_batch

                # Compute Huber loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                self.loss_moving_avg = 0.95*self.loss_moving_avg + 0.05*loss.detach().cpu().numpy()

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

        # don't forget to update target periodically in the outer loop

    def update_target(self):
        # Update the target network, copying all weights and biases in DQN
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # done
    def _do_dqn_step(self):
        """
        Here is the episode inner loop of DQN.
        This DQN implemention is most vanilla one,
        in which we sequentially
        1. get state
        2. select action
        3. get the next state and reward (in the next LP round, after the LP solver solved for our cuts)
        4. store transition in memory
        5. optimize the policy on the replay memory
        When the instance is solved, the episode ends, and we start solving another instance,
        continuing with the latest policy parameters.
        This DQN agent should only be included as a separator in the next instance SCIP model.
        The priority of calling the DQN separator should be the lowest, so it will be able to
        see all the available cuts.
        """
        # get the current state, a dictionary of available cuts (keyed by their names,
        # and query the statistics related to the previous action (cut activity)
        cur_state, available_cuts = self.model.getState(state_format='tensor', get_available_cuts=True, query=self.prev_action)

        # validate the solver behavior
        if self.prev_action is not None:
            # assert that all the selected cuts were actually applied
            # otherwise, there is either a bug or a cut safety/feasibility issue.
            assert (self.prev_action['selected'] == self.prev_action['applied']).all()

        # finish with the previos step:
        self._update_episode_stats(current_round_ncuts=available_cuts['ncuts'])

        # if the are no avialable cuts, the solution is probably feasible,
        # and anyway, the episode is ended since we don't have anything to do.
        # so in this case we should just jump out,
        # without storing the current state-action pair,
        # since the terminal state is not important.
        if available_cuts['ncuts'] > 0:

            # select an action
            action = self._select_action(cur_state).detach().cpu().numpy().astype(np.bool)
            available_cuts['selected'] = action
            # force SCIP to take the selected cuts in action and discard the others
            if sum(action) > 0:
                # need to do it onlt if there are selected actions
                self.model.forceCuts(action)
                # set SCIP maxcutsroot and maxcuts to the number of selected cuts,
                # in order to prevent it from adding more or less cuts
                self.model.setIntParam('separating/maxcuts', int(sum(action)))
                self.model.setIntParam('separating/maxcutsroot', int(sum(action)))
            else:
                # TODO we need somehow to tell scip to stop cutting, without breaking the system. 
                # flush the separation storage
                self.model.clearCuts()

            # SCIP will execute the action,
            # and return here in the next LP round -
            # unless the instance is solved and the episode is done.
            # store the current state and action for
            # computing later the n-step rewards and the (s,a,r',s') transitions
            self.state_action_pairs.append((cur_state, available_cuts))
            self.prev_action = available_cuts
            self.prev_state = cur_state

    # done
    def finish_episode(self):
        """
        Compute rewards, push transitions into memory
        and log stats
        """
        # SCIP can terminate an episode (due to, say, node_limit or lp_iterations_limit)
        # after executing the LP without calling DQN.
        # In this case we need to compute by hand the tightness of the last action taken,
        # because the solver allowes to access the information only in SCIP_STAGE.SOLVING
        if self.cut_generator is not None:
            assert self.nseparounds == self.cut_generator.nseparounds

        if self.prev_action.get('tightness_penalty', None) is None:
            nvars = self.model.getNVars()
            ncuts = self.prev_action['ncuts']
            cuts_nnz_vals = self.prev_state['cut_nzrcoef']['vals']
            cuts_nnz_rowidxs = self.prev_state['cut_nzrcoef']['rowidxs']
            cuts_nnz_colidxs = self.prev_state['cut_nzrcoef']['colidxs']
            cuts_matrix = sp.sparse.coo_matrix((cuts_nnz_vals, (cuts_nnz_rowidxs, cuts_nnz_colidxs)), shape=[ncuts, nvars]).toarray()
            final_solution = self.model.getBestSol()
            sol_vector = [self.model.getSolVal(final_solution, x_i) for x_i in self.x.values()]
            sol_vector += [self.model.getSolVal(final_solution, y_ij) for y_ij in self.y.values()]
            sol_vector = np.array(sol_vector)
            # tightness of all cuts added at the previous round (including the discarded cuts)
            tightness = self.prev_action['rhss'] - cuts_matrix @ sol_vector
            # assign tightness penalty only to the selected cuts.
            self.prev_action['tightness_penalty'] = np.zeros_like(self.prev_action['selected'], dtype=np.float32)
            self.prev_action['tightness_penalty'][self.prev_action['selected']] = tightness[self.prev_action['selected']]
            # assume that all the selected actions were actually applied,
            # although we cannot verify it
            self.prev_action['applied'] = self.prev_action['selected']

            # update the rest of statistics needed to compute rewards
            self._update_episode_stats(current_round_ncuts=ncuts)


        # compute rewards and other stats for the whole episode,
        # and if in training session, push transitions into memory
        self._compute_rewards_and_stats()
        # increase the number of episodes done
        if self.training:
            self.i_episode += 1

    # done
    def sepaexeclp(self):
        if self.hparams.get('debug', False):
            print('dqn')

        # assert proper behavior
        self.nseparounds += 1
        if self.cut_generator is not None:
            assert self.nseparounds == self.cut_generator.nseparounds
            # assert self.nseparounds == self.model.getNLPs() todo: is it really important?

        if self.model.getNLPIterations() < self.lp_iterations_limit:
            self._do_dqn_step()
        # return {"result": SCIP_RESULT.DIDNOTRUN}
        return {"result": SCIP_RESULT.DIDNOTFIND}

    # done
    def _update_episode_stats(self, current_round_ncuts):
        # collect statistics related to the action taken at the previous round
        # since getNCuts takes into account the cuts added by the cut generator before
        # calling the cut selection agent, we need to subtract the current round ncuts from
        # this value
        self.episode_stats['ncuts'].append(self.model.getNCuts() - current_round_ncuts)
        self.episode_stats['ncuts_applied'].append(self.model.getNCutsApplied())
        self.episode_stats['solving_time'].append(self.model.getSolvingTime())
        self.episode_stats['processed_nodes'].append(self.model.getNNodes())
        self.episode_stats['gap'].append(self.model.getGap())
        self.episode_stats['lp_rounds'].append(self.model.getNLPs())
        self.episode_stats['lp_iterations'].append(self.model.getNLPIterations())
        self.episode_stats['dualbound'].append(self.model.getDualbound())

    # done
    def eval(self):
        self.training = False
        self.policy_net.eval()

    # done
    def train(self):
        self.training = True
        self.policy_net.train()

    # done
    def _compute_rewards_and_stats(self):
        """
        Compute action-wise reward and store (s,a,r,s') transitions in memory
        By the way, compute so metrics for plotting, e.g.
        1. dualbound integral,
        2. gap integral,
        3. nactive/napplied,
        4. napplied/navailable
        """
        lp_iterations_limit = self.lp_iterations_limit
        gap = self.episode_stats['gap']
        dualbound = self.episode_stats['dualbound']
        lp_iterations = self.episode_stats['lp_iterations']

        # compute the area under the curve:
        dualbound_area = get_normalized_areas(t=lp_iterations, ft=dualbound, t_support=lp_iterations_limit, reference=self.baseline['optimal_value'])
        gap_area = get_normalized_areas(t=lp_iterations, ft=gap, t_support=lp_iterations_limit, reference=0)  # optimal gap is always 0

        objective_area = dualbound_area if self.dqn_objective == 'dualbound_integral' else gap_area

        if self.training:
            # compute n-step returns for each state-action pair (s_t, a_t)
            # and store a transition (s_t, a_t, r_t, s_{t+n}
            n_transitions = len(self.state_action_pairs)
            n_steps = self.nstep_learning
            gammas = self.gamma**np.arange(n_steps).reshape(-1, 1)  # [1, gamma, gamma^2, ... gamma^{n-1}]
            indices = np.arange(n_steps).reshape(1, -1) + np.arange(n_transitions).reshape(-1, 1)  # indices of sliding windows
            # in case of n_steps > 1, pad objective_area with zeros only for avoiding overflow
            max_index = np.max(indices)
            if max_index >= len(objective_area):
                objective_area = np.pad(objective_area, (0, max_index+1-len(objective_area)), 'constant', constant_values=0)
            # take sliding windows of width n_step from objective_area
            # with minus because we want to minimize the area under the curve

            n_step_rewards = - objective_area[indices]
            # compute returns
            # R[t] = - ( r[t] + gamma*r[t+1] + ... + gamma^(n-1)r[t+n-1] )
            R = n_step_rewards @ gammas
            # assign rewards and store transitions (s,a,r,s')
            for step, ((state, action), joint_reward) in enumerate(zip(self.state_action_pairs, R)):
                next_state, _ = self.state_action_pairs[step+n_steps] if step+n_steps < n_transitions else (None, None)

                # credit assignment:
                # R is a joint reward for all cuts applied at each step.
                # now, assign to each cut its reward according to its tightness
                # tightness == 0 if the cut is tight, and > 0 otherwise. (<0 means violated and should happen)
                # so we punish inactive cuts by decreasing their reward to
                # R * (1 + tightness)
                tightness_penalty = action['tightness_penalty']
                if self.hparams.get('credit_assignment', True):
                    credit = 1 + tightness_penalty
                    reward = joint_reward * credit
                else:
                    reward = joint_reward * np.ones_like(tightness_penalty)

                transition = get_transition(state, action['selected'], reward, next_state)
                self.memory.push(transition)

        # compute some stats and store in buffer
        active_applied_ratio = []
        applied_available_ratio = []
        for _, action in self.state_action_pairs:
            tightness_penalty = action['tightness_penalty']
            # because of numerical errors, we consider as zero |value| < 1e-6
            approximately_zero = np.abs(tightness_penalty) < 1e-6
            tightness_penalty[approximately_zero] = 0

            applied = action['applied']
            is_active = tightness_penalty[applied] == 0
            active_applied_ratio.append(sum(is_active)/sum(applied) if sum(applied) > 0 else 0)
            applied_available_ratio.append(sum(applied)/len(applied) if len(applied) > 0 else 0)
        # store episode results in tmp_stats_buffer
        dualbound_integral = sum(dualbound_area)
        gap_integral = sum(gap_area)
        self.tmp_stats_buffer['dualbound_integral'].append(dualbound_integral)
        self.tmp_stats_buffer['gap_integral'].append(gap_integral)
        self.tmp_stats_buffer['active_applied_ratio'].append(np.mean(active_applied_ratio))
        self.tmp_stats_buffer['applied_available_ratio'].append(np.mean(applied_available_ratio))
        if self.baseline.get('dualbound_integral', None) is not None:
            # this is evaluation round.
            self.test_stats_buffer['db_int_improvement'].append(dualbound_integral/self.baseline['dualbound_integral'])
            self.test_stats_buffer['gap_int_improvement'].append(gap_integral/self.baseline['gap_integral'])

    # done
    def log_stats(self, save_best=False):
        """
        Average tmp_stats_buffer values, log to tensorboard dir,
        and reset tmp_stats_buffer for the next round.
        This function should be called periodically during training,
        and at the end of every evaluation session - <valid/test>_set_<easy/medium/hard>
        """
        if save_best:
            self._save_if_best()
        cur_time_sec = time() - self.start_time + self.walltime_offset
        print(f'Episode {self.i_episode} \t| ', end='')
        for k, vals in self.tmp_stats_buffer.items():
            avg = np.mean(vals)
            std = np.std(vals)
            print('{}: {:.4f} \t| '.format(k, avg), end='')
            self.writer.add_scalar(k + '/' + self.dataset_name, avg,
                                   global_step=self.i_episode,
                                   walltime=cur_time_sec)
            self.writer.add_scalar(k + '_std' + '/' + self.dataset_name, std,
                                   global_step=self.i_episode,
                                   walltime=cur_time_sec)

            self.tmp_stats_buffer[k] = []

        if len(self.test_stats_buffer['db_int_improvement']) > 0:
            for k, vals in self.test_stats_buffer.items():
                avg = np.mean(vals)
                std = np.std(vals)
                print('{}: {:.4f} \t| '.format(k, avg), end='')
                self.writer.add_scalar(k + '/' + self.dataset_name, avg,
                                       global_step=self.i_episode,
                                       walltime=cur_time_sec)
                self.writer.add_scalar(k+'_std' + '/' + self.dataset_name, std,
                                       global_step=self.i_episode,
                                       walltime=cur_time_sec)
                self.test_stats_buffer[k] = []

        # log the average loss of the last training session
        print('Loss: {:.4f} \t| '.format(self.loss_moving_avg), end='')
        self.writer.add_scalar('Training_Loss', self.loss_moving_avg,
                               global_step=self.i_episode,
                               walltime=cur_time_sec)
        print(f'Step: {self.steps_done} \t| ', end='')

        d = int(np.floor(cur_time_sec/(3600*24)))
        h = int(np.floor(cur_time_sec/3600) - 24*d)
        m = int(np.floor(cur_time_sec/60) - 60*(24*d + h))
        s = int(cur_time_sec) % 60
        print('Iteration Time: {:.1f}[sec]\t| '.format(cur_time_sec - self.last_time_sec), end='')
        print('Total Time: {}-{:02d}:{:02d}:{:02d}'.format(d, h, m, s))
        self.last_time_sec = cur_time_sec

    # done
    def save_checkpoint(self, filepath=None):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'i_episode': self.i_episode,
            'walltime_offset': time() - self.start_time + self.walltime_offset,
            'best_perf': self.best_perf,
            'loss_moving_avg': self.loss_moving_avg,
        }, filepath if filepath is not None else self.checkpoint_filepath)
        print('Saved checkpoint to: ', filepath if filepath is not None else self.checkpoint_filepath)

    # done
    def _save_if_best(self):
        """Save the model if show the best performance on the validation set.
        The performance is the -(dualbound/gap integral),
        according to the DQN objective"""
        perf = -np.mean(self.tmp_stats_buffer[self.dqn_objective])
        if perf > self.best_perf[self.dataset_name]:
            self.best_perf[self.dataset_name] = perf
            self.save_checkpoint(filepath=os.path.join(self.logdir, f'best_{self.dataset_name}_checkpoint.pt'))

    # done
    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_filepath):
            print('Checkpoint file does not exist! starting from scratch.')
            return
        checkpoint = torch.load(self.checkpoint_filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.i_episode = checkpoint['i_episode']
        self.walltime_offset = checkpoint['walltime_offset']
        self.best_perf = checkpoint['best_perf']
        self.loss_moving_avg = checkpoint['loss_moving_avg']
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        print('Loaded checkpoint from: ', self.checkpoint_filepath)
