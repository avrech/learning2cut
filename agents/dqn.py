from pyscipopt import  Sepa, SCIP_RESULT, SCIP_STAGE
from time import time
import networkx as nx
import numpy as np
from utils.data import Transition
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
from utils.scip_models import maxcut_mccormic_model
from torch.utils.tensorboard import SummaryWriter

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
        self.memory = ReplayMemory(hparams.get('memory_capacity', 10))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = hparams.get('batch_size', 2)
        self.gamma = hparams.get('gamma', 0.999)
        self.eps_start = hparams.get('eps_start', 0.9)
        self.eps_end = hparams.get('eps_end', 0.05)
        self.eps_decay = hparams.get('eps_decay', 200)
        self.target_update = hparams.get('target_update', 2)
        self.policy_net = Qnet(hparams=hparams).to(self.device)
        self.target_net = Qnet(hparams=hparams).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        # value aggregation method for the target Q values
        if hparams.get('value_aggr', 'mean') == 'max':
            self.aggr_func = scatter_max
        elif hparams.get('value_aggr', 'mean') == 'mean':
            self.aggr_func = scatter_mean
        self.reward_func = hparams.get('reward_func', 'db_integral_credit')
        self.nstep_learning = hparams.get('nstep_learning', 1)
        self.dqn_objective = hparams.get('dqn_objective', 'dualbound_integral')

        # training stuff
        self.steps_done = 0
        self.i_episode = 0
        self.training = True
        self.checkpoint_freq = hparams.get('checkpoint_freq', 100)
        self.walltime_offset = 0
        self.start_time = time()

        # file system paths
        self.datapath = hparams.get('data_abspath', 'data')
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

        # logging
        self.writer = SummaryWriter(log_dir=os.path.join(self.logdir, 'tensorboard'))
        # todo compute fscore of p = nactive/napplied and q = nactive / (napplied + nstillviolated)
        self.dataset_key = 'train_set'  # or <easy/medium/hard>_<valid_set/test_set>
        self.log_freq = hparams.get('log_freq', 100)
        # tmp buffer for holding each episode results until averaging and appending to experiment_stats
        self.tmp_stats_buffer = {'dualbound_integral': [], 'gap_integral': [], 'active_applied_ratio': [], 'applied_available_ratio': []}
        self.best_perf = {'easy': -1000000, 'medium': -1000000, 'hard': -1000000}

    # done
    def init_episode(self, G, x, y, baseline=None, dataset_key='train_set'):
        self.G = G
        self.x = x
        self.y = y
        self.baseline = baseline  # todo, generate baseline where?
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
        self.dataset_key = dataset_key

    # done
    def select_action(self, state):
        if self.training:
            # take epsilon-greedy action
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                            math.exp(-1. * self.steps_done / self.ps_decay)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return self.policy_net(state).max(1)[1]
            else:
                random_action = torch.randint_like(state.a, low=0, high=1, dtype=torch.long)
                return random_action
        else:
            # in test time, take greedy action
            return self.policy_net(state).max(1)[1]

    # done
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # todo consider all features to parse correctly
        batch = Batch().from_data_list(transitions, follow_batch=['x_c', 'x_v', 'x_a', 'nx_c', 'nx_v', 'nx_a']).to(self.device)

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
        ).gather(1, action_batch)

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
                                           dim_size=self.batch_size)        # output tensor size in dim after scattering
        # override with zeros the values of terminal states which are zero by convention
        next_state_values[batch.ns_terminal] = 0
        # scatter the next state values graph-wise to update all action-wise rewards
        next_state_values.scatter_(dim=0, index=batch.x_a_batch, src=next_state_values)

        # now compute the expected Q values for each action separately
        reward_batch = batch.r
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network, copying all weights and biases in DQN
        if self.i_episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # done
    def do_dqn_step(self):
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
        # finish with the previos step:
        self.update_episode_stats()

        # get the current state, a dictionary of available cuts (keyed by their names,
        # and query the statistics related to the previous action (cut activity)
        cur_state, available_cuts = self.model.getState(state_format='tensor', get_available_cuts=True, query=self.prev_action)

        # validate the solver behavior
        if self.prev_action is not None:
            # assert that all the selected cuts were actually applied
            # otherwise, there is either a bug or a cut safety/feasibility issue.
            assert all(self.prev_action['selected'] == self.prev_action['applied'])

        # select an action
        action = self.select_action(cur_state).detach().cpu().numpy().astype(np.int32)
        available_cuts['selected'] = action
        # and force SCIP to take the selected cuts in action, and discard the others
        self.model.forceCuts(action)

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
        # compute by hand the activity of the last action,
        # because the solver doesn't allow to access the information in SCIP_STAGE.SOLVED
        ncols = self.model.getNLPcols()
        ncuts = self.prev_action['ncuts']
        cuts_nnz_vals = self.prev_state['cut_nzrcoef']['vals']
        cuts_nnz_rowidxs = self.prev_state['cut_nzrcoef']['rowidxs']
        cuts_nnz_colidxs = self.prev_state['cut_nzrcoef']['colidxs']
        cuts_matrix = sp.sparse.coo_matrix((cuts_nnz_vals,(cuts_nnz_rowidxs,cuts_nnz_colidxs)), shape=[ncuts, ncols]).toarray()
        final_solution = self.model.getBestSol()
        sol_vector = [self.model.getSolVal(final_solution, x_i) for x_i in self.x.values()]
        sol_vector += [self.model.getSolVal(final_solution, y_ij) for y_ij in self.y.values()]
        sol_vector = np.array(sol_vector)
        activities = cuts_matrix @ sol_vector - self.prev_action['constants']
        self.prev_action['activity'] = activities
        # assume that all the selected actions were actually applied,
        # although we cannot verify it
        self.prev_action['applied'] = self.prev_action['selected']
        # update the rest of statistics needed to compute rewards
        self.update_episode_stats()
        # compute rewards and other stats for the whole episode,
        # and if in training session, push transitions into memory
        self.compute_rewards_and_stats()
        # increase the number of episodes done
        if self.training:
            self.i_episode += 1

    # done
    def sepaexeclp(self):
        self.do_dqn_step()
        return {"result": SCIP_RESULT.DIDNOTRUN}

    # done
    def update_episode_stats(self):
        # collect statistics at the beginning of each round, starting from the second round.
        # the statistics are collected before taking any action, and refer to the last round.
        # NOTE: the last update must be done after the solver terminates optimization,
        # outside of this module, by calling McCormicCycleSeparator.update_stats() one more time.
        self.episode_stats['ncuts'].append(self.model.getNCuts())
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
    def compute_rewards_and_stats(self):
        """
        Compute action-wise reward and store (s,a,r,s') transitions in memory
        By the way, compute so metrics for plotting, e.g.
        1. dualbound integral,
        2. gap integral,
        3. nactive/napplied,
        4. napplied/navailable
        """
        lp_iterations_limit = self.hparams['lp_iterations_limit']
        gap = self.episode_stats['gap']
        dualbound = self.episode_stats['dualbound']
        lp_iterations = self.episode_stats['lp_iterations']

        # extend the dualbound and lp_iterations to a common support
        extended = False
        if lp_iterations[-1] < lp_iterations_limit:
            gap.append(gap[-1])
            dualbound.append(dualbound[-1])
            lp_iterations.append(lp_iterations_limit)
            extended = True
        gap = np.array(gap)
        dualbound = np.array(dualbound)
        lp_iterations = np.array(lp_iterations)

        # normalize the dualbound (according to the optimal_value) to [0,1]
        # such that it will start from 1 and end at zero (if optimal)
        dualbound = np.abs(dualbound - self.baseline['optimal_value']) / dualbound[0]

        # normalize the gap to start at 1 and end at zero (if optimal)
        gap = gap / gap[0]

        # normalize lp_iterations to [0,1]
        lp_iterations = lp_iterations / lp_iterations_limit

        # compute the area under the curve using first order interpolation
        gap_diff = gap[1:] - gap[:-1]
        dualbound_diff = dualbound[1:] - dualbound[:-1]
        lp_iterations_diff = lp_iterations[1:] - lp_iterations[:-1]
        gap_area = lp_iterations_diff * (gap[:-1] + gap_diff/2)
        dualbound_area = lp_iterations_diff * (dualbound[:-1] + dualbound_diff/2)
        if extended:
            # add the extension area to the last transition area
            gap_area[-2] += gap_area[-1]
            dualbound_area[-2] += dualbound_area[-1]
            # truncate the extension, and leave n-areas for the n-transition done
            gap_area = gap_area[:-1]
            dualbound_area = dualbound_area[:-1]
        objective_area = dualbound_area if self.dqn_objective == 'dualbound_integral' else gap_area

        if self.training:
            # compute n-step returns for each state-action pair (s_t, a_t)
            # and store a transition (s_t, a_t, r_t, s_{t+n}
            n_transitions = len(self.state_action_pairs)
            n_steps = self.nstep_learning
            gammas = self.gamma**np.arange(n_steps).reshape(-1, 1)  # [1, gamma, gamma^2, ... gamma^{n-1}]
            indices = np.arange(n_steps).reshape(1, -1) + np.arange(n_transitions).reshape(-1, 1)  # indices of sliding windows
            # take sliding windows of width n_step from objective_area
            # with minus because we want to minimize the area under the curve
            n_step_rewards = - objective_area[indices]
            # compute returns
            # R[t] = - ( r[t] + gamma*r[t+1] + ... + gamma^(n-1)r[t+n-1] )
            R = n_step_rewards @ gammas
            # assign rewards and store transitions (s,a,r,s')
            for step, state, action, joint_reward in enumerate(zip(self.state_action_pairs, R)):
                next_state = self.state_action_pairs[step+n_steps] if step+n_steps < n_transitions else None

                # credit assignment:
                # R is a joint reward for all cuts applied at each step.
                # now, assign to each cut its reward according to its activity
                # |activity| = 0 if the cut is tight, and > 0 otherwise.
                # so we punish inactive cuts by decreasing their reward to
                # R * (1 + |activity|)
                activity = action['activity']
                if self.hparams.get('credit_assignment', True):
                    credit = 1 + activity
                    reward = joint_reward * credit

                transition = Transition(state, action, reward, next_state)
                self.memory.push(transition)

        # compute some stats and store in buffer
        active_applied_ratio = []
        applied_available_ratio = []
        for _, action in self.state_action_pairs:
            activity = action['activity']
            applied = action['applied'].astype(np.bool)
            is_active = activity[applied] == 0
            active_applied_ratio.append(sum(is_active)/sum(applied))
            applied_available_ratio.append(sum(applied)/len(applied))
        # store episode results in tmp_stats_buffer
        self.tmp_stats_buffer['dualbound_integral'].append(sum(dualbound_area))
        self.tmp_stats_buffer['gap_integral'].append(sum(gap_area))
        self.tmp_stats_buffer['active_applied_ratio'].append(np.mean(active_applied_ratio))
        self.tmp_stats_buffer['applied_available_ratio'].append(np.mean(applied_available_ratio))
        # todo compute for the whole trajectory
        # # compute reward
        # db_improvement = np.abs(self.episode_stats['dualbound'][-1] - self.episode_stats['dualbound'][-2]) * self.db_scale
        # lp_iterations = (self.episode_stats['lp_iterations'][-1] - self.episode_stats['lp_iterations'][-2]) * self.lpiter_scale
        # activity = self.prev_action['activity']
        # if self.reward_func == 'db_improvement':
        #     return np.full_like(activity, fill_value=db_improvement)
        #
        # elif self.reward_func == 'db_integral':
        #     return np.full_like(activity, fill_value=- db_improvement * lp_iterations)
        #
        # elif self.reward_func == 'db_improvement_credit':
        #     return db_improvement * (1 + activity)
        #
        # elif self.reward_func == 'db_integral_credit':
        #     return db_improvement * lp_iterations * (activity - 1)
        #
        # elif self.reward_func == 'db_lpiter_fscore':
        #     # compute the harmonic average of p=db_improvement and q=1/lp_iterations
        #     # fscore = p*q/(p+q)
        #     fscore = db_improvement / (db_improvement * lp_iterations + 1)
        #     # this fscore will be high iff its both elements will be high,
        #     # i.e great dual bound improvement in a few lp iterations
        #     return np.full_like(activity, fill_value=fscore)
        #
        # elif self.reward_func == 'db_lpiter_fscore_credit':
        #     # compute the fscore as above,
        #     # and assign the credit to the active constraints only.
        #     fscore = db_improvement / (db_improvement * lp_iterations + 1)
        #     return fscore * (1 + activity)

    # done
    def log_stats(self):
        """
        Average tmp_stats_buffer values, log to tensorboard dir,
        and reset tmp_stats_buffer for the next round.
        This function should be called periodically during training,
        and at the end of every evaluation session - <valid/test>_set_<easy/medium/hard>
        """
        for k, vals in self.tmp_stats_buffer.items():
            avg = np.mean(vals)
            self.writer.add_scalar(k + '/' + self.dataset_key, avg,
                                   global_step=self.i_episode,
                                   walltime=time() - self.start_time + self.walltime_offset)
            self.tmp_stats_buffer[k] = []

    # done
    def save_checkpoint(self, filepath=None):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'i_episode': self.i_episode,
            'walltime_offset': time() - self.start_time + self.walltime_offset,
            'best_perf': self.best_perf
        }, filepath if filepath is not None else self.checkpoint_filepath)

    def save_if_best(self):
        """Save the model if show the best performance on the validation set.
        The performance is the -(dualbound/gap integral),
        according to the DQN objective"""
        perf = -np.mean(self.tmp_stats_buffer[self.dqn_objective])
        if perf > self.best_perf[self.dataset_key]:
            self.best_perf[self.dataset_key] = perf
            self.save_checkpoint(filepath=os.path.join(self.logdir, f'best_{self.dataset_key}_checkpoint.pt'))

    # done
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.i_episode = checkpoint['i_episode']
        self.walltime_offset = checkpoint['walltime_offset']
        self.best_perf = checkpoint['best_perf']
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer.to(self.device)


def testDQN():
    pass

if __name__ == '__main__':
    testDQN()
