from pyscipopt import  Sepa, SCIP_RESULT
from time import time
import numpy as np
from utils.data import get_transition
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
from tqdm import tqdm
from utils.functions import get_normalized_areas
from collections import namedtuple
import matplotlib as mpl
import pickle
from utils.scip_models import maxcut_mccormic_model
from utils.buffer import ReplayBuffer, PrioritizedReplayBuffer
from separators.mccormick_cycle_separator import MccormickCycleSeparator
mpl.rc('figure', max_open_warning=0)
# mpl.rcParams['text.antialiased'] = False
# mpl.use('agg')
import matplotlib.pyplot as plt
StateActionContext = namedtuple('StateActionQValuesContext', ('scip_state', 'action', 'q_values', 'transformer_context'))


class GDQN(Sepa):
    def __init__(self, name='DQN', hparams={}, **kwargs):
        """
        Sample scip.Model state every time self.sepaexeclp is invoked.
        Store the generated data object in
        """
        self.name = name
        self.hparams = hparams

        # DQN stuff
        self.use_per = hparams.get('use_per', True)
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(hparams.get('memory_capacity', 1000000), alpha=hparams.get('priority_alpha', 0.6))
            self.priority_beta_start = hparams.get('priority_beta_start', 0.4)
            self.priority_beta_end = hparams.get('priority_beta_end', 1.0)
            self.priority_beta_decay = hparams.get('priority_beta_decay', 10000)
        else:
            self.memory = ReplayBuffer(hparams.get('memory_capacity', 1000000))
        self.device = torch.device(f"cuda:{hparams['gpu_id']}" if torch.cuda.is_available() and hparams.get('gpu_id', None) is not None else "cpu")
        self.batch_size = hparams.get('batch_size', 64)
        self.gamma = hparams.get('gamma', 0.999)
        self.eps_start = hparams.get('eps_start', 0.9)
        self.eps_end = hparams.get('eps_end', 0.05)
        self.eps_decay = hparams.get('eps_decay', 200)
        self.policy_net = TQnet(hparams=hparams).to(self.device) if hparams.get('dqn_arch', 'TQNet') == 'TQNet' else Qnet(hparams=hparams).to(self.device)
        self.target_net = TQnet(hparams=hparams).to(self.device) if hparams.get('dqn_arch', 'TQNet') == 'TQNet' else Qnet(hparams=hparams).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=hparams.get('lr', 0.001), weight_decay=hparams.get('weight_decay', 0.0001))
        # value aggregation method for the target Q values
        if hparams.get('value_aggr', 'mean') == 'max':
            self.aggr_func = scatter_max
        elif hparams.get('value_aggr', 'mean') == 'mean':
            self.aggr_func = scatter_mean
        self.nstep_learning = hparams.get('nstep_learning', 1)
        self.dqn_objective = hparams.get('dqn_objective', 'db_auc')
        self.use_transformer = hparams.get('dqn_arch', 'TQNet') == 'TQNet'
        self.empty_action_penalty = self.hparams.get('empty_action_penalty', 0)

        # training stuff
        self.num_env_steps_done = 0
        self.num_sgd_steps_done = 0
        self.num_policy_updates = 0
        self.i_episode = 0
        self.training = True
        self.walltime_offset = 0
        self.start_time = time()
        self.last_time_sec = self.walltime_offset

        # file system paths
        # todo - set worker-specific logdir for distributed DQN
        self.logdir = hparams.get('logdir', 'results')
        self.checkpoint_filepath = os.path.join(self.logdir, 'checkpoint.pt')

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
        self.cut_generator = None
        self.nseparounds = 0
        self.dataset_name = 'trainset'  # or <easy/medium/hard>_<validset/testset>
        self.lp_iterations_limit = -1

        # logging
        self.is_tester = True
        self.is_worker = True
        self.is_learner = True
        self.writer = SummaryWriter(log_dir=os.path.join(self.logdir, 'tensorboard'))
        # todo compute fscore of p = nactive/napplied and q = nactive / (napplied + nstillviolated)
        # tmp buffer for holding each episode results until averaging and appending to experiment_stats
        self.tmp_stats_buffer = {'db_auc': [], 'gap_auc': [], 'active_applied_ratio': [], 'applied_available_ratio': []}
        self.test_stats_buffer = {'db_auc_imp': [], 'gap_auc_imp': []}
        # best performance log for validation sets
        self.best_perf = {k: -1000000 for k in hparams['datasets'].keys() if k[:8] == 'validset'}
        self.loss_moving_avg = 0
        # self.figures = {'Dual_Bound_vs_LP_Iterations': [], 'Gap_vs_LP_Iterations': []}
        self.figures = {}

    # done
    def init_episode(self, G, x, y, lp_iterations_limit, cut_generator=None, baseline=None, dataset_name='trainset', scip_seed=None):
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
                            math.exp(-1. * self.num_env_steps_done / self.eps_decay)
            self.num_env_steps_done += 1

            if sample > eps_threshold:
                # take greedy action
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is the index of where max element was found.
                    # we pick action with the larger expected reward.
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
                    )
                    greedy_action = q_values.max(1)[1].detach().cpu().numpy().astype(np.bool)
                    if self.use_transformer:
                        # return also the decoder context to store for backprop
                        greedy_action_decoder_context = self.policy_net.decoder_context
                    else:
                        greedy_action_decoder_context = None
                    return greedy_action, q_values.detach().cpu(), greedy_action_decoder_context

            else:
                # randomize action
                random_action = torch.randint_like(batch.a, low=0, high=2, dtype=torch.float32).cpu()
                if self.use_transformer:
                    # we create random decoder context to store for backprop.
                    # we randomize inference order for building the context,
                    # since it is exactly what the decoder does.
                    decoder_edge_attr_list = []
                    decoder_edge_index_list = []
                    ncuts = random_action.shape[0]
                    inference_order = torch.randperm(ncuts)
                    edge_index_dec = torch.cat([torch.arange(ncuts).view(1, -1),
                                                torch.empty((1, ncuts), dtype=torch.long)], dim=0)
                    edge_attr_dec = torch.zeros((ncuts, 2), dtype=torch.float32)
                    # iterate over all cuts in the random order, and set each one a context
                    for cut_index in inference_order:
                        # set all edges to point from all cuts to the currently processed one (focus the attention mechanism)
                        edge_index_dec[1, :] = cut_index
                        # store the context (edge_index_dec and edge_attr_dec) of the current iteration
                        decoder_edge_attr_list.append(edge_attr_dec.clone())
                        decoder_edge_index_list.append(edge_index_dec.clone())
                        # assign the random action of cut_index to the context of the next round
                        edge_attr_dec[cut_index, 0] = 1  # mark the current cut as processed
                        edge_attr_dec[cut_index, 1] = random_action[cut_index]  # mark the cut as selected or not

                    # finally, stack the decoder edge_attr and edge_index tensors, and make a transformer context
                    random_edge_attr_dec = torch.cat(decoder_edge_attr_list, dim=0)
                    random_edge_index_dec = torch.cat(decoder_edge_index_list, dim=1)
                    random_action_decoder_context = TransformerDecoderContext(random_edge_index_dec, random_edge_attr_dec)
                else:
                    random_edge_attr_dec = None
                    random_edge_index_dec = None
                    random_action_decoder_context = None
                # for prioritized experience replay we need the q_values to compute the initial priorities
                # whether we take a random action or not.
                # For transformer, we compute the random action q_values based on the random decoder context,
                # and we do it in parallel like we do in sgd_step()
                # For non-transformer model, it doesn't affect anything
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
                    edge_index_dec=random_edge_index_dec.to(self.device),
                    edge_attr_dec=random_edge_attr_dec.to(self.device)
                ).detach().cpu()
                random_action = random_action.numpy().astype(np.bool)
                return random_action, q_values, random_action_decoder_context
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
                    edge_attr_a2a=batch.edge_attr_a2a
                )
                greedy_action = q_values.max(1)[1].cpu().numpy().astype(np.bool)
                # return None, None for q_values and decoder context,
                # since they are used only while generating experience
                return greedy_action, None, None

    # done
    def optimize_model(self):
        """
        Single threaded DQN policy update function.
        Sample uniformly a batch from the memory and execute one SGD pass
        todo - support PER
               importance sampling +
               when recovering from failures wait like in Ape-X paper appendix
        """
        if len(self.memory) < self.batch_size:
            return
        if self.use_per:
            # todo sample with beta
            beta = self.priority_beta_end - (self.priority_beta_end - self.priority_beta_start) * math.exp(-1. * self.num_sgd_steps_done / self.priority_beta_decay)
            transitions, weights, idxes = self.memory.sample(self.batch_size, beta)
            new_priorities = self.sgd_step(transitions=transitions, importance_sampling_correction_weights=weights)
            # update priorities
            self.memory.update_priorities(idxes, new_priorities)

        else:
            transitions = self.memory.sample(self.batch_size)
            self.sgd_step(transitions)

        self.num_policy_updates += 1

    def sgd_step(self, transitions, importance_sampling_correction_weights=None):
        """ implement the basic DQN optimization step """
        # todo consider all features in follow_batch to parse correctly

        # old replay buffer returned transitions as separated Transition objects
        batch = Batch().from_data_list(transitions, follow_batch=['x_c', 'x_v', 'x_a', 'ns_x_c', 'ns_x_v', 'ns_x_a']).to(self.device)

        action_batch = batch.a

        # Compute Q(s, a):
        # the model computes Q(s,:), then we select the columns of the actions taken, i.e action_batch.
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
            edge_index_dec=batch.edge_index_dec,  # transformer stuff
            edge_attr_dec=batch.edge_attr_dec     # transformer stuff
        )
        q_values = q_values.gather(1, action_batch.unsqueeze(1))

        # Compute the Bellman target for all next states.
        # Expected values of actions for non_terminal_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0] for DQN, or based on the policy_net
        # prediction for DDQN.
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was terminal.
        # The value of a state is computed as in BQN paper https://arxiv.org/pdf/1711.08946.pdf

        # next-state action-wise values
        if self.use_transformer:
            # trick to parallelize the next state q_vals computation:
            # since the optimal target q values should be
            # independent of the inference order,
            # we infer each cut q-values like it was processed first in the inference loop.
            # thus, the edge_attr_dec is zeros, to indicate that each
            # cut is processed conditioning on "no-cut-selected"
            ns_edge_index_dec = batch.ns_edge_index_a2a
            ns_edge_attr_dec = torch.zeros_like(ns_edge_index_dec, dtype=torch.float32).t()
        else:
            ns_edge_index_dec = None
            ns_edge_attr_dec = None

        # for each graph in the next state batch, and for each cut, compute the q_values
        # using the target network.
        target_next_q_values = self.target_net(
            x_c=batch.ns_x_c,
            x_v=batch.ns_x_v,
            x_a=batch.ns_x_a,
            edge_index_c2v=batch.ns_edge_index_c2v,
            edge_index_a2v=batch.ns_edge_index_a2v,
            edge_attr_c2v=batch.ns_edge_attr_c2v,
            edge_attr_a2v=batch.ns_edge_attr_a2v,
            edge_index_a2a=batch.ns_edge_index_a2a,
            edge_attr_a2a=batch.ns_edge_attr_a2a,
            edge_index_dec=ns_edge_index_dec,
            edge_attr_dec=ns_edge_attr_dec
        )

        # compute the target using either DQN target or DDQN target
        if self.hparams.get('update_rule', 'DQN') == 'DQN':
            # y = r + gamma max_a' target_net(s', a')
            max_target_next_q_values = target_next_q_values.max(1)[0].detach()
        elif self.hparams.get('update_rule', 'DQN') == 'DDQN':
            # y = r + gamma target_net(s', argmax_a' policy_net(s', a'))
            policy_next_q_values = self.policy_net(
                x_c=batch.ns_x_c,
                x_v=batch.ns_x_v,
                x_a=batch.ns_x_a,
                edge_index_c2v=batch.ns_edge_index_c2v,
                edge_index_a2v=batch.ns_edge_index_a2v,
                edge_attr_c2v=batch.ns_edge_attr_c2v,
                edge_attr_a2v=batch.ns_edge_attr_a2v,
                edge_index_a2a=batch.ns_edge_index_a2a,
                edge_attr_a2a=batch.ns_edge_attr_a2a,
                edge_index_dec=ns_edge_index_dec,
                edge_attr_dec=ns_edge_attr_dec
            )
            argmax_policy_next_q_values = policy_next_q_values.max(1)[1].detach()
            max_target_next_q_values = target_next_q_values.gather(1, argmax_policy_next_q_values).detach()

        # aggregate the action-wise values using mean or max,
        # and generate for each graph in the batch a single value
        max_target_next_q_values_aggr = self.aggr_func(max_target_next_q_values,   # source vector
                                           batch.ns_x_a_batch,              # target index of each element in source
                                           dim=0,                           # scattering dimension
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
            loss = (importance_sampling_correction_weights * F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1), reduction='none')).mean()
        else:
            # generate equal weights for all losses
            loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))

        # loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1)) - original pytorch example
        self.loss_moving_avg = 0.95*self.loss_moving_avg + 0.05*loss.detach().cpu().numpy()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # todo - for distributed learning return losses to update priorities - double check
        if self.use_per:
            # for transition, compute the norm of its TD-error
            td_error = torch.abs(q_values - target_q_values.unsqueeze(1)).detach()
            td_error = torch.clamp(td_error, min=1e-8)
            # to compute p,q norm take power p and compute sqrt q.
            td_error_l2_norm = torch.sqrt(scatter_add(td_error ** 2, batch.x_a_batch, # target index of each element in source
                                                      dim=0,                          # scattering dimension
                                                      dim_size=self.batch_size))      # output tensor size in dim after scattering

            new_priorities = td_error_l2_norm.squeeze().cpu().numpy().tolist()
            self.num_sgd_steps_done += 1
            return new_priorities
        else:
            return None

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

            # select an action, and get the decoder context for a case we use transformer and q_values for PER
            action, q_values, decoder_context = self._select_action(cur_state)
            available_cuts['selected'] = action
            # force SCIP to take the selected cuts and discard the others
            if sum(action) > 0:
                # need to do it only if there are selected actions
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
            self.state_action_qvalues_context_list.append(StateActionContext(cur_state, available_cuts, q_values, decoder_context))
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

        if self.prev_action.get('normalized_slack', None) is None:
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
        dualbound_area = get_normalized_areas(t=lp_iterations, ft=dualbound, t_support=lp_iterations_limit, reference=self.baseline['optimal_value'])
        gap_area = get_normalized_areas(t=lp_iterations, ft=gap, t_support=lp_iterations_limit, reference=0)  # optimal gap is always 0
        if self.dqn_objective == 'db_auc':
            objective_area = dualbound_area
        elif self.dqn_objective == 'gap_auc':
            objective_area = gap_area
        else:
            raise NotImplementedError

        if self.training:
            # compute n-step returns for each state-action pair (s_t, a_t)
            # and store a transition (s_t, a_t, r_t, s_{t+n}
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
                # get the next n-step state and q values. if the episode already terminated,
                # return 0 as q_values, since the q value of the terminal state and afterward is zero by convention
                next_state, _, next_q_values, _ = self.state_action_qvalues_context_list[step + n_steps] if step + n_steps < n_transitions else (None, None, None, None)

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

                transition = get_transition(scip_state=state,
                                            action=action['selected'],
                                            transformer_decoder_context=transformer_decoder_context,
                                            reward=reward,
                                            scip_next_state=next_state)

                if self.use_per:
                    # todo - compute initial priority for PER based on the policy q_values.
                    #        compute the TD error for each action in the current state as we do in sgd_step,
                    #        and then take the norm of the resulting cut-wise TD-errors as the initial priority
                    selected_action = torch.from_numpy(action['selected']).unsqueeze(1).long()  # cut-wise action
                    q_values = q_values.gather(1, selected_action)
                    # todo verify behavior with terminal state
                    if next_q_values is None:
                        # next state is terminal, and its q_values are 0 by convention
                        target_q_values = torch.from_numpy(reward)
                    else:
                        max_next_q_values = next_q_values.max(1)[0]
                        if self.hparams.get('value_aggr', 'mean') == 'max':
                            max_next_q_values_aggr = max_next_q_values.max()
                        if self.hparams.get('value_aggr', 'mean') == 'mean':
                            max_next_q_values_aggr = max_next_q_values.mean()
                        max_next_q_values_broadcast = torch.full_like(q_values, fill_value=max_next_q_values_aggr)
                        target_q_values = torch.from_numpy(reward) + (self.gamma ** self.nstep_learning) * max_next_q_values_broadcast
                    td_error = torch.abs(q_values - target_q_values)
                    td_error = torch.clamp(td_error, min=1e-8)
                    # todo - support pq norm
                    initial_priority = torch.norm(td_error).item()  # default L2 norm
                    # todo - support PER, local buffer and remote PER
                    self.memory.add(transition, initial_priority)
                else:
                    self.memory.add(transition)

        # compute some stats and store in buffer
        active_applied_ratio = []
        applied_available_ratio = []
        for _, action, _, _ in self.state_action_qvalues_context_list:
            normalized_slack = action['normalized_slack']
            # because of numerical errors, we consider as zero |value| < 1e-6
            approximately_zero = np.abs(normalized_slack) < 1e-6
            normalized_slack[approximately_zero] = 0

            applied = action['applied']
            is_active = normalized_slack[applied] == 0
            active_applied_ratio.append(sum(is_active)/sum(applied) if sum(applied) > 0 else 0)
            applied_available_ratio.append(sum(applied)/len(applied) if len(applied) > 0 else 0)
        # store episode results in tmp_stats_buffer
        db_auc = sum(dualbound_area)
        gap_auc = sum(gap_area)
        self.tmp_stats_buffer['db_auc'].append(db_auc)
        self.tmp_stats_buffer['gap_auc'].append(gap_auc)
        self.tmp_stats_buffer['active_applied_ratio'].append(np.mean(active_applied_ratio))
        self.tmp_stats_buffer['applied_available_ratio'].append(np.mean(applied_available_ratio))
        if self.baseline.get('rootonly_stats', None) is not None:
            # this is evaluation round.
            self.test_stats_buffer['db_auc_imp'].append(db_auc/self.baseline['rootonly_stats'][self.scip_seed]['db_auc'])
            self.test_stats_buffer['gap_auc_imp'].append(gap_auc/self.baseline['rootonly_stats'][self.scip_seed]['gap_auc'])

    # done
    def log_stats(self, save_best=False, plot_figures=False):
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
        global_step = self.num_policy_updates
        print(f'Global step: {global_step} | {self.dataset_name}\t|', end='')
        cur_time_sec = time() - self.start_time + self.walltime_offset

        # todo tester
        if self.is_tester:
            if plot_figures:
                self.decorate_figures()

            if save_best:
                # self._save_if_best()
                perf = np.mean(self.tmp_stats_buffer[self.dqn_objective])  # todo bug with (-) ?
                if perf > self.best_perf[self.dataset_name]:
                    self.best_perf[self.dataset_name] = perf
                    self.save_checkpoint(filepath=os.path.join(self.logdir, f'best_{self.dataset_name}_checkpoint.pt'))
                    self.save_figures(filename_prefix='best')

            # add episode figures (for validation and test sets only)
            if plot_figures:
                for figname in ['Dual_Bound_vs_LP_Iterations', 'Gap_vs_LP_Iterations']:
                    self.writer.add_figure(figname + '/' + self.dataset_name, self.figures[figname]['fig'],
                                           global_step=global_step, walltime=cur_time_sec)

            # plot dualbound and gap auc improvement over the baseline (for validation and test sets only)
            if len(self.test_stats_buffer['db_auc_imp']) > 0:
                for k, vals in self.test_stats_buffer.items():
                    avg = np.mean(vals)
                    std = np.std(vals)
                    print('{}: {:.4f} | '.format(k, avg), end='')
                    self.writer.add_scalar(k + '/' + self.dataset_name, avg, global_step=global_step, walltime=cur_time_sec)
                    self.writer.add_scalar(k+'_std' + '/' + self.dataset_name, std, global_step=global_step, walltime=cur_time_sec)
                    self.test_stats_buffer[k] = []

        # todo worker
        if self.is_worker:
            # plot normalized dualbound and gap auc
            for k, vals in self.tmp_stats_buffer.items():
                avg = np.mean(vals)
                std = np.std(vals)
                print('{}: {:.4f} | '.format(k, avg), end='')
                self.writer.add_scalar(k + '/' + self.dataset_name, avg, global_step=global_step, walltime=cur_time_sec)
                self.writer.add_scalar(k + '_std' + '/' + self.dataset_name, std, global_step=global_step, walltime=cur_time_sec)
                self.tmp_stats_buffer[k] = []

        # todo learner
        if self.is_learner:
            # log the average loss of the last training session
            print('Loss: {:.4f} | '.format(self.loss_moving_avg), end='')
            self.writer.add_scalar('Training_Loss', self.loss_moving_avg, global_step=global_step, walltime=cur_time_sec)
            print(f'Step: {self.num_env_steps_done} | ', end='')


        d = int(np.floor(cur_time_sec/(3600*24)))
        h = int(np.floor(cur_time_sec/3600) - 24*d)
        m = int(np.floor(cur_time_sec/60) - 60*(24*d + h))
        s = int(cur_time_sec) % 60
        print('Iteration Time: {:.1f}[sec]| '.format(cur_time_sec - self.last_time_sec), end='')
        print('Total Time: {}-{:02d}:{:02d}:{:02d}'.format(d, h, m, s))
        self.last_time_sec = cur_time_sec

    def init_figures(self, nrows=10, ncols=3, col_labels=['seed_i']*3, row_labels=['graph_i']*10):
        for figname in ['Dual_Bound_vs_LP_Iterations', 'Gap_vs_LP_Iterations']:
            fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)
            fig.set_size_inches(w=8, h=10)
            fig.set_tight_layout(True)
            self.figures[figname] = {'fig': fig, 'axes': axes}
        self.figures['nrows'] = nrows
        self.figures['ncols'] = ncols
        self.figures['col_labels'] = col_labels
        self.figures['row_labels'] = row_labels

    def add_episode_subplot(self, row, col):
        """
        plot the last episode curves to subplot in position (row, col)
        plot dqn agent dualbound/gap curves together with the baseline curves.
        should be called after each validation/test episode with row=graph_idx, col=seed_idx
        """
        dqn_lpiter, dqn_db, dqn_gap = self.episode_stats['lp_iterations'], self.episode_stats['dualbound'], self.episode_stats['gap']
        if dqn_lpiter[-1] < self.lp_iterations_limit:
            # extend curve to the limit
            dqn_lpiter = dqn_lpiter + [self.lp_iterations_limit]
            dqn_db = dqn_db + dqn_db[-1:]
            dqn_gap = dqn_gap + dqn_gap[-1:]
        bsl_lpiter = self.baseline['rootonly_stats'][self.scip_seed]['lp_iterations']
        bsl_db = self.baseline['rootonly_stats'][self.scip_seed]['dualbound']
        bsl_gap = self.baseline['rootonly_stats'][self.scip_seed]['gap']
        if bsl_lpiter[-1] < self.lp_iterations_limit:
            # extend curve to the limit
            bsl_lpiter = bsl_lpiter + [self.lp_iterations_limit]
            bsl_db = bsl_db + bsl_db[-1:]
            bsl_gap = bsl_gap + bsl_gap[-1:]
        # plot dualbound
        # fig = plt.figure()

        ax = self.figures['Dual_Bound_vs_LP_Iterations']['axes'][row, col]
        ax.plot(dqn_lpiter, dqn_db, 'b', label='DQN')
        ax.plot(bsl_lpiter, bsl_db, 'r', label='SCIP default')
        ax.plot([0, self.baseline['lp_iterations_limit']], [self.baseline['optimal_value']]*2, 'k', label='optimal value')
        # plt.legend()
        # plt.xlabel('LP Iterations')
        # plt.ylabel('Dualbound')
        # plt.title(f'SCIP Seed: {self.scip_seed}')
        # plt.setp([ax.get_xticklines() + ax.get_yticklines() + ax.get_xgridlines() + ax.get_ygridlines()], antialiased=False)
        # self.figures['Dual_Bound_vs_LP_Iterations'].append(fig)

        # plot gap

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        ax = self.figures['Gap_vs_LP_Iterations']['axes'][row, col]
        ax.plot(dqn_lpiter, dqn_gap, 'b', label='DQN')
        ax.plot(bsl_lpiter, bsl_gap, 'r', label='SCIP default')
        ax.plot([0, self.baseline['lp_iterations_limit']], [0, 0], 'k', label='optimal gap')
        # plt.setp([ax.get_xticklines() + ax.get_yticklines() + ax.get_xgridlines() + ax.get_ygridlines()], antialiased=False)
        # self.figures['Gap_vs_LP_Iterations'].append(fig)

    def decorate_figures(self, legend=True, col_labels=True, row_labels=True):
        """ save figures to png file """
        # decorate (title, labels etc.)
        nrows, ncols = self.figures['nrows'], self.figures['ncols']
        for figname in ['Dual_Bound_vs_LP_Iterations', 'Gap_vs_LP_Iterations']:
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
            'num_policy_updates': self.num_policy_updates,
            'i_episode': self.i_episode,
            'walltime_offset': time() - self.start_time + self.walltime_offset,
            'best_perf': self.best_perf,
            'loss_moving_avg': self.loss_moving_avg,
        }, filepath if filepath is not None else self.checkpoint_filepath)
        if self.hparams.get('verbose', 1) > 1:
            print('Saved checkpoint to: ', filepath if filepath is not None else self.checkpoint_filepath)

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
            print('Checkpoint file does not exist! starting from scratch.')
            return
        checkpoint = torch.load(self.checkpoint_filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_env_steps_done = checkpoint['num_env_steps_done']
        self.num_sgd_steps_done = checkpoint['num_sgd_steps_done']
        self.num_policy_updates = checkpoint['num_policy_updates']
        self.i_episode = checkpoint['i_episode']
        self.walltime_offset = checkpoint['walltime_offset']
        self.best_perf = checkpoint['best_perf']
        self.loss_moving_avg = checkpoint['loss_moving_avg']
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        print('Loaded checkpoint from: ', self.checkpoint_filepath)

    def load_datasets(self):
        """ load train/valid/test sets """
        hparams = self.hparams

        # datasets and baselines
        datasets = hparams['datasets']
        for dataset_name, dataset in datasets.items():
            datasets[dataset_name]['datadir'] = os.path.join(hparams['datadir'], dataset_name,
                                                       "barabasi-albert-n{}-m{}-weights-{}-seed{}".format(
                                                           dataset['graph_size'], dataset['barabasi_albert_m'],
                                                           dataset['weights'], dataset['dataset_generation_seed']))

            # read all graphs with their baselines from disk
            dataset['instances'] = []
            for filename in tqdm(os.listdir(datasets[dataset_name]['datadir']), desc=f'Loading {dataset_name}'):
                with open(os.path.join(datasets[dataset_name]['datadir'], filename), 'rb') as f:
                    G, baseline = pickle.load(f)
                    if baseline['is_optimal']:
                        dataset['instances'].append((G, baseline))
                    else:
                        print(filename, ' is not solved to optimality')
            dataset['num_instances'] = len(dataset['instances'])

        # for the validation and test datasets compute some metrics:
        for dataset_name, dataset in datasets.items():
            if dataset_name == 'trainset25':
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

        return datasets

    def initialize_training(self):
        # fix random seed for all experiment
        if self.hparams.get('seed', None) is not None:
            np.random.seed(self.hparams['seed'])
            torch.manual_seed(self.hparams['seed'])

        # initialize agent
        self.set_training_mode()
        if self.hparams.get('resume_training', False):
            self.load_checkpoint()

    def execute_episode(self, G, baseline, lp_iterations_limit, dataset_name, scip_seed=None):
        # create SCIP model for G
        hparams = self.hparams
        model, x, y = maxcut_mccormic_model(G, use_general_cuts=hparams.get('use_general_cuts',
                                                                            False))  # disable default cuts

        # include cycle inequalities separator with high priority
        cycle_sepa = MccormickCycleSeparator(G=G, x=x, y=y, name='MLCycles', hparams=hparams)
        model.includeSepa(cycle_sepa, 'MLCycles',
                          "Generate cycle inequalities for the MaxCut McCormic formulation",
                          priority=1000000, freq=1)

        # reset self to start a new episode
        self.init_episode(G, x, y, lp_iterations_limit, cut_generator=cycle_sepa, baseline=baseline,
                          dataset_name=dataset_name, scip_seed=scip_seed)

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
        self.finish_episode()

    def evaluate(self, datasets):
        # evaluate the model on the validation and test sets
        if self.num_policy_updates == 0:
            # wait until the model starts learning
            return
        global_step = self.num_policy_updates
        self.set_eval_mode()
        for dataset_name, dataset in datasets.items():
            if dataset_name == 'trainset25':
                continue
            if global_step % dataset['eval_interval'] == 0:
                self.init_figures(nrows=dataset['num_instances'],
                                  ncols=len(dataset['scip_seed']),
                                  col_labels=[f'Seed={seed}' for seed in dataset['scip_seed']],
                                  row_labels=[f'inst {inst_idx}' for inst_idx in
                                              range(dataset['num_instances'])])
                for inst_idx, (G, baseline) in enumerate(dataset['instances']):
                    for seed_idx, scip_seed in enumerate(dataset['scip_seed']):
                        self.execute_episode(G, baseline, dataset['lp_iterations_limit'], dataset_name=dataset_name,
                                             scip_seed=scip_seed)
                        self.add_episode_subplot(inst_idx, seed_idx)
                self.log_stats(save_best=(dataset_name[:8] == 'validset'), plot_figures=True)
        self.set_training_mode()

    def train_single_thread(self):
        self.initialize_training()
        datasets = self.load_datasets()

        trainset = datasets['trainset25']
        graph_indices = torch.randperm(trainset['num_instances'])
        hparams = self.hparams

        # training infinite loop
        for i_episode in range(self.i_episode + 1, hparams['num_episodes']):
            # sample graph randomly
            graph_idx = graph_indices[i_episode % len(graph_indices)]
            G, baseline = trainset['instances'][graph_idx]
            if hparams.get('debug', False):
                filename = os.listdir(datasets['trainset25']['datadir'])[graph_idx]
                filename = os.path.join(datasets['trainset25']['datadir'], filename)
                print(f'instance no. {graph_idx}, filename: {filename}')

            # execute episode and collect experience
            self.execute_episode(G, baseline, trainset['lp_iterations_limit'], dataset_name=trainset['dataset_name'])

            # perform 1 optimization step
            self.optimize_model()

            if self.num_sgd_steps_done % hparams.get('target_update_interval', 1000) == 0:
                self.update_target()

            if self.num_policy_updates > 0:
                global_step = self.num_policy_updates
                if global_step % hparams.get('log_interval', 100) == 0:
                    self.log_stats()

                # evaluate periodically
                self.evaluate(datasets)

                if global_step % hparams.get('checkpoint_interval', 100) == 0:
                    self.save_checkpoint()

            if i_episode % len(graph_indices) == 0:
                graph_indices = torch.randperm(trainset['num_instances'])

        return 0

    def test_distributed_functionality(self):
        """
        Tests the following functionality:
        1. Worker -> PER packets:
            a. store transitions in local buffer.
            b. encode local buffer periodically and send to PER.
            c. decode worker's packet and push transitions into the PER storage
        2. PER <-> Learner packets:
            a. encode transitions batch and send to Learner
            b. decode PER->Learner packet and perform SGD
            c. send back encoded new_priorities to PER, decode on PER side, and update priorities

        Learner -> ParamServer -> Worker packets are not tested, since we didn't touch this type of packets.
        """
        self.initialize_training()
        datasets = self.load_datasets()

        trainset = datasets['trainset25']
        graph_indices = torch.randperm(trainset['num_instances'])
        hparams = self.hparams

        # training infinite loop
        for i_episode in range(self.i_episode + 1, hparams['num_episodes']):
            # sample graph randomly
            graph_idx = graph_indices[i_episode % len(graph_indices)]
            G, baseline = trainset['instances'][graph_idx]
            if hparams.get('debug', False):
                filename = os.listdir(datasets['trainset25']['datadir'])[graph_idx]
                filename = os.path.join(datasets['trainset25']['datadir'], filename)
                print(f'instance no. {graph_idx}, filename: {filename}')

            # execute episodes, collect experience and store in self.local_buffer
            self.execute_episode(G, baseline, trainset['lp_iterations_limit'], dataset_name=trainset['dataset_name'])

            # if self.local_buffer is full enough, encode Worker->PER packet and send

            # decode Worker->PER packet and push into PER

            # encode PER->Learner packet and send to Learner

            # decode packet on Learner and perform SGD step

            # send back new_priorities to PER, and update

            # # perform 1 optimization step
            # self.optimize_model()

            # this part is the same as in train_single_thread()
            if self.num_sgd_steps_done % hparams.get('target_update_interval', 1000) == 0:
                self.update_target()

            if self.num_policy_updates > 0:
                global_step = self.num_policy_updates
                if global_step % hparams.get('log_interval', 100) == 0:
                    self.log_stats()

                # evaluate periodically
                self.evaluate(datasets)

                if global_step % hparams.get('checkpoint_interval', 100) == 0:
                    self.save_checkpoint()

            if i_episode % len(graph_indices) == 0:
                graph_indices = torch.randperm(trainset['num_instances'])

        return 0

