from copy import deepcopy
import ray
import torch
import torch.nn.functional as F
# distributedRL base class for learner - implements the distributed part
from common.abstract.learner import Learner
from agents.dqn import GDQN


@ray.remote(num_gpus=1)
class DQNLearner(Learner, GDQN):
    def __init__(self, hparams, **kwargs):
        # brain, cfg: dict, comm_config: dict - old distributedRL stuff
        super().__init__(brain=None, cfg=hparams, comm_config=hparams, hparams=hparams)

        # self.num_step = self.cfg["num_step"]
        # self.gamma = self.cfg["gamma"]
        # self.tau = self.cfg["tau"]
        # self.network = self.brain[0]
        # self.network.to(self.device)
        # self.target_network = self.brain[1]
        # self.target_network.to(self.device)
        # self.network_optimizer = torch.optim.Adam(
        #     self.network.parameters(), lr=self.cfg["learning_rate"]
        # )
        # self.target_optimizer = torch.optim.Adam(
        #     self.target_network.parameters(), lr=self.cfg["learning_rate"]
        # )

    def write_log(self):
        # todo - call DQN.log_stats() or modify to log the relevant metrics
        print("TODO: incorporate Tensorboard...")

    def learning_step(self, data: tuple):
        """
        Get a list of transitions, weights and idxes. do DQN step as in DQN.optimize_model()
        update the weights, and send it back to PER to update weights
        transitions: a list of Transition objects
        weights: Apex weights
        idxes: indices of transitions in the buffer to update priorities after the SGD step
        """
        # todo: transition, weights, idxes = data
        transitions, weights, idxes = data
        # todo - reconstruct Transition list from the arrays received
        loss, new_priorities = self.sgd_step(transitions)
        # todo - call DQN sgd_step


        # ORIGINAL CODE
        # states, actions, rewards, next_states, dones, weights, idxes = data
        # states = torch.FloatTensor(states).to(self.device)
        # actions = torch.LongTensor(actions).to(self.device)
        # rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
        # next_states = torch.FloatTensor(next_states).to(self.device)
        # dones = torch.FloatTensor(dones).to(self.device).view(-1, 1)
        #
        # curr_q1 = self.network.forward(states).gather(1, actions.unsqueeze(1))
        # curr_q2 = self.target_network.forward(states).gather(1, actions.unsqueeze(1))
        #
        # bootstrap_q = torch.min(
        #     torch.max(self.network(next_states), 1)[0],
        #     torch.max(self.target_network(next_states), 1)[0],
        # )  # todo - what is this formula ?
        #
        # bootstrap_q = bootstrap_q.view(bootstrap_q.size(0), 1)
        # target_q = rewards + (1 - dones) * self.gamma ** self.num_step * bootstrap_q
        # weights = torch.FloatTensor(weights).to(self.device).mean()
        #
        # loss1 = weights * F.mse_loss(curr_q1, target_q.detach())  # importance sampling correction
        # loss2 = weights * F.mse_loss(curr_q2, target_q.detach())
        #
        # self.network_optimizer.zero_grad()
        # loss1.backward()
        # self.network_optimizer.step()
        #
        # self.target_optimizer.zero_grad()
        # loss2.backward()
        # self.target_optimizer.step()
        #
        # step_info = (loss1, loss2)
        # new_priorities = torch.abs(target_q - curr_q1).detach().view(-1)
        # new_priorities = torch.clamp(new_priorities, min=1e-8)
        # new_priorities = new_priorities.cpu().numpy().tolist()
        step_info = (loss, )
        return step_info, idxes, new_priorities

    def get_params(self):
        # model = deepcopy(self.policy_net.cpu())
        # model = model.cpu()
        return self.params_to_numpy(self.policy_net)