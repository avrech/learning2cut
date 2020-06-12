""" Worker class copied and modified from https://github.com/cyoon1729/distributedRL """
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque
import numpy as np
import pyarrow as pa
import torch
import torch.nn as nn
import zmq
import ray
from agents.dqn import GDQN
from utils.data import Transition
import os
from torch.utils.tensorboard import SummaryWriter


class Worker(ABC):
    def __init__(self,
                 worker_id: int,
                 hparams: dict,
                 **kwargs
    ):
        self.worker_id = worker_id
        self.cfg = hparams
        self.device = hparams.get("worker_device", 'cpu')

        # unpack communication configs
        self.pubsub_port = hparams["pubsub_port"]
        self.pullpush_port = hparams["pullpush_port"]

        # initialize zmq sockets
        print(f"[Worker {self.worker_id}]: initializing sockets..")
        self.sub_socket = None
        self.push_socket = None
        self.initialize_sockets()

    @abstractmethod
    def write_log(self):
        """Log performance (e.g. using Tensorboard)"""
        pass

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action with worker's brain"""
        pass

    @abstractmethod
    def preprocess_data(self, **kwargs) -> list:
        """Preprocess collected data if necessary (e.g. n-step)"""
        pass

    @abstractmethod
    def collect_data(self) -> list:
        """Run environment and collect data until stopping criterion satisfied"""
        pass

    @abstractmethod
    def test_run(self):
        """Specifically for the performance-testing worker"""
        pass

    @abstractmethod
    def get_model(self):
        pass

    def synchronize_params(self, new_params_packet):
        """Synchronize worker's policy_net with parameter server"""
        new_params = pa.deserialize(new_params_packet)
        model = self.get_model()
        for param, new_param in zip(model.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)

    def initialize_sockets(self):
        # for receiving params from learner
        context = zmq.Context()
        self.sub_socket = context.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.sub_socket.connect(f"tcp://127.0.0.1:{self.pubsub_port}")

        # for sending replay data to buffer
        time.sleep(1)
        context = zmq.Context()
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://127.0.0.1:{self.pullpush_port}")

    @staticmethod
    @abstractmethod
    def pack_replay_data(replay_data):
        """
        Pack replay_data as standard python objects list, tuple, numpy.array
        """
        pass

    def send_replay_data(self, replay_data):
        replay_data_packet = self.pack_replay_data(replay_data)
        self.push_socket.send(replay_data_packet)

    def receive_new_params(self):
        new_params_packet = False
        try:
            new_params_packet = self.sub_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            return False

        if new_params_packet:
            self.synchronize_params(new_params_packet)
            return True

    def run(self):
        while True:
            replay_data = self.collect_data()
            self.send_replay_data(replay_data)
            self.receive_new_params()
            # todo - consider checkpointing


class ApeXWorker(Worker):
    """Abstract class for ApeX distrbuted workers """

    def __init__(self,
                 worker_id: int,
                 worker_brain: nn.Module,
                 cfg: dict,
                 comm_cfg: dict
                 ):
        super().__init__(worker_id, worker_brain, cfg, comm_cfg)
        self.nstep_queue = deque(maxlen=self.cfg["num_step"])
        self.worker_buffer_size = self.cfg["worker_buffer_size"]
        self.gamma = self.cfg["gamma"]
        self.num_step = self.cfg["num_step"]

    def preprocess_data(self, nstepqueue: Deque) -> tuple:
        # todo - replace by compute_reward_and stats
        discounted_reward = 0
        _, _, _, last_state, done = nstepqueue[-1]
        for transition in list(reversed(nstepqueue)):
            state, action, reward, _, _ = transition
            discounted_reward = reward + self.gamma * discounted_reward
        nstep_data = (state, action, discounted_reward, last_state, done)

        q_value = self.policy_net.forward(
            torch.FloatTensor(state).unsqueeze(0).to(self.device)
        )[0][action]

        bootstrap_q = torch.max(
            self.policy_net.forward(
                torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            ),
            1,
        )

        target_q_value = (
            discounted_reward + self.gamma ** self.num_step * bootstrap_q[0]
        )

        priority_value = torch.abs(target_q_value - q_value).detach().view(-1)
        priority_value = torch.clamp(priority_value, min=1e-8)
        priority_value = priority_value.cpu().numpy().tolist()

        return nstep_data, priority_value

    def collect_data(self, verbose=False):
        # todo - replace by execute episode or more generally dqn experiment loop.
        """Fill worker buffer until some stopping criterion is satisfied"""
        local_buffer = []
        nstep_queue = deque(maxlen=self.num_step)

        while len(local_buffer) < self.worker_buffer_size:
            episode_reward = 0
            done = False
            state = self.env.reset()
            while not done:
                action = self.select_action(state)
                transition = self.environment_step(state, action)
                next_state = transition[-2]
                done = transition[-1]
                reward = transition[-3]
                episode_reward = episode_reward + reward

                nstep_queue.append(transition)
                if (len(nstep_queue) == self.num_step) or done:
                    nstep_data, priorities = self.preprocess_data(nstep_queue)
                    local_buffer.append([nstep_data, priorities])

                state = next_state

            if verbose:
                print(f"Worker {self.worker_id}: {episode_reward}")

        return local_buffer


@ray.remote
class GDQNWorker(Worker, GDQN):
    def __init__(self,
                 worker_id,
                 hparams,
                 is_tester=False,
                 **kwargs
                 ):
        super().__init__(worker_id=worker_id, hparams=hparams)
        # set GDQN instance role
        self.is_learner = False
        self.is_worker = not is_tester
        self.is_tester = is_tester
        # set worker specific logdir
        self.logdir = os.path.join(self.logdir, f'worker-{worker_id}')
        self.writer = SummaryWriter(log_dir=self.logdir)
        self.checkpoint_filepath = os.path.join(self.logdir, 'checkpoint.pt')

    def select_action(self, state: np.ndarray) -> np.ndarray:
        print('not relevant - do nothing!')
        raise NotImplementedError

    def environment_step(self, state: np.ndarray, action: np.ndarray) -> tuple:
        # next_state, reward, done, _ = self.env.step(action)
        # return (state, action, reward, next_state, done)
        print('not relevant - do nothing!')
        raise NotImplementedError

    def write_log(self):
        print("TODO: include Tensorboard..")

    def test_run(self):
        # self.eps_greedy = 0
        update_step = 0
        update_interval = self.cfg["param_update_interval"]
        self.initialize_training()
        datasets = self.load_datasets()
        while True:
            if self.receive_new_params():
                # todo - run here eval episodes (done)
                #  update GDQN such that it logs metrics according to this update_step.
                # self.num_policy_updates should be incremented in receive_new_params()
                # every time new params are available
                # todo - verify behavior
                self.evaluate(datasets)
                update_step = update_step + update_interval
                # todo checkpoint
                self.save_checkpoint()

    def preprocess_data(self, nstepqueue: Deque) -> tuple:
        print('not relevant - do nothing!')
        raise NotImplementedError

    def collect_data(self, verbose=False):
        # todo
        #  - replace by execute episode
        #  - or more generally dqn experiment loop.
        """Fill local buffer until some stopping criterion is satisfied"""
        local_buffer = []
        trainset = self.datasets['trainset25']
        while len(local_buffer) < self.hparams.get('local_buffer_size'):
            # sample graph randomly
            graph_idx = self.graph_indices[self.i_episode + 1 % len(self.graph_indices)]
            G, baseline = trainset['instances'][graph_idx]

            # execute episodes, collect experience and append to local_buffer
            trajectory = self.execute_episode(G, baseline, trainset['lp_iterations_limit'],
                                              dataset_name=trainset['dataset_name'])
            local_buffer += trajectory
            if self.i_episode + 1 % len(self.graph_indices) == 0:
                self.graph_indices = torch.randperm(trainset['num_instances'])
        return local_buffer

    @staticmethod
    def pack_replay_data(replay_data: [(Transition, np.ndarray)]):
        """
        Convert a list of (Transition, initial_weights) to list of (TransitionNumpyTuple, initial_priorities.numpy())
        :param replay_data: list of (Transition, initial_priorities)
        :return:
        """
        replay_data_packet = []
        for transition, initial_priority in replay_data:
            replay_data_packet.append((transition.to_numpy_tuple(), initial_priority))
        replay_data_packet = pa.serialize(replay_data_packet).to_buffer()
        return replay_data_packet

    def receive_new_params(self):
        updated = super().receive_new_params()
        if updated:
            self.num_param_updates += 1
        return updated

    def get_model(self):
        return self.policy_net

