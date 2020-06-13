# distributedRL base class for learner - implements the distributed part
import ray
from agents.dqn import GDQN
import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import Union
import numpy as np
import pyarrow as pa
import torch
import zmq
from utils.data import Transition
from collections import namedtuple
import os
from torch.utils.tensorboard import SummaryWriter
PriorityMessage = namedtuple('PriorityMessage', ('idxes', 'new_priorities'))


class Learner(ABC):
    def __init__(self, hparams={}, **kwargs):
        super(Learner, self).__init__(hparams=hparams, **kwargs)
        self.cfg = hparams
        self.replay_data_queue = deque(maxlen=1000)

        # unpack communication configs
        self.param_update_interval = self.cfg.get("param_update_interval", 50)
        self.repreq_port = hparams["repreq_port"]
        self.pubsub_port = hparams["pubsub_port"]

        # initialize zmq sockets
        print("[Learner]: initializing sockets..")
        self.pub_socket = None
        self.rep_socket = None
        self.initialize_sockets()
        self.update_step = 0

    @abstractmethod
    def write_log(self):
        pass

    @abstractmethod
    def learning_step(self, data: tuple):
        pass

    @abstractmethod
    def get_model(self) -> np.ndarray:
        """Return model params for synchronization"""
        pass

    @staticmethod
    def params_to_numpy(model: torch.nn.Module):
        params = []
        # todo - why deepcopy fails on TypeError: can't pickle torch._C.ScriptFunction objects
        # new_model = deepcopy(model)
        # state_dict = new_model.cpu().state_dict()
        for param in model.state_dict().values():
            params.append(param.cpu().numpy())
        return params

    def initialize_sockets(self):
        # For sending new params to workers
        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://127.0.0.1:{self.pubsub_port}")

        # For receiving batch from, sending new priorities to Buffer # write another with PUSH/PULL for non PER version
        context = zmq.Context()
        self.rep_socket = context.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://127.0.0.1:{self.repreq_port}")

    def get_params_packet(self):
        model = self.get_model()
        params = self.params_to_numpy(model)
        params_packet = pa.serialize(params).to_buffer()
        return params_packet

    def publish_params(self):
        params_packet = self.get_params_packet()
        self.pub_socket.send(params_packet)

    @staticmethod
    def unpack_batch_packet(batch_packet):
        """ inverse operation to PERServer.get_replay_data_packet() """
        transition_numpy_tuples, weights_numpy, idxes = pa.deserialize(batch_packet)
        weights = torch.from_numpy(weights_numpy)
        transitions = [Transition.from_numpy_tuple(npt) for npt in transition_numpy_tuples]
        return transitions, weights, idxes

    def recv_batch(self):
        batch_packet = self.rep_socket.recv()
        batch = self.unpack_batch_packet(batch_packet)
        self.replay_data_queue.append(batch)

    @staticmethod
    def pack_priorities(priorities_message: PriorityMessage):
        priorities_packet = pa.serialize(priorities_message).to_buffer()
        return priorities_packet

    def send_new_priorities(self, priority_message):
        new_priors_packet = self.pack_priorities(priority_message)
        self.rep_socket.send(new_priors_packet)

    def run(self):
        time.sleep(3)
        self.update_step = 0
        while True:
            self.recv_batch()
            replay_data = self.replay_data_queue.pop()
            idxes, priorities = self.learning_step(replay_data)
            self.update_step = self.update_step + 1
            self.send_new_priorities((idxes, priorities))

            if self.update_step % self.param_update_interval == 0:
                self.publish_params()


class GDQNLearner(Learner, GDQN):
    def __init__(self, hparams, use_gpu=True, gpu_id=None, **kwargs):
        # brain, cfg: dict, comm_config: dict - old distributedRL stuff
        super(GDQNLearner, self).__init__(hparams=hparams, use_gpu=use_gpu, gpu_id=gpu_id, **kwargs)
        # set GDQN instance role
        self.is_learner = True
        self.is_worker = False
        self.is_tester = False
        # set worker specific logdir
        self.logdir = os.path.join(self.logdir, 'learner')
        self.writer = SummaryWriter(log_dir=self.logdir)
        self.checkpoint_filepath = os.path.join(self.logdir, 'checkpoint.pt')
        # self.device = torch.device(f"cuda:{hparams['gpu_id']}" if torch.cuda.is_available() and hparams.get('gpu_id', None) is not None and hparams.get('learner_device', 'cpu') =='cuda' else "cpu")
        # self.policy_net = self.policy_net.to(self.device)
        # self.target_net = self.target_net.to(self.device)

    def write_log(self):
        # todo - call DQN.log_stats() or modify to log the relevant metrics
        print("TODO: incorporate Tensorboard...")

    def learning_step(self, data: tuple):
        """
        Get a list of transitions, weights and idxes. do DQN step as in DQN.optimize_model()
        update the weights, and send it back to PER to update weights
        transitions: a list of Transition objects
        weights: torch.tensor Apex weights
        idxes: indices of transitions in the buffer to update priorities after the SGD step
        """
        # todo: transition, weights, idxes = data
        transitions, weights, idxes = data
        # todo - reconstruct Transition list from the arrays received
        new_priorities = self.sgd_step(transitions, importance_sampling_correction_weights=weights)
        # todo - log stats and checkpoint
        if self.num_param_updates % self.hparams.get('log_interval', 100) == 0:
            self.log_stats()
        if self.num_param_updates % self.hparams.get('checkpoint_interval', 100) == 0:
            self.save_checkpoint()

        return idxes, new_priorities

    def get_model(self):
        return self.policy_net


@ray.remote(num_gpus=1)
class RayGDQNLearner(GDQNLearner):
    """ Ray remote actor wrapper """
    def __init__(self, hparams, **kwargs):
        super(RayGDQNLearner, self).__init__(hparams=hparams)
