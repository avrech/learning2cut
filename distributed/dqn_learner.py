# distributedRL base class for learner - implements the distributed part
import ray
from agents.dqn import GDQN
import time
from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import pyarrow as pa
import torch
import zmq
from utils.data import Transition
import os
from torch.utils.tensorboard import SummaryWriter


class Learner(ABC):
    def __init__(self, hparams={}, **kwargs):
        super(Learner, self).__init__(hparams=hparams, **kwargs)
        self.cfg = hparams
        self.replay_data_queue = deque(maxlen=hparams.get('max_pending_requests', 10)+1)
        self.new_priorities_queue = deque(maxlen=hparams.get('max_pending_requests', 10)+1)
        self.new_params_queue = deque(maxlen=hparams.get('max_pending_requests', 10)+1)

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
        if len(self.new_params_queue) > 0:
            params_packet = self.new_params_queue.popleft()
            self.pub_socket.send(params_packet)

    def push_new_params_to_queue(self):
        params_packet = self.get_params_packet()
        self.new_params_queue.append(params_packet)

    @staticmethod
    def unpack_batch_packet(batch_packet):
        """ inverse operation to PERServer.get_replay_data_packet() """
        transition_numpy_tuples, weights_numpy, idxes, data_ids = pa.deserialize(batch_packet)
        weights = torch.from_numpy(weights_numpy)
        transitions = [Transition.from_numpy_tuple(npt) for npt in transition_numpy_tuples]
        return transitions, weights, idxes, data_ids

    def recv_batch(self, blocking=True):
        if blocking:
            batch_packet = self.rep_socket.recv()
        else:
            try:
                batch_packet = self.rep_socket.recv(zmq.DONTWAIT)
            except zmq.Again:
                return
        batch = self.unpack_batch_packet(batch_packet)
        self.replay_data_queue.append(batch)

    @staticmethod
    def pack_priorities(priorities_message):
        priorities_packet = pa.serialize(priorities_message).to_buffer()
        return priorities_packet

    def send_new_priorities(self):
        if len(self.new_priorities_queue) > 0:
            new_priorities = self.new_priorities_queue.popleft()
            new_priors_packet = self.pack_priorities(new_priorities)
            self.rep_socket.send(new_priors_packet)

    def run(self):
        time.sleep(3)

        while True:
            self.recv_batch()  # todo run in background
            replay_data = self.replay_data_queue.pop()
            new_priorities = self.learning_step(replay_data)
            self.update_step = self.update_step + 1
            self.new_priorities_queue.append(new_priorities)
            self.send_new_priorities()

            if self.update_step % self.param_update_interval == 0:
                self.push_new_params_to_queue()
                self.publish_params()

    def run_io(self):
        """
        asynchronously receive data and return new priorities to replay server,
        and publish new params to workers
        """
        time.sleep(1)
        while True:
            self.recv_batch(blocking=False)
            self.send_new_priorities()
            self.publish_params()

    def run_optimize_model(self):
        """
        asynchronously
        pop batch from replay_data_queue,
        push new priorities to queue
        and periodically push updated params to param_queue
        """
        time.sleep(3)
        while True:
            if len(self.replay_data_queue) > 0:
                replay_data = self.replay_data_queue.popleft()  # thread-safe pop
                new_priorities = self.learning_step(replay_data)
                self.new_priorities_queue.append(new_priorities)
                self.update_step = self.update_step + 1
                if self.update_step % self.param_update_interval == 0:
                    self.push_new_params_to_queue()


class GDQNLearner(Learner, GDQN):
    def __init__(self, hparams, use_gpu=True, gpu_id=None, use_ray_gpu_id=False, **kwargs):
        if use_ray_gpu_id:
            gpu_id = ray.get_gpu_ids()[0]  # set the visible gpu id for the specific learner instance
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

    def learning_step(self, batch):
        """
        Get a list of transitions, weights and idxes. do DQN step as in DQN.optimize_model()
        update the weights, and send it back to PER to update weights
        transitions: a list of Transition objects
        weights: torch.tensor Apex weights
        idxes: indices of transitions in the buffer to update priorities after the SGD step
        """
        # todo: transition, weights, idxes = data
        transitions, weights, idxes, data_ids = batch
        # todo - reconstruct Transition list from the arrays received
        new_priorities = self.sgd_step(transitions, importance_sampling_correction_weights=weights)
        # todo - log stats and checkpoint
        if self.num_param_updates % self.hparams.get('log_interval', 100) == 0:
            self.log_stats()
        if self.num_param_updates % self.hparams.get('checkpoint_interval', 100) == 0:
            self.save_checkpoint()

        return idxes, new_priorities, data_ids

    def get_model(self):
        return self.policy_net


# # todo replace this with RayGDQNLearner = ray.remote(num_gpus=1, num_cpus=2)(GDQNLearner) in run script
# @ray.remote(num_gpus=1, num_cpus=2)
# class RayGDQNLearner(GDQNLearner):
#     """ Ray remote actor wrapper """
#     def __init__(self, hparams, use_gpu=True, gpu_id=None):
#         super().__init__(hparams=hparams, use_gpu=use_gpu, gpu_id=gpu_id)
