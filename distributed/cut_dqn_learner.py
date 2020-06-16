# distributedRL base class for learner - implements the distributed part
from agents.cut_dqn_agent import CutDQNAgent
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


class CutDQNLearner(CutDQNAgent):
    def __init__(self, hparams, use_gpu=True, gpu_id=None, **kwargs):
        super(CutDQNLearner, self).__init__(hparams=hparams, use_gpu=use_gpu, gpu_id=gpu_id, **kwargs)
        # set GDQN instance role
        self.is_learner = True
        self.is_worker = False
        self.is_tester = False

        # todo set learner specific logdir
        learner_logdir = os.path.join(self.logdir, 'tensorboard', 'learner')
        self.writer = SummaryWriter(log_dir=learner_logdir)
        self.checkpoint_filepath = os.path.join(self.logdir, 'learner_checkpoint.pt')

        self.replay_data_queue = deque(maxlen=hparams.get('max_pending_requests', 10)+1)
        self.new_priorities_queue = deque(maxlen=hparams.get('max_pending_requests', 10)+1)
        self.new_params_queue = deque(maxlen=hparams.get('max_pending_requests', 10)+1)
        # todo - inherited and recovered in GDQN
        #  self.num_param_updates = 0

        # number of SGD steps between each workers update
        self.param_sync_interval = hparams.get("param_sync_interval", 50)

        # initialize zmq sockets
        print("[Learner]: initializing sockets..")
        # for receiving batch from replay server
        context = zmq.Context()
        self.replay_server_2_learner_port = hparams["replay_server_2_learner_port"]
        self.replay_server_2_learner_socket = context.socket(zmq.PULL)
        self.replay_server_2_learner_socket.bind(f'tcp://127.0.0.1:{self.replay_server_2_learner_port}')
        # for sending back new priorities to replay server
        context = zmq.Context()
        self.learner_2_replay_server_port = hparams["learner_2_replay_server_port"]
        self.learner_2_replay_server_socket = context.socket(zmq.PUSH)
        self.learner_2_replay_server_socket.bind(f'tcp://127.0.0.1:{self.learner_2_replay_server_port}')
        # for publishing new params to workers
        context = zmq.Context()
        self.params_pubsub_port = hparams["params_pubsub_port"]
        self.params_pub_socket = context.socket(zmq.PUB)
        self.params_pub_socket.bind(f"tcp://127.0.0.1:{self.params_pubsub_port}")

    @staticmethod
    def params_to_numpy(model: torch.nn.Module):
        params = []
        # todo - why deepcopy fails on TypeError: can't pickle torch._C.ScriptFunction objects
        # new_model = deepcopy(model)
        # state_dict = new_model.cpu().state_dict()
        for param in model.state_dict().values():
            params.append(param.cpu().numpy())
        return params

    def get_params_packet(self, packet_id):
        """
        pack the learner params together with unique packet_id,
        which is essentially the self.num_param_updates counter.
        This packet_id will be used to synchronize the test-worker global_step robustly to failures.
        """
        model = self.get_model()  # return base class torch.nn.Module
        params = self.params_to_numpy(model)
        params_packet = (params, packet_id)
        params_packet = pa.serialize(params_packet).to_buffer()
        return params_packet

    def publish_params(self):
        if len(self.new_params_queue) > 0:
            params_packet = self.new_params_queue.popleft()  # thread-safe
            params_packet = pa.serialize(params_packet).to_buffer()
            self.params_pub_socket.send(params_packet)

    def prepare_new_params_to_workers(self):
        """
        Periodically snapshot the learner policy params,
        and push into new_params_queue with the corresponding num_params_update.
        In addition, checkpoint the model and statistics, to properly recover from failures.
        Both Learner and Worker classes will use this checkpoint for recovering.
        The test-worker instead has its own checkpoint, including the latest log stats and some more stuff.
        The test worker will synchronize to the learner state every params update, so it won't be
        affected from checkpointing separately.
        """
        if self.num_learning_steps_done > 0 and self.num_learning_steps_done % self.param_sync_interval == 0:
            self.num_param_updates += 1

            # prepare params_packet
            model = self.get_model()  # return base class torch.nn.Module
            params = self.params_to_numpy(model)
            params_packet = (params, int(self.num_param_updates))
            self.new_params_queue.append(params_packet)  # thread-safe

            # log stats here - to be synchronized with the workers and tester logs.
            # todo - if self.num_param_updates > 0 and self.num_param_updates % self.hparams.get('log_interval', 100) == 0:
            self.log_stats()
            self.save_checkpoint()

    @staticmethod
    def unpack_batch_packet(batch_packet):
        """ inverse operation to PERServer.get_replay_data_packet() """
        transition_numpy_tuples, weights_numpy, idxes, data_ids = pa.deserialize(batch_packet)
        weights = torch.from_numpy(weights_numpy)
        transitions = [Transition.from_numpy_tuple(npt) for npt in transition_numpy_tuples]
        return transitions, weights, idxes, data_ids

    def recv_batch(self, blocking=True):
        """
        Receive a batch from replay server.
        Return True if any batch received, otherwise False
        """
        received = False
        if blocking:
            batch_packet = self.replay_server_2_learner_socket.recv()
            received = True
        else:
            try:
                batch_packet = self.replay_server_2_learner_socket.recv(zmq.DONTWAIT)
            except zmq.Again:
                pass
        if received:
            batch = self.unpack_batch_packet(batch_packet)
            self.replay_data_queue.append(batch)
        return received

    @staticmethod
    def pack_priorities(priorities_message):
        priorities_packet = pa.serialize(priorities_message).to_buffer()
        return priorities_packet

    def send_new_priorities(self):
        if len(self.new_priorities_queue) > 0:
            new_priorities = self.new_priorities_queue.popleft()
            new_priors_packet = self.pack_priorities(new_priorities)
            self.learner_2_replay_server_socket.send(new_priors_packet)

    def run(self):
        time.sleep(3)
        while True:
            self.recv_batch()  # todo run in background
            replay_data = self.replay_data_queue.pop()
            new_priorities = self.learning_step(replay_data)
            self.new_priorities_queue.append(new_priorities)
            self.send_new_priorities()

            self.prepare_new_params_to_workers()
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

    def optimize_model(self):
        """
        Overrides GDQN.optimize_model()
        Instead of sampling from the local replay buffer like in the single thread GDQN,
        we pop one batch from replay_data_queue, process and push new priorities to new_priorities_queue.
        Sending those priorities, and updating the workers will be done asynchronously in a separate thread.
        """
        if len(self.replay_data_queue) > 0:
            transitions, weights, idxes, data_ids = self.replay_data_queue.popleft()  # thread-safe pop
            new_priorities = self.sgd_step(transitions, importance_sampling_correction_weights=weights)
            packet = (idxes, new_priorities, data_ids)
            self.new_priorities_queue.append(packet)  # thread safe append
            # todo verify
            if self.num_learning_steps_done % self.hparams.get('target_update_interval', 1000) == 0:
                self.update_target()

    def run_optimize_model(self):
        """
        asynchronously
        pop batch from replay_data_queue,
        push new priorities to queue
        and periodically push updated params to param_queue
        """
        time.sleep(3)
        while True:
            self.optimize_model()
            self.prepare_new_params_to_workers()
