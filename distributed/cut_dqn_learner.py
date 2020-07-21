# distributedRL base class for learner - implements the distributed part
from agents.cut_dqn_agent import CutDQNAgent
import time
from collections import deque
import pyarrow as pa
import torch
import zmq
from utils.data import Transition
import os
from torch.utils.tensorboard import SummaryWriter
import threading


class CutDQNLearner(CutDQNAgent):
    """
    This actor executes in parallel two tasks:
        a. receive batch, send priorities and publish params.
        b. optimize model.
    According to
    https://stackoverflow.com/questions/54937456/how-to-make-an-actor-do-two-things-simultaneously
    Ray's Actor.remote() cannot execute to tasks in parallel. We adopted the suggested solution in the
    link above, and run the IO in a background process on Actor instantiation.
    The main Actor's thread will optimize the model using the GPU.
    """
    def __init__(self, hparams, use_gpu=True, gpu_id=None, run_io=False, **kwargs):
        super(CutDQNLearner, self).__init__(hparams=hparams, use_gpu=use_gpu, gpu_id=gpu_id, **kwargs)
        # set GDQN instance role
        self.is_learner = True
        self.is_worker = False
        self.is_tester = False

        # idle time monitor
        self.idle_time_sec = 0

        # set learner specific logdir
        learner_logdir = os.path.join(self.logdir, 'tensorboard', 'learner')
        self.writer = SummaryWriter(log_dir=learner_logdir)
        # set checkpoint file path for learner and workers
        self.checkpoint_filepath = os.path.join(self.logdir, 'learner_checkpoint.pt')

        self.replay_data_queue = deque(maxlen=hparams.get('max_pending_requests', 10)+1)
        self.new_priorities_queue = deque(maxlen=hparams.get('max_pending_requests', 10)+1)
        self.new_params_queue = deque(maxlen=hparams.get('max_pending_requests', 10)+1)
        # todo - inherited and recovered in CutDQNAgent
        #  self.num_param_updates = 0

        # number of SGD steps between each workers update
        self.param_sync_interval = hparams.get("param_sync_interval", 50)
        self.print_prefix = '[Learner] '
        # initialize zmq sockets
        print(self.print_prefix, "initializing sockets..")
        # for receiving batch from replay server
        context = zmq.Context()
        self.replay_server_2_learner_port = hparams["replay_server_2_learner_port"]
        self.replay_server_2_learner_socket = context.socket(zmq.PULL)
        self.replay_server_2_learner_socket.bind(f'tcp://127.0.0.1:{self.replay_server_2_learner_port}')
        # for sending back new priorities to replay server
        context = zmq.Context()
        self.learner_2_replay_server_port = hparams["learner_2_replay_server_port"]
        self.learner_2_replay_server_socket = context.socket(zmq.PUSH)
        self.learner_2_replay_server_socket.connect(f'tcp://127.0.0.1:{self.learner_2_replay_server_port}')
        # for publishing new params to workers
        context = zmq.Context()
        self.params_pubsub_port = hparams["learner_2_workers_pubsub_port"]
        self.params_pub_socket = context.socket(zmq.PUB)
        self.params_pub_socket.connect(f"tcp://127.0.0.1:{self.params_pubsub_port}")
        self.initialize_training()
        if run_io:
            self.background_io = threading.Thread(target=self.run_io, args=())
            self.background_io.start()

    @staticmethod
    def params_to_numpy(model: torch.nn.Module):
        params = []
        # todo - why deepcopy fails on TypeError: can't pickle torch._C.ScriptFunction objects
        # new_model = deepcopy(model)
        # state_dict = new_model.cpu().state_dict()
        for param in model.state_dict().values():
            params.append(param.cpu().numpy())
        return params

    # todo - this method is unused. remove it
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
            # attach a 'topic' to the packet and send
            message = pa.serialize(('new_params', params_packet)).to_buffer()
            self.params_pub_socket.send(message)

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
        if self.num_sgd_steps_done > 0 and self.num_sgd_steps_done % self.param_sync_interval == 0:
            self.num_param_updates += 1

            # prepare params_packet
            model = self.get_model()  # return base class torch.nn.Module
            params = self.params_to_numpy(model)
            params_packet = (params, int(self.num_param_updates))
            self.new_params_queue.append(params_packet)  # thread-safe

            # log stats here - to be synchronized with the workers and tester logs.
            # todo - if self.num_param_updates > 0 and self.num_param_updates % self.hparams.get('log_interval', 100) == 0:
            cur_time_sec = time.time() - self.start_time + self.walltime_offset
            info = {'Idle time': '{:.2f}%'.format(self.idle_time_sec / (cur_time_sec - self.last_time_sec))}
            self.log_stats(info=info)
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
                received = True
            except zmq.Again:
                pass
        if received:
            batch = self.unpack_batch_packet(batch_packet)
            self.replay_data_queue.append(batch)
        return received

    def send_new_priorities(self):
        if len(self.new_priorities_queue) > 0:
            new_priorities = self.new_priorities_queue.popleft()
            new_priorities_packet = pa.serialize(new_priorities).to_buffer()
            self.learner_2_replay_server_socket.send(new_priorities_packet)

    # old version
    # def run(self):
    #     self.initialize_training()
    #     time.sleep(3)
    #     while True:
    #         self.recv_batch()  # todo run in background
    #         replay_data = self.replay_data_queue.pop()
    #         new_priorities = self.learning_step(replay_data)
    #         self.new_priorities_queue.append(new_priorities)
    #         self.send_new_priorities()
    #
    #         self.prepare_new_params_to_workers()
    #         self.publish_params()

    def run_io(self):
        """
        asynchronously receive data and return new priorities to replay server,
        and publish new params to workers
        """
        print(self.print_prefix + 'started io process in background...')
        print(self.print_prefix + 'sending "restart" message to replay_server...')
        restart_message = pa.serialize("restart").to_buffer()
        self.learner_2_replay_server_socket.send(restart_message)

        time.sleep(2)
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
        # wait until there is any batch ready for processing, and count the idle time
        idle_time_start = time.time()
        idle_time_end = time.time()
        while not self.replay_data_queue:
            idle_time_end = time.time()
        self.idle_time_sec = idle_time_end - idle_time_start

        # pop one batch and perform one SGD step
        transitions, weights, idxes, data_ids = self.replay_data_queue.popleft()  # thread-safe pop
        is_demonstration = idxes < self.hparams.get('replay_buffer_n_demonstrations', 0)
        new_priorities = self.sgd_step(transitions, importance_sampling_correction_weights=weights, is_demonstration=is_demonstration)  # todo- sgd demonstrations
        packet = (idxes, new_priorities, data_ids)
        self.new_priorities_queue.append(packet)  # thread safe append
        # todo verify
        if self.num_sgd_steps_done % self.hparams.get('target_update_interval', 1000) == 0:
            self.update_target()

    def run(self):
        """
        asynchronously
        pop batch from replay_data_queue,
        push new priorities to queue
        and periodically push updated params to param_queue
        """
        print(self.print_prefix + 'started main optimization loop')
        time.sleep(1)
        while True:
            self.optimize_model()
            self.prepare_new_params_to_workers()

