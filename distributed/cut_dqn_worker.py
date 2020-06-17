""" Worker class copied and modified from https://github.com/cyoon1729/distributedRL """
import pyarrow as pa
import torch
import zmq
from agents.cut_dqn_agent import CutDQNAgent
import os
from torch.utils.tensorboard import SummaryWriter


class CutDQNWorker(CutDQNAgent):
    def __init__(self,
                 worker_id,
                 hparams,
                 is_tester=False,
                 use_gpu=False,
                 gpu_id=None,
                 **kwargs
                 ):
        super(CutDQNWorker, self).__init__(hparams=hparams, use_gpu=use_gpu, gpu_id=gpu_id, **kwargs)
        # set GDQN instance role
        self.worker_id = worker_id
        self.is_learner = False
        self.is_worker = not is_tester
        self.is_tester = is_tester

        # set worker specific tensorboard logdir
        worker_logdir = os.path.join(self.logdir, 'tensorboard', f'worker-{worker_id}')
        self.writer = SummaryWriter(log_dir=worker_logdir)
        # set special checkpoint file for tester (workers use the learner checkpoints)
        if is_tester:
            self.checkpoint_filepath = os.path.join(self.logdir, 'tester_checkpoint.pt')
        else:
            self.checkpoint_filepath = os.path.join(self.logdir, 'learner_checkpoint.pt')

        # initialize zmq sockets
        # use socket.connect() instead of .bind() because workers are the least stable part in the system
        # (not supposed to but rather suspected to be)
        print(f"[Worker {self.worker_id}]: initializing sockets..")
        # for receiving params from learner
        context = zmq.Context()
        self.params_pubsub_port = hparams["params_pubsub_port"]
        self.params_sub_socket = context.socket(zmq.SUB)
        self.params_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.params_sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.params_sub_socket.connect(f"tcp://127.0.0.1:{self.params_pubsub_port}")

        # for sending replay data to buffer
        if self.is_worker:
            context = zmq.Context()
            self.workers_2_replay_server_port = hparams["workers_2_replay_server_port"]
            self.worker_2_replay_server_socket = context.socket(zmq.PUSH)
            self.worker_2_replay_server_socket.connect(f'tcp://127.0.0.1:{self.workers_2_replay_server_port}')

    def synchronize_params(self, new_params_packet):
        """Synchronize worker's policy_net with learner's policy_net params """
        new_params, params_id = pa.deserialize(new_params_packet)
        model = self.get_model()
        for param, new_param in zip(model.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)
        # synchronize the global step counter GDQN.num_param_updates with the value arrived from learner.
        # this makes GDQN.log_stats() robust to Worker failures, and useful in recovering and resume training.
        self.num_param_updates = params_id
        # test should evaluate model here and then log stats.
        # workers should log stats before synchronizing, to plot the statistics collected by the previous policy,
        # together with the previous policy's params_id.

    def send_replay_data(self, replay_data):
        replay_data_packet = self.pack_replay_data(replay_data)
        self.worker_2_replay_server_socket.send(replay_data_packet)

    def recv_new_params(self):
        """
        receive the latest published params.
        In a case several param packets are waiting,
        receive all of them and take the latest one.
        """
        received = False
        while True:
            try:
                # if there is a waiting packet
                new_params_packet = self.params_sub_socket.recv(zmq.DONTWAIT)
                received = True
            except zmq.Again:
                # no packets are waiting
                break  # and go to synchronize

        if received:
            self.synchronize_params(new_params_packet)
        return received

    def run(self):
        self.initialize_training()
        self.load_datasets()
        while True:
            replay_data = self.collect_data()
            self.send_replay_data(replay_data)
            received_new_params = self.recv_new_params()
            if received_new_params:
                # todo - Log stats with global_step == num_param_updates-1 (== the previous params_id).
                #        This is because the current stats are related to the previous policy.
                #        In addition, maybe workers shouldn't log stats every update ?
                # if self.num_param_updates > 0 and self.num_param_updates % self.hparams['log_interval'] == 0:
                self.log_stats(global_step=self.num_param_updates-1, print_prefix=f'[Worker {self.worker_id}]\t')

    def test_run(self):
        # self.eps_greedy = 0
        self.initialize_training()
        datasets = self.load_datasets()
        while True:
            if self.recv_new_params():
                # todo consider not ignoring eval interval
                self.evaluate(datasets, ignore_eval_interval=True, print_prefix='[Tester]\t')
                self.save_checkpoint()

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
    def pack_replay_data(replay_data):
        """
        Convert a list of (Transition, initial_weights) to list of (TransitionNumpyTuple, initial_priorities.numpy())
        :param replay_data: list of (Transition, float initial_priority)
        :return:
        """
        replay_data_packet = []
        for transition, initial_priority in replay_data:
            replay_data_packet.append((transition.to_numpy_tuple(), initial_priority))
        replay_data_packet = pa.serialize(replay_data_packet).to_buffer()
        return replay_data_packet
