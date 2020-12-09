""" Worker class copied and modified from https://github.com/cyoon1729/distributedRL """
import pyarrow as pa
import torch
import zmq
from agents.cut_dqn_agent import CutDQNAgent
import os
# from torch.utils.tensorboard import SummaryWriter
import wandb
import pickle


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
        self.generate_demonstration_data = False

        # # # set worker specific tensorboard logdir
        # # worker_logdir = os.path.join(self.run_dir, 'tensorboard', f'worker-{worker_id}')  # todo remove after wandb is verified
        # # self.writer = SummaryWriter(log_dir=worker_logdir)  # todo remove after wandb is verified
        # wandb.init(resume='allow', # hparams['resume'],
        #            id=hparams['run_id'],
        #            project=hparams['project'],
        #            reinit=True  # todo for distributed_unittest.py
        #            )

        # set special checkpoint file for tester (workers use the learner checkpoints)
        if is_tester:
            self.checkpoint_filepath = os.path.join(self.run_dir, 'tester_checkpoint.pt')
        else:
            self.checkpoint_filepath = os.path.join(self.run_dir, 'learner_checkpoint.pt')

        self.print_prefix = f'[Worker {self.worker_id}] '
        # initialize zmq sockets
        # use socket.connect() instead of .bind() because workers are the least stable part in the system
        # (not supposed to but rather suspected to be)
        print(self.print_prefix, "initializing sockets..")
        # for receiving params from learner and requests from replay server
        context = zmq.Context()
        self.send_2_apex_socket = context.socket(zmq.PUSH)  # for sending logs
        self.sub_socket = context.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all topics
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)  # keep only last message received
        # connect to the main apex process
        self.send_2_apex_socket.connect(f'tcp://127.0.0.1:{hparams["com"]["apex_port"]}')
        self.print(f'connecting to apex_port: {hparams["com"]["apex_port"]}')

        # connect to learner pub socket
        self.sub_socket.connect(f'tcp://127.0.0.1:{hparams["com"]["learner_2_workers_pubsub_port"]}')
        self.print(f'connecting to learner_2_workers_pubsub_port: {hparams["com"]["learner_2_workers_pubsub_port"]}')

        # connect to replay_server pub socket
        self.sub_socket.connect(f'tcp://127.0.0.1:{hparams["com"]["replay_server_2_workers_pubsub_port"]}')
        self.print(f'connecting to replay_server_2_workers_pubsub_port: {hparams["com"]["replay_server_2_workers_pubsub_port"]}')

        # for sending replay data to buffer
        if self.is_worker:
            context = zmq.Context()
            self.worker_2_replay_server_socket = context.socket(zmq.PUSH)
            self.worker_2_replay_server_socket.connect(f'tcp://127.0.0.1:{hparams["com"]["workers_2_replay_server_port"]}')
            self.print(f'connecting to workers_2_replay_server_port: {hparams["com"]["workers_2_replay_server_port"]}')

        # save pid to run_dir
        pid = os.getpid()
        pid_file = os.path.join(hparams["run_dir"], f'{self.actor_name}_pid.txt')
        self.print(f'saving pid {pid} to {pid_file}')
        with open(pid_file, 'w') as f:
            f.writelines(str(pid) + '\n')

    @property
    def actor_name(self):
        return "tester" if self.is_tester else f"worker_{self.worker_id}"

    def synchronize_params(self, new_params_packet):
        """Synchronize worker's policy_net with learner's policy_net params """
        new_params, params_id = new_params_packet
        model = self.get_model()
        for param, new_param in zip(model.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)
        # synchronize the global step counter self.num_param_updates with the value arrived from learner.
        # this makes self.log_stats() robust to Worker failures, missed packets and in resumed training.
        assert self.num_param_updates < params_id, f"global step counter is not consistent between learner and worker: TestWorker.num_param_updates={self.num_param_updates}, ParamsID={params_id}"
        self.num_param_updates = params_id
        # test should evaluate model here and then log stats.
        # workers should log stats before synchronizing, to plot the statistics collected by the previous policy,
        # together with the previous policy's params_id.

    def send_replay_data(self, replay_data):
        replay_data_packet = self.pack_replay_data(replay_data)
        self.worker_2_replay_server_socket.send(replay_data_packet)

    def read_message(self, message):
        new_params_packet = None
        message = pa.deserialize(message)
        if message[0] == 'new_params':
            new_params_packet = message[1]
        elif message[0] == 'generate_demonstration_data':
            print(self.print_prefix, 'collecting demonstration data')
            self.generate_demonstration_data = True
        elif message[0] == 'generate_agent_data':
            self.generate_demonstration_data = False
            print(self.print_prefix, 'collecting agent data')
        else:
            raise ValueError
        return new_params_packet

    def recv_messages(self, wait_for_new_params=False):
        """
        Subscribe to learner and replay_server messages.
        if topic == 'new_params' update model and return received_new_params.
           topic == 'generate_demonstration_data' set self.generate_demonstration_data True
           topic == 'generate_egent_data' set self.generate_demonstration_data False
        """
        new_params_packet = None
        if wait_for_new_params:
            while new_params_packet is None:
                message = self.sub_socket.recv()
                new_params_packet = self.read_message(message)
        else:
            try:
                message = self.sub_socket.recv(zmq.DONTWAIT)
                new_params_packet = self.read_message(message)
            except zmq.Again:
                # no packets are waiting
                pass

        if new_params_packet is not None:
            self.synchronize_params(new_params_packet)
            received_new_params = True
        else:
            received_new_params = False
        return received_new_params

    def run(self):
        """ uniform remote run wrapper for tester and worker actors """
        if self.is_tester:
            self.run_test()
        else:
            self.run_work()

    def run_work(self):
        self.initialize_training()
        self.load_datasets()
        while True:
            received_new_params = self.recv_messages()
            if received_new_params:
                # todo - Log stats with global_step == num_param_updates-1 (== the previous params_id).
                #        This is because the current stats are related to the previous policy.
                #        In addition, maybe workers shouldn't log stats every update ?
                # if self.num_param_updates > 0 and self.num_param_updates % self.hparams['log_interval'] == 0:
                global_step, log_dict = self.log_stats(global_step=self.num_param_updates - 1, log_directly=False)
                logs_packet = ('log', f'worker_{self.worker_id}', [('global_step', global_step)] + [(k, v) for k, v in log_dict.items()])
                logs_packet = pa.serialize(logs_packet).to_buffer()
                self.send_2_apex_socket.send(logs_packet)

            replay_data = self.collect_data()
            self.send_replay_data(replay_data)

    def run_test(self):
        # self.eps_greedy = 0
        self.initialize_training()
        datasets = self.load_datasets()
        while True:
            received = self.recv_messages(wait_for_new_params=True)
            assert received
            # todo consider not ignoring eval interval
            global_step, log_dict = self.evaluate(datasets, ignore_eval_interval=True, log_directly=False)
            logs_packet = ('log', 'tester', [('global_step', global_step)] + [(k, v) for k, v in log_dict.items()])
            logs_packet = pa.serialize(logs_packet).to_buffer()
            self.send_2_apex_socket.send(logs_packet)
            self.save_checkpoint()

    def collect_data(self):
        # todo
        #  - replace by execute episode
        #  - or more generally dqn experiment loop.
        """Fill local buffer until some stopping criterion is satisfied"""
        local_buffer = []
        trainset = self.trainset
        while len(local_buffer) < self.hparams.get('local_buffer_size'):
            # sample graph randomly
            graph_idx = self.graph_indices[(self.i_episode + 1) % len(self.graph_indices)]
            G, baseline = trainset['instances'][graph_idx]

            # execute episodes, collect experience and append to local_buffer
            trajectory = self.execute_episode(G, baseline, trainset['lp_iterations_limit'],
                                              dataset_name=trainset['dataset_name'],
                                              demonstration_episode=self.generate_demonstration_data)

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
        for transition, initial_priority, is_demonstration in replay_data:
            replay_data_packet.append((transition.to_numpy_tuple(), initial_priority, is_demonstration))
        replay_data_packet = pa.serialize(replay_data_packet).to_buffer()
        return replay_data_packet
