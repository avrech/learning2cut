import pyarrow as pa
import zmq
from utils.buffer import PrioritizedReplayBuffer
import torch
import os


class PrioritizedReplayServer(PrioritizedReplayBuffer):
    def __init__(self, config):
        super(PrioritizedReplayServer, self).__init__(config)
        self.config = config
        self.pending_priority_requests_cnt = 0
        self.max_pending_requests = config.get('max_pending_requests', 10)
        self.minimum_size = config.get('replay_buffer_minimum_size', 128*100)

        # initialize zmq sockets
        print("[ReplayServer]: initializing sockets..")
        # for sending a batch to learner
        context = zmq.Context()
        self.replay_server_2_learner_port = config["replay_server_2_learner_port"]
        self.replay_server_2_learner_socket = context.socket(zmq.PUSH)
        self.replay_server_2_learner_socket.connect(f'tcp://127.0.0.1:{self.replay_server_2_learner_port}')
        # for receiving new priorities from learner
        context = zmq.Context()
        self.learner_2_replay_server_port = config["learner_2_replay_server_port"]
        self.learner_2_replay_server_socket = context.socket(zmq.PULL)
        self.learner_2_replay_server_socket.bind(f'tcp://127.0.0.1:{self.learner_2_replay_server_port}')
        # for receiving replay data from workers
        context = zmq.Context()
        self.workers_2_replay_server_port = config["workers_2_replay_server_port"]
        self.workers_2_replay_server_socket = context.socket(zmq.PULL)
        self.workers_2_replay_server_socket.bind(f'tcp://127.0.0.1:{self.workers_2_replay_server_port}')

        # failure tolerance
        self.logdir = config.get('logdir', 'results')
        self.checkpoint_filepath = os.path.join(self.logdir, 'replay_server_checkpoint.pt')
        # checkpoint every time params are published (like the other components do)
        self.checkpoint_interval = config.get("param_sync_interval", 50)

    def save_checkpoint(self):
        torch.save({
            'num_sgd_steps_done': self.num_sgd_steps_done
        }, self.checkpoint_filepath)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_filepath):
            print('Checkpoint file does not exist! starting from scratch.')
            return
        checkpoint = torch.load(self.checkpoint_filepath)
        self.num_sgd_steps_done = checkpoint['num_sgd_steps_done']
        print('Loaded checkpoint from: ', self.checkpoint_filepath)

    def get_batch_packet(self):
        transitions, weights, idxes, data_ids = self.sample()
        # weights returned as torch.tensor by default.
        # transitions remain Transition.to_numpy_tuple() object,
        # idxes are list.
        # need only to convert weights to standard numpy array
        batch = (transitions, weights, idxes, data_ids)
        batch_packet = pa.serialize(batch).to_buffer()
        return batch_packet

    @staticmethod
    def unpack_priorities(priorities_packet):
        unpacked_priorities = pa.deserialize(priorities_packet)
        return unpacked_priorities  # idxes, priorities, data_ids

    def send_batches(self):
        # wait for receiving minimum_size transitions from workers,
        # to avoid the learner from overfitting
        if len(self.storage) >= self.minimum_size:
            # send batches to learner up to max_pending_requests (learner's queue capacity)
            while self.pending_priority_requests_cnt < self.max_pending_requests:
                batch_packet = self.get_batch_packet()
                self.replay_server_2_learner_socket.send(batch_packet)
                self.pending_priority_requests_cnt += 1

    def recv_new_priorities(self):
        # receive all the waiting new_priorities packets (non-blocking)
        # unpack, and update memory priorities
        while True:
            try:
                new_priorities_packet = self.learner_2_replay_server_socket.recv(zmq.DONTWAIT)
            except zmq.Again:
                break  # no priority packets received. break and return.
            idxes, new_priorities, batch_ids = self.unpack_priorities(new_priorities_packet)
            self.update_priorities(idxes, new_priorities, batch_ids)
            self.pending_priority_requests_cnt -= 1  # decrease pending requests counter

    @staticmethod
    def unpack_replay_data(replay_data_packet):
        encoded_replay_data = pa.deserialize(replay_data_packet)
        return encoded_replay_data

    def recv_replay_data(self):
        # receive replay data from all workers.
        # try at most num_workers times to avoid an infinite loop,
        # which can happen in a case there are multiple workers,
        # so that while the server finishes receiving the n'th packet
        # new replay data packets arrive.
        # todo - balance between the number of workers and the network load playing with worker's local buffer size
        #  such that each worker sends larger packets in larger intervals,
        #  and tune this interval to match the replay server cycle time
        for _ in range(self.config['num_workers']):
            try:
                new_replay_data_packet = self.workers_2_replay_server_socket.recv(zmq.DONTWAIT)
            except zmq.Again:
                break

            new_replay_data_list = self.unpack_replay_data(new_replay_data_packet)
            # for transition_and_priority_tuple in new_replay_data:
            #     self.add(transition_and_priority_tuple)
            self.add_data_list(new_replay_data_list)

    def run(self):
        if self.config.get('resume_training', False):
            self.load_checkpoint()

        while True:
            self.recv_replay_data()
            self.send_batches()
            self.recv_new_priorities()
            if self.num_sgd_steps_done > 0 and self.num_sgd_steps_done % self.checkpoint_interval == 0:
                self.save_checkpoint()
