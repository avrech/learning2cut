import pyarrow as pa
import zmq
from utils.buffer import PrioritizedReplayBuffer
import torch
import os
from tqdm import tqdm


class PrioritizedReplayServer(PrioritizedReplayBuffer):
    def __init__(self, config):
        super(PrioritizedReplayServer, self).__init__(config)
        self.config = config
        self.pending_priority_requests_cnt = 0
        self.max_pending_requests = config.get('max_pending_requests', 10)
        self.minimum_size = config.get('replay_buffer_minimum_size', 128*100)
        self.collecting_demonstrations = len(self.storage) < self.n_demonstrations
        self.pbar = tqdm(total=self.capacity, desc='[Replay Server] Filling {} data'.format('demonstration' if self.collecting_demonstrations else 'agent'))
        self.filling = True
        self.print_prefix = '[ReplayServer] '
        # initialize zmq sockets
        print(self.print_prefix, "initializing sockets...")
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
        # for publishing data requests to workers
        context = zmq.Context()
        self.data_request_pubsub_port = config["replay_server_2_workers_pubsub_port"]
        self.data_request_pub_socket = context.socket(zmq.PUB)
        self.data_request_pub_socket.connect(f"tcp://127.0.0.1:{self.data_request_pubsub_port}")

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
            print(self.print_prefix, 'Checkpoint file does not exist! starting from scratch.')
            return
        checkpoint = torch.load(self.checkpoint_filepath)
        self.num_sgd_steps_done = checkpoint['num_sgd_steps_done']
        print(self.print_prefix, 'Loaded checkpoint from: ', self.checkpoint_filepath)

    def get_batch_packet(self):
        batch = self.sample()
        batch_packet = pa.serialize(batch).to_buffer()
        return batch_packet

    # todo - this is essentially the same like unpcak_replay_data. unify those two to one general unpack()
    @staticmethod
    def unpack_priorities(priorities_packet):
        unpacked_priorities = pa.deserialize(priorities_packet)
        return unpacked_priorities

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
                new_packet = self.learner_2_replay_server_socket.recv(zmq.DONTWAIT)
            except zmq.Again:
                break  # no packet received. break and return.
            # unpack the received packet, and classify it
            message = self.unpack_priorities(new_packet)
            if message == "restart":
                # learner has been restarted.
                # reset the pending_priority_requests_cnt to avoid deadlocks
                print(self.print_prefix, 'received "restart" message from learner. resetting pending requests counter...')
                self.pending_priority_requests_cnt = 0

            else:
                # this is a new_priorities packet
                idxes, new_priorities, batch_ids = message
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
            if self.filling:
                if len(self.storage) + len(new_replay_data_list) < self.capacity:
                    self.pbar.update(len(new_replay_data_list))
                else:
                    # now filled the capacity.
                    # increment the progress bar by the amount left and close
                    self.pbar.update(self.capacity - len(self.storage))
                    self.pbar.close()
                    self.filling = False

            self.add_data_list(new_replay_data_list)
            if self.collecting_demonstrations and self.next_idx >= self.n_demonstrations:
                # change pbar description
                self.pbar.set_description('[Replay Server] Filling agent data')

    def run(self):
        if self.config.get('resume_training', False):
            self.load_checkpoint()

        # assert behaviour when recovering from failures / restarting replay server
        if len(self.storage) < self.n_demonstrations:
            assert self.collecting_demonstrations
        else:
            assert not self.collecting_demonstrations

        if self.collecting_demonstrations:
            print(self.print_prefix, 'Publishing demonstration data request')
            message = pa.serialize(('generate_demonstration_data', )).to_buffer()
            self.data_request_pub_socket.send(message)

        while True:
            self.recv_replay_data()
            self.send_batches()
            self.recv_new_priorities()
            if self.num_sgd_steps_done > 0 and self.num_sgd_steps_done % self.checkpoint_interval == 0:
                self.save_checkpoint()

            if self.collecting_demonstrations and self.next_idx >= self.n_demonstrations:
                # request workers to generate agent data from now on
                self.collecting_demonstrations = False
                print(self.print_prefix, 'Publishing agent data request')
                message = pa.serialize(('generate_agent_data',)).to_buffer()
                self.data_request_pub_socket.send(message)
