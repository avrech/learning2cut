import ray
import pyarrow as pa
import zmq
from utils.buffer import PrioritizedReplayBuffer


class PrioritizedReplayServer(PrioritizedReplayBuffer):
    def __init__(self, config):
        super(PrioritizedReplayServer, self).__init__(config)
        self.config = config
        self.pending_priority_requests_cnt = 0
        self.max_pending_requests = config.get('max_pending_requests', 10)

        # initialize zmq sockets
        print("[ReplayServer]: initializing sockets..")
        # for sending a batch to learner
        context = zmq.Context()
        self.replay_server_2_learner_port = config["replay_server_2_learner_port"]
        self.replay_server_2_learner_socket = context.socket(zmq.PUSH)
        self.replay_server_2_learner_socket.bind(f'tcp://127.0.0.1:{self.replay_server_2_learner_port}')
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

    def send_batch_recv_priorities(self):
        # send batches to learner up to max_pending_requests
        while self.pending_priority_requests_cnt < self.max_pending_requests:
            batch_packet = self.get_batch_packet()
            self.replay_server_2_learner_socket.send(batch_packet)
            self.pending_priority_requests_cnt += 1

        # receive and update priorities (non-blocking) until no more priorities received
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
        # try at most num_workers times, to balance between the number of workers and the network load.
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
        while True:
            self.recv_replay_data()
            if len(self.storage) > self.batch_size * 10:  # x10 because we don't want to make the learner overfitting at the beginning
                self.send_batch_recv_priorities()
