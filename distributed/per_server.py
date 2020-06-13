import ray
import pyarrow as pa
import zmq
from utils.buffer import PrioritizedReplayBuffer


class PrioritizedReplayBufferServer(PrioritizedReplayBuffer):
    def __init__(self, config):
        super(PrioritizedReplayBufferServer, self).__init__(config)
        self.config = config

        # # unpack buffer configs
        # self.max_num_updates = self.cfg["max_num_updates"]
        # self.priority_alpha = self.cfg["priority_alpha"]
        # self.priority_beta = self.cfg["priority_beta_start"]
        # self.priority_beta_end = self.cfg["priority_beta_end"]
        # self.priority_beta_increment = (
        #     self.priority_beta_end - self.priority_beta
        # ) / self.max_num_updates

        # self.batch_size = self.cfg["batch_size"]

        # self.buffer = PrioritizedReplayBuffer(
        #     self.cfg["buffer_max_size"], self.priority_alpha
        # )

        # unpack communication configs
        self.repreq_port = config["repreq_port"]
        self.pullpush_port = config["pullpush_port"]

        # initialize zmq sockets
        print("[Buffer]: initializing sockets..")
        self.initialize_sockets()

    def initialize_sockets(self):
        # for sending batch to learner and retrieving new priorities
        context = zmq.Context()
        self.rep_socket = context.socket(zmq.REQ)
        self.rep_socket.connect(f"tcp://127.0.0.1:{self.repreq_port}")

        # for receiving replay data from workers
        context = zmq.Context()
        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://127.0.0.1:{self.pullpush_port}")

    def get_batch_packet(self):
        transitions, weights, idxes = self.sample()
        # weights returned as torch.tensor by default.
        # transitions remain Transition.to_numpy_tuple() object,
        # idxes are list.
        # need only to convert weights to standard numpy array
        batch = (transitions, weights.numpy(), idxes)
        # batch = self.encode_sample(sample)  # todo data remained encoded from worker
        batch_packet = pa.serialize(batch).to_buffer()
        return batch_packet

    @staticmethod
    def unpack_priorities(priorities_packet):
        idxes, priorities = pa.deserialize(priorities_packet)
        return idxes, priorities

    def send_batch_recv_priors(self):
        batch_packet = self.get_batch_packet()
        # send batch and request priorities (blocking recv)
        self.rep_socket.send(batch_packet)

        # receive and update priorities
        new_priors_packet = self.rep_socket.recv()
        idxes, new_priorities = self.unpack_priorities(new_priors_packet)
        self.update_priorities(idxes, new_priorities)

    @staticmethod
    def unpack_replay_data(replay_data_packet):
        encoded_replay_data = pa.deserialize(replay_data_packet)
        return encoded_replay_data

    def recv_replay_data(self):
        new_replay_data_packet = False
        try:
            new_replay_data_packet = self.pull_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass

        if new_replay_data_packet:
            new_replay_data = self.unpack_replay_data(new_replay_data_packet)
            for replay_data, initial_priority in new_replay_data:
                self.add(replay_data, initial_priority)

    def run(self):
        while True:
            self.recv_replay_data()
            if len(self.buffer) > self.batch_size:
                self.send_batch_recv_priors()


@ray.remote
class RayPrioritizedReplayBufferServer(PrioritizedReplayBufferServer):
    """ Ray remote actor wrapper for PrioritizedReplayBufferServer """
    def __init__(self, config):
        super().__init__(config)
