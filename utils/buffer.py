import random
import numpy as np
from utils.segtree import MinSegmentTree, SegmentTree, SumSegmentTree
from utils.data import Transition
import math


class ReplayBuffer(object):
    def __init__(self, capacity, n_demonstrations=0):
        """
        Create Replay buffer of Transition objects.
        The first n_demonstrations samples (assumed to come from expert behaviour) remain permanently in the buffer.

        :param capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        :param n_demonstrations: int
            number of samples to save for demonstration data
        """
        assert n_demonstrations <= capacity
        self.storage = []
        self.capacity = capacity
        self.n_demonstrations = n_demonstrations
        self.next_idx = 0

    def __len__(self):
        return len(self.storage)

    def add(self, data):
        """
        append data to storage list, overriding the oldest elements (round robin)
        :param data: Transition object
        :return:
        """
        if self.next_idx == len(self.storage) < self.capacity:
            # extend the memory for storing the new data
            self.storage.append(None)
        self.storage[self.next_idx] = data

        # increment the next index round robin, overriding the oldest non-demonstration data
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.capacity:
            # if n_demonstrations < capacity, agent data is stored in idxes = [n_demonstrations:capacity]
            # if n_demonstrations = capacity, only demonstration data is generated and overrides the oldest samples.
            # (kind of supervised learning in fact)
            self.next_idx = self.n_demonstrations % self.capacity

    def add_data_list(self, buffer):
        # push all transitions from local_buffer into memory
        for data in buffer:
            self.add(data)

    def sample(self, batch_size):
        assert batch_size <= len(self.storage)
        # sample batch_size unique transitions
        idxes = random.sample(range(len(self.storage)), batch_size)
        transitions = [self.storage[idx] for idx in idxes]
        return transitions

    def __len__(self):
        return len(self.storage)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, config={}):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(config.get('replay_buffer_capacity', 100000),
                                                      n_demonstrations=config.get('replay_buffer_n_demonstrations', 0))
        self.config = config
        self.batch_size = config.get('batch_size', 128)
        self.demonstration_priority_bonus = config.get('replay_buffer_demonstration_priority_bonus', 0.00001)
        self._alpha = config.get('priority_alpha', 0.4)
        assert self._alpha >= 0

        it_capacity = 1
        while it_capacity < self.capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._next_unique_id = 0
        self._data_unique_ids = -np.ones((self.capacity,))
        self._data_sizes = np.zeros((self.capacity,), dtype=float)
        self.storage_size = 0

        # unpack buffer configs
        # self.max_num_updates = self.cfg["max_num_updates"]
        # self.priority_beta = self.cfg["priority_beta_start"]
        # self.priority_beta_end = self.cfg["priority_beta_end"]
        # self.priority_beta_increment = (self.priority_beta_end - self.priority_beta) / self.max_num_updates

        self.priority_beta_start = config.get('priority_beta_start', 0.4)
        self.priority_beta_end = config.get('priority_beta_end', 1.0)
        self.priority_beta_decay = config.get('priority_beta_decay', 10000)
        self.num_sgd_steps_done = 0  # saved and loaded from checkpoint
        # beta = self.priority_beta_end - (self.priority_beta_end - self.priority_beta_start) * \
        #        math.exp(-1. * self.num_sgd_steps_done / self.priority_beta_decay)
        self.print_prefix = '[ReplayBuffer] '

    def add(self, data: tuple([Transition, float, bool, float])):
        """
        Push Transition into storage and update its initial priority.
        :param data: Tuple containing Transition and initial_priority float
        :return:
        """
        transition, initial_priority, is_demonstration, size_gbyte = data
        if self.next_idx < self.n_demonstrations and not is_demonstration:
            # ignore non-demonstration data if we are in demonstrations collection phase.
            return 0

        # check memory usage
        if len(self.storage) < self.capacity:
            total_mem = self.storage_size + size_gbyte
            if total_mem > self.config['replay_buffer_max_mem'] - 0.5:
                # we are close to the mem limit (<0.5GB free space).
                # limit capacity to storage length + 1.
                # this data will be appended, but the next data will override the oldest.
                self.capacity = len(self.storage) + 1
                # todo update pbar size in replay server
        else:
            # we are going to override the oldest data in buffer.
            # remove elements from the end of storage until there will be enough space
            while self.storage_size + size_gbyte - self._data_sizes[self.next_idx] >= self.config['replay_buffer_max_mem']:
                # delete the last element in storage and update self.next_idx if needed
                self.storage_size -= self._data_sizes[len(self.storage)-1]
                self._data_sizes[len(self.storage)-1] = 0
                del self.storage[-1]
                if self.next_idx > len(self.storage):
                    self.next_idx = len(self.storage)

        idx = self.next_idx

        super().add(transition)
        # assign unique id to data
        # such that new arriving data won't be updated with outdated new_priorities arriving from the learner.
        # (critical for distributed setting)
        self._data_unique_ids[idx] = self._next_unique_id
        # update the initial priority
        self.update_priorities(np.array([idx]), np.array([initial_priority]), np.array([self._next_unique_id]))  # todo verify
        # increment _next_unique_id
        self._next_unique_id = self._next_unique_id + 1  # todo handle int overflow
        return 1

    def add_data_list(self, buffer):
        """ Adds a list of  (Transition, priority, is_demonstration) to storage,
        and returns the number of transitions actually added.
        In case demonstration data is required, agent data will be ignored. """
        # push all (transitions, initial_priority) tuples from buffer into memory
        n = 0
        for data in buffer:
            n += self.add(data)
        return n

    def _sample_proportional(self, batch_size):
        """ todo ask Chris what is going on here """
        res = []
        p_total = self._it_sum.sum(0, len(self.storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size=None, beta=None):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float (Optional)
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
            By default scheduled to start at 0.4, and slowly converge to 1 in exponential rate.
        Returns
        -------
        batch: torch_geometric Batch of Transition
            The batch object used in DQN.sgd_step()
        weights: torch.tensor
            Array of shape (batch_size,) and dtype torch.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        time_stamps: np.array
            Array of shape (batch_size,) and dtype int
            the write_time_stamps of the sampled transitions
        """
        if batch_size is None:
            batch_size = self.batch_size
        if beta is None:
            beta = self.priority_beta_end - (self.priority_beta_end - self.priority_beta_start) * \
                   math.exp(-1. * self.num_sgd_steps_done / self.priority_beta_decay)

        assert beta > 0
        assert len(self.storage) >= batch_size

        idxes = np.array(self._sample_proportional(batch_size))

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.storage)) ** (-beta)
            weights.append(weight / max_weight)

        weights = np.array(weights, dtype=np.float32)
        # encoded_sample = self._encode_sample(idxes, weights) - old code
        transitions = [self.storage[idx] for idx in idxes]  # todo isn't there any efficient sampling way not list comprehension?
        self.num_sgd_steps_done += 1  # increment global counter to decay beta across training

        data_ids = self._data_unique_ids[idxes]

        # return a tuple of transitions, importance sampling correction weights, idxes to update later the priorities and data unique ids
        return transitions, weights, idxes, data_ids

    def update_priorities(self, idxes, priorities, data_ids):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert len(idxes) == len(data_ids)
        assert all(priorities > 0), priorities

        for idx, priority, data_id in zip(idxes, priorities, data_ids):
            # filter invalid packets
            if not 0 <= idx <= len(self.storage):
                print(self.print_prefix, 'received invalid index')
                # todo - possibly the whole packet is corrupted/outdated, should we stop here to save time?
                continue
            if data_id != self._data_unique_ids[idx]:
                # data_id will be equal to self._data_unique_ids[idx] for newly added data,
                # and for returned priorities whose corresponding stored data hasn't been overridden between
                # sending the batch to the learner and receiving the updated priorities.
                # in a case those two unique ids are not equal,
                # the position idx in the storage was overridden by a new replay data,
                # so the current priority is not relevant any more and can be discarded
                continue

            # update priority:
            # for demonstration data (permanently stored in 0 <= idxes < self.n_demonstrations)
            # we add a positive bonus
            if idx < self.n_demonstrations:
                priority += self.demonstration_priority_bonus

            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)


# from pytorch dqn tutorial
class ReplayMemory(object):
    """
    Stores transitions of type utils.data.Transition in a single huge list
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            # extend the memory for storing the new transition
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
