import random
import torch
from collections import deque
import numpy as np
from torch_geometric.data.batch import Batch
from utils.segtree import MinSegmentTree, SegmentTree, SumSegmentTree
from utils.data import Transition


class ReplayBuffer(object):
    def __init__(self, capacity):
        """
        Create Replay buffer of Transition objects
        Parameters
        ----------
        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._capacity = capacity
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, transition: Transition):
        """
        append data to storage list, overriding the oldest elements (round robin)
        :param transition: Transition object
        :param kwargs:
        :return:
        """
        if len(self._storage) < self._capacity:
            # extend the memory for storing the new data
            self._storage.append(None)
        self._storage[self._next_idx] = transition
        # todo support heap.pop to override the min prioritized data, must be synchronized with PER priority updates
        # increment the next index round robin
        self._next_idx = (self._next_idx + 1) % self._capacity

    def sample(self, batch_size):
        assert batch_size <= len(self._storage)
        # sample batch_size unique transitions
        transitions = random.sample(self._storage, batch_size)
        return transitions

    def _encode_sample(self, sample):
        """
        Returns a tuple of standard np.array objects,
        """
        encoded_sample = []
        for transition in sample:
            encoded_sample.append(transition.to_numpy_tuple())
        return encoded_sample

    def _decode_sample(self, encoded_sample):
        """
        Returns the original torch_geometric Batch of Transitions.
        Useful for decoding samples on the learner side.
        """
        decoded_sample = []
        for transition_numpy_tuple in encoded_sample:
            decoded_sample.append(Transition.from_numpy_tuple(transition_numpy_tuple))
        return decoded_sample

    def __len__(self):
        return len(self._storage)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha):
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
        super(PrioritizedReplayBuffer, self).__init__(capacity)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, transition, initial_priority):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(transition)
        # update here the initial priority
        # because it is not safe to build on later update in another place
        # safe code:
        self.update_priorities([idx], [initial_priority]) # todo verify
        # old and unsafe:
        # self._it_sum[idx] = self._max_priority ** self._alpha
        # self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        """ todo ask Chris what is going on here """
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
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
        """
        assert beta > 0
        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)

        weights = torch.tensor(weights, dtype=torch.float32)
        # encoded_sample = self._encode_sample(idxes, weights) - old code
        transitions = [self._storage[idx] for idx in idxes]  # todo isn't there any efficient sampling way not list comprehension?
        # return a tuple of batch, importance sampling correction weights and idxes to update later the priorities
        return transitions, weights, idxes

    def _encode_sample(self, sample):
        """
        For distributed setting, we need to serialize sample into numpy arrays.
        Let
            batch, weights, idxes = sample
        The batch is encoded by the base ReplayBuffer class which handles this data type,
        weights are straightforwardly converted into and from numpy arrays,
        and idxes remain np.array since they are used only in the replay buffer, and need not be changed.
        """
        transitions, weights, idxes = sample
        # idxes are already stored as np.array, so need only to encode weights and batch
        # transitions are encoded using the base class
        encoded_transitions = super()._encode_sample(transitions)
        encoded_weights = weights.numpy()
        return encoded_transitions, encoded_weights, idxes

    def _decode_sample(self, encoded_sample):
        encoded_transitions, encoded_weights, idxes = encoded_sample
        transitions = super()._decode_sample(encoded_transitions)
        weights = torch.from_numpy(encoded_weights)
        return transitions, weights, idxes

    def update_priorities(self, idxes, priorities):
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
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
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