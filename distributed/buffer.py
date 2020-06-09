import random
from collections import deque
from utils.data import Transition
import numpy as np
from torch_geometric.data.batch import Batch
from utils.segtree import MinSegmentTree, SegmentTree, SumSegmentTree
import torch

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._capacity = size
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
        # todo support heap.pop to override the min prioritized data
        # increment the next index round robin
        self._next_idx = (self._next_idx + 1) % self._capacity

    # def sample(self, batch_size):
    #     """Sample a batch of experiences.
    #     Parameters
    #     ----------
    #     batch_size: int
    #         How many transitions to sample.
    #     Returns
    #     -------
    #     batch: Batch
    #         Batch of Transition objects
    #     """
    #     idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]  # todo can potentially sample the same transition twice
    #     return self._encode_sample(idxes)

    def sample(self, batch_size):
        assert batch_size <= len(self._storage)
        # sample batch_size unique transitions
        transitions = random.sample(self._storage, batch_size)
        # create a batch object
        batch = Batch().from_data_list(transitions, follow_batch=['x_c', 'x_v', 'x_a', 'ns_x_c', 'ns_x_v', 'ns_x_a'])
        return batch

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
        # todo consider to update here the initial priority
        #  because it is not safe to build on later update in another place
        # safe code:
        self.update_priorities([idx], [initial_priority])
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
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
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
        weights = np.array(weights)
        # todo encode sample in a single transition object
        encoded_sample = self._encode_sample(idxes, weights)
        # return tuple(list(encoded_sample) + [weights, idxes])
        return encoded_sample

    def _encode_sample(self, idxes, weights):
        """
        ReplayBuffer returned object is Batch.
        This object can be used directly in DQN.sgd_step()

        For distributed setting, we need to serialize it into numpy arrays,
        and this will be done in the child classes.
        The batch object keys can be used to encode the batch as tuple of numpy arrays:
        # break batch into tuple of np.arrays
        encoded_sample = (batch[k].numpy() for k in batch.keys)
        in order rebuild a Batch object from such a tuple,
        we need a dummy Batch() built from 2 arbitrary transition objects, dummy_batch, and then
        for k, np_array in zip(dummy_batch.keys, encoded_sample):
            dummy_batch[k] = torch.from_numpy(np_array)
        the resulting dummy_batch is identical to the original batch,
        and specifically it keeps its transition samples order,
        so the related priorities and weights can be used properly.
        """
        # read transitions from storage
        transitions = self._storage[idxes]
        # create a batch object
        batch = Batch().from_data_list(transitions, follow_batch=['x_c', 'x_v', 'x_a', 'ns_x_c', 'ns_x_v', 'ns_x_a'])
        # todo store the importance sampling correction weights and corresponding idxes in batch
        batch['per_weight'] = torch.from_numpy(weights)
        batch['per_index'] = torch.from_numpy(idxes)
        return batch

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
