""" run_apex_dqn.py copied and modified from https://github.com/cyoon1729/distributedRL """
import ray
import torch

from apex_dqn.dqn_learner import DQNLearner
from apex_dqn.dqn_worker import DQNWorker
from apex_dqn.models import ConvDQN, ConvDuelingDQN
from architectures.apex import ApeX
from common.utils.buffer_helper import PrioritizedReplayBufferHelper
from common.utils.utils import read_config

ray.init()


if __name__ == "__main__":

    cfg, comm_cfg = read_config("config.yml")

    dqn = ConvDuelingDQN(cfg["obs_dim"], cfg["action_dim"])
    target_dqn = ConvDuelingDQN(cfg["obs_dim"], cfg["action_dim"])
    brain = (dqn, target_dqn)

    ApeXDQN = ApeX(DQNWorker, DQNLearner, brain, cfg, comm_cfg)
    ApeXDQN.spawn()
    ApeXDQN.train()