import ray
from distributed.replay_server import PrioritizedReplayServer
from distributed.cut_dqn_worker import CutDQNWorker
from distributed.cut_dqn_learner import CutDQNLearner, RayLearner
import time


class ApeXDQN:
    def __init__(self, cfg, use_gpu=True):
        self.cfg = cfg
        self.num_workers = self.cfg["num_workers"]
        self.num_learners = self.cfg["num_learners"]
        self.use_gpu = use_gpu

        # actors
        self.workers = None
        self.test_worker = None
        self.learner = None
        self.replay_server = None
        self.all_actors = None

    def spawn(self):
        # wrap all actor classes with ray.remote to make them running remotely
        ray_worker = ray.remote(CutDQNWorker)
        # ray_learner = ray.remote(num_gpus=int(self.use_gpu), num_cpus=2)(CutDQNLearner)
        ray_replay_buffer = ray.remote(PrioritizedReplayServer)

        # Spawn all components
        self.workers = [ray_worker.remote(n, hparams=self.cfg) for n in range(1, self.num_workers + 1)]
        self.test_worker = ray_worker.remote('Test', hparams=self.cfg, is_tester=True)
        # instantiate learner and run its io method in a background process
        # self.learner = ray_learner.remote(hparams=self.cfg, use_gpu=self.use_gpu, run_io=True)
        self.learner = RayLearner.remote(hparams=self.cfg, use_gpu=self.use_gpu, run_io=True)
        self.replay_server = ray_replay_buffer.remote(config=self.cfg)

    def train(self):
        print("Running main training loop...")

        ready_ids, remaining_ids = ray.wait(
            [worker.run.remote() for worker in self.workers] +
            [self.learner.run_optimize_model.remote()] +
            [self.replay_server.run.remote()] +
            [self.test_worker.test_run.remote()]
        )
        time.sleep(self.cfg.get('time_limit', 3600*24))
        print('finished')
