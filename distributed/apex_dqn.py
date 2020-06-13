import ray
from distributed.per_server import RayPrioritizedReplayBufferServer as ReplayServer
from distributed.worker import RayGDQNWorker as Worker
from distributed.dqn_learner import RayGDQNLearner as Learner


class ApeXDQN:
    def __init__(self, cfg, use_gpu=True, gpu_id=None):
        self.cfg = cfg
        self.num_workers = self.cfg["num_workers"]
        self.num_learners = self.cfg["num_learners"]
        self.worker_class = Worker
        self.learner_class = Learner
        self.replay_server_class = ReplayServer
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        # actors
        self.workers = None
        self.test_worker = None
        self.learner = None
        self.replay_server = None
        self.all_actors = None

    def spawn(self):
        # Spawn all components
        self.workers = [Worker.remote(n, hparams=self.cfg) for n in range(1, self.num_workers + 1)]
        self.test_worker = Worker.remote('Test', hparams=self.cfg, is_tester=True)
        self.learner = Learner.remote(hparams=self.cfg, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        self.replay_server = ReplayServer.remote(config=self.cfg)

    def train(self):
        print("Running main training loop...")
        ray.wait(
            [worker.run.remote() for worker in self.workers] +
            [self.learner.run.remote()] +
            [self.replay_server.run.remote()] +
            [self.test_worker.test_run.remote()]
        )
