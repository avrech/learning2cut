import ray
from distributed.per_server import PrioritizedReplayServer
from distributed.worker import GDQNWorker
from distributed.dqn_learner import GDQNLearner

ray.init()


@ray.remote
def run_learner_io(learner):
    learner.run_io.remote()


@ray.remote
def run_learner_optimize_model(learner):
    learner.run_optimize_model()


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
        ray_worker = ray.remote(GDQNWorker)
        ray_learner = ray.remote(GDQNLearner, num_gpus=int(self.use_gpu), num_cpus=2)
        ray_replay_buffer = ray.remote(PrioritizedReplayServer)

        # Spawn all components
        self.workers = [ray_worker.remote(n, hparams=self.cfg) for n in range(1, self.num_workers + 1)]
        self.test_worker = ray_worker.remote('Test', hparams=self.cfg, is_tester=True)
        self.learner = ray_learner.remote(hparams=self.cfg, use_gpu=self.use_gpu, use_ray_gpu_id=self.use_gpu)
        self.replay_server = ray_replay_buffer.remote(config=self.cfg)

    def train(self):
        print("Running main training loop...")

        ray.wait(
            [worker.run.remote() for worker in self.workers] +
            # run the learner io and optimize_model methods in two parallel sub-processes
            [run_learner_io.remote(self.learner), run_learner_optimize_model.remote(self.learner)] +
            [self.replay_server.run.remote()] +
            [self.test_worker.test_run.remote()]
        )
