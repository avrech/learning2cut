import ray
from distributed.replay_server import PrioritizedReplayServer
from distributed.cut_dqn_worker import CutDQNWorker
from distributed.cut_dqn_learner import CutDQNLearner


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
        """
        Instantiate all components as Ray detached Actors.
        Detached actors have global unique names, they run independently of the current script
        python driver, and thus they can be shut down and restarted offline,
        possibly with updated code.
        Use case: when debugging/upgrading the learner/tester code, we don't need to shut down
        the replay server.
        For reference see: https://docs.ray.io/en/master/advanced.html#dynamic-remote-parameters
        the "Detached Actors" section.
        """
        # wrap base classes with ray.remote to make them remote "Actor"s
        ray_worker = ray.remote(CutDQNWorker)
        ray_learner = ray.remote(num_gpus=int(self.use_gpu), num_cpus=2)(CutDQNLearner)
        ray_replay_buffer = ray.remote(PrioritizedReplayServer)

        # spawn all components as detached processes
        # todo - verify actor.options(name='...').remote(...) if options works and if it is possible update code.
        self.workers = [ray_worker.options(name=f'worker_{n}').remote(n, hparams=self.cfg) for n in range(1, self.num_workers + 1)]
        self.test_worker = ray_worker.options(name='tester').remote('Test', hparams=self.cfg, is_tester=True)
        # instantiate learner and run its io process in a background thread
        self.learner = ray_learner.options(name='learner').remote(hparams=self.cfg, use_gpu=self.use_gpu, run_io=True)
        self.replay_server = ray_replay_buffer.options(name=f'replay_server').remote(config=self.cfg)

    def train(self):
        print("Running main training loop...")

        ready_ids, remaining_ids = ray.wait(
            [worker.run.remote() for worker in self.workers] +
            [self.learner.run_optimize_model.remote()] +
            [self.replay_server.run.remote()] +
            [self.test_worker.test_run.remote()]
        )
        # todo - find a good way to block the main program here, so ray will continue tracking all actors, restart etc.
        ray.get(ready_ids, timeout=self.cfg.get('time_limit', 3600*24))
        print('finished')
