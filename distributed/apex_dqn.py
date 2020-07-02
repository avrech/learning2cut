import ray
from distributed.replay_server import PrioritizedReplayServer
from distributed.cut_dqn_worker import CutDQNWorker
from distributed.cut_dqn_learner import CutDQNLearner


class ApeXDQN:
    def __init__(self, cfg, use_gpu=True):
        self.cfg = cfg
        self.num_workers = self.cfg["num_workers"]
        self.use_gpu = use_gpu

        # actors
        # self.workers = None
        # self.tester = None
        # self.learner = None
        # self.replay_server = None

        # container of all ray actors
        self.actors = {f'worker_{n}': None for n in range(1, self.num_workers + 1)}
        self.actors['tester'] = None
        self.actors['learner'] = None
        self.actors['replay_server'] = None

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
        ray_replay_server = ray.remote(PrioritizedReplayServer)

        # spawn all actors as detached actors with globally unique names.
        # those detached actors can be accessed from any driver connecting to the current ray server,
        # using the global unique names.
        for n in range(1, self.num_workers + 1):
            self.actors[f'worker_{n}'] = ray_worker.options(name=f'worker_{n}').remote(n, hparams=self.cfg)
        self.actors['tester'] = ray_worker.options(name='tester').remote('Test', hparams=self.cfg, is_tester=True)
        # instantiate learner and run its io process in a background thread
        self.actors['learner'] = ray_learner.options(name='learner').remote(hparams=self.cfg, use_gpu=self.use_gpu, run_io=True)
        self.actors['replay_server'] = ray_replay_server.options(name='replay_server').remote(config=self.cfg)

    def train(self):
        print("Running main training loop...")
        ready_ids, remaining_ids = ray.wait([actor.run.remote() for actor in self.actors.values()])
        # ready_ids, remaining_ids = ray.wait(
        #     [worker.run.remote() for worker in self.workers] +
        #     [self.learner.run.remote()] +
        #     [self.replay_server.run.remote()] +
        #     [self.tester.run.remote()]
        # )
        # todo - find a good way to block the main program here, so ray will continue tracking all actors, restart etc.
        ray.get(ready_ids + remaining_ids, timeout=self.cfg.get('time_limit', 3600*48))
        print('finished')

    def restart(self, actors=[], force_restart=False):
        """ Re-launch learner as detached actor """
        # todo - look for running learner, kill, instantiate a new one (with potentially updated code), and restart.
        actors = list(self.actors.keys()) if len(actors) == 0 else actors
        # get running actors if any.
        running_actors = {}
        for actor_name in actors:
            try:
                actor = ray.get_actor(actor_name)
                running_actors[actor_name] = actor
            except ValueError as e:
                # if actor_name doesn't exist, ray will raise a ValueError exception saying this
                print(e)
                running_actors[actor_name] = None

        ray_worker = ray.remote(CutDQNWorker)
        ray_learner = ray.remote(num_gpus=int(self.use_gpu), num_cpus=2)(CutDQNLearner)
        ray_replay_server = ray.remote(PrioritizedReplayServer)

        # restart all actors
        for actor_name in actors:
            running_actor = running_actors[actor_name]
            if running_actor is not None:
                if force_restart:
                    print(f'killing {actor_name}...')
                    ray.kill(running_actor)
                else:
                    print(f'request ignored, {actor_name} is already running.'
                          f'use --force-restart to kill the existing {actor_name} and restart a new one.')
                    continue

            print(f'restarting {actor_name}...')
            if actor_name == 'learner':
                learner = ray_learner.options(name='learner').remote(hparams=self.cfg, use_gpu=self.use_gpu, run_io=True)
                learner.run.remote()
            elif actor_name == 'tester':
                tester = ray_worker.options(name='tester').remote('Test', hparams=self.cfg, is_tester=True)
                tester.run.remote()
            elif actor_name == 'replay_server':
                replay_server = ray_replay_server.options(name='replay_server').remote(config=self.cfg)
                replay_server.run.remote()
            else:
                prefix, worker_id = actor_name.split('_')
                worker_id = int(worker_id)
                assert prefix == 'worker' and worker_id in range(1, self.num_workers + 1)
                worker = ray_worker.options(name=actor_name).remote(worker_id, hparams=self.cfg)
                worker.run.remote()

    def debug_actor(self, actor_name):
        # kill the existing one if any
        try:
            actor = ray.get_actor(actor_name)
            # if actor exists, kill it
            print(f'killing the existing {actor_name}...')
            ray.kill(actor_name)

        except ValueError as e:
            # if actor_name doesn't exist, ray will raise a ValueError exception saying this
            print(e)

        print(f'instantiating {actor_name} locally for debug...')
        if actor_name == 'learner':
            learner = CutDQNLearner(hparams=self.cfg, use_gpu=self.use_gpu, run_io=True)
            learner.run()
        elif actor_name == 'tester':
            tester = CutDQNWorker('Test', hparams=self.cfg, is_tester=True)
            tester.run()
        elif actor_name == 'replay_server':
            replay_server = PrioritizedReplayServer(config=self.cfg)
            replay_server.run()
        else:
            prefix, worker_id = actor_name.split('_')
            worker_id = int(worker_id)
            assert prefix == 'worker' and worker_id in range(1, self.num_workers + 1)
            worker = CutDQNWorker(worker_id, hparams=self.cfg)
            worker.run()
