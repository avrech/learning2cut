import ray
from distributed.replay_server import PrioritizedReplayServer
from distributed.cut_dqn_worker import CutDQNWorker
from distributed.cut_dqn_learner import CutDQNLearner
import os
import pickle
import wandb
import zmq


class ApeXDQN:
    def __init__(self, cfg, use_gpu=True):
        self.cfg = cfg
        self.num_workers = self.cfg["num_workers"]
        self.use_gpu = use_gpu
        self.learner_gpu = use_gpu and self.cfg.get('learner_gpu', True)
        self.worker_gpu = use_gpu and self.cfg.get('worker_gpu', True)
        self.tester_gpu = use_gpu and self.cfg.get('tester_gpu', True)

        # container of all ray actors
        self.actors = {f'worker_{n}': None for n in range(1, self.num_workers + 1)}
        self.actors['tester'] = None
        self.actors['learner'] = None
        self.actors['replay_server'] = None

        # communication settings
        # we reuse the existing ports iff some actors are still running.
        # otherwise, we randomize new ports.
        # todo when reusing, there is a chance that a zombie process occupies some port.
        #  in that case zmq will crash on binding to that port.
        #  the user should then clean those zombie processes (run again with --clean_zombie)
        if self.cfg['resume'] and len(self.get_running_actors()) > 0:

            # reuse ports if any actor is still running
            with open(os.path.join(self.cfg['run_dir'], 'com_cfg.pkl'), 'rb') as f:
                com_cfg = pickle.load(f)
            print('loaded ports from ', os.path.join(self.cfg['run_dir'], 'com_cfg.pkl'))
        else:
            com_cfg = self.find_free_ports()
            # pickle ports to experiment dir
            with open(os.path.join(self.cfg['run_dir'], 'com_cfg.pkl'), 'wb') as f:
                pickle.dump(com_cfg, f)
            print('saved ports to ', os.path.join(self.cfg['run_dir'], 'com_cfg.pkl'))

        self.cfg['com'] = com_cfg

    def find_free_ports(self):
        """ finds free ports for all actors and returns a dictionary of all ports """
        ports = {}
        # replay server
        context = zmq.Context()
        learner_2_replay_server_socket = context.socket(zmq.PULL)
        workers_2_replay_server_socket = context.socket(zmq.PULL)
        data_request_pub_socket = context.socket(zmq.PUB)
        replay_server_2_learner_socket = context.socket(zmq.PULL)
        params_pub_socket = context.socket(zmq.PUB)
        ports["learner_2_replay_server_port"] = learner_2_replay_server_socket.bind_to_random_port('tcp://127.0.0.1', min_port=self.cfg['min_port'], max_port=self.cfg['min_port'] + self.cfg['port_range'])
        ports["workers_2_replay_server_port"] = workers_2_replay_server_socket.bind_to_random_port('tcp://127.0.0.1', min_port=self.cfg['min_port'], max_port=self.cfg['min_port'] + self.cfg['port_range'])
        ports["replay_server_2_workers_pubsub_port"] = data_request_pub_socket.bind_to_random_port('tcp://127.0.0.1', min_port=self.cfg['min_port'], max_port=self.cfg['min_port'] + self.cfg['port_range'])
        ports["replay_server_2_learner_port"] = replay_server_2_learner_socket.bind_to_random_port('tcp://127.0.0.1', min_port=self.cfg['min_port'], max_port=self.cfg['min_port'] + self.cfg['port_range'])
        ports["learner_2_workers_pubsub_port"] = params_pub_socket.bind_to_random_port('tcp://127.0.0.1', min_port=self.cfg['min_port'], max_port=self.cfg['min_port'] + self.cfg['port_range'])
        learner_2_replay_server_socket.close()
        workers_2_replay_server_socket.close()
        data_request_pub_socket.close()
        replay_server_2_learner_socket.close()
        params_pub_socket.close()

        return ports

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
        ray_worker = ray.remote(num_gpus=int(self.worker_gpu), num_cpus=1)(CutDQNWorker)
        ray_tester = ray.remote(num_gpus=int(self.tester_gpu), num_cpus=1)(CutDQNWorker)
        ray_learner = ray.remote(num_gpus=int(self.learner_gpu), num_cpus=2)(CutDQNLearner)
        ray_replay_server = ray.remote(PrioritizedReplayServer)

        # spawn all actors as detached actors with globally unique names.
        # those detached actors can be accessed from any driver connecting to the current ray server,
        # using the global unique names.
        for n in range(1, self.num_workers + 1):
            self.actors[f'worker_{n}'] = ray_worker.options(name=f'worker_{n}').remote(n, hparams=self.cfg, use_gpu=self.worker_gpu)
        self.actors['tester'] = ray_tester.options(name='tester').remote('Test', hparams=self.cfg, use_gpu=self.tester_gpu, is_tester=True)
        # instantiate learner and run its io process in a background thread
        self.actors['learner'] = ray_learner.options(name='learner').remote(hparams=self.cfg, use_gpu=self.learner_gpu, run_io=True)
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

    def get_running_actors(self, actors=None):
        actors = actors if actors is not None else list(self.actors.keys())
        running_actors = {}
        for actor_name in actors:
            try:
                actor = ray.get_actor(actor_name)
                running_actors[actor_name] = actor
            except ValueError as e:
                # if actor_name doesn't exist, ray will raise a ValueError exception saying this
                print(e)
                running_actors[actor_name] = None
        return running_actors

    def restart(self, actors=[], force_restart=False):
        """ Re-launch learner as detached actor """
        # todo - look for running learner, kill, instantiate a new one (with potentially updated code), and restart.
        actors = list(self.actors.keys()) if len(actors) == 0 else actors
        running_actors = self.get_running_actors(actors)

        ray_worker = ray.remote(num_gpus=int(self.worker_gpu), num_cpus=1)(CutDQNWorker)
        ray_tester = ray.remote(num_gpus=int(self.tester_gpu), num_cpus=1)(CutDQNWorker)
        ray_learner = ray.remote(num_gpus=int(self.learner_gpu), num_cpus=2)(CutDQNLearner)
        ray_replay_server = ray.remote(PrioritizedReplayServer)
        handles = []
        # restart all actors
        for actor_name in actors:
            running_actor = running_actors[actor_name]
            if running_actor is not None:
                if force_restart:
                    print(f'killing {actor_name}...')
                    ray.kill(running_actor)
                else:
                    print(f'request ignored, {actor_name} is already running. '
                          f'use --force-restart to kill the existing {actor_name} and restart a new one.')
                    continue

            print(f'restarting {actor_name}...')
            if actor_name == 'learner':
                learner = ray_learner.options(name='learner').remote(hparams=self.cfg, use_gpu=self.learner_gpu, run_io=True)
                handles.append(learner.run.remote())
            elif actor_name == 'tester':
                tester = ray_tester.options(name='tester').remote('Test', hparams=self.cfg, is_tester=True, use_gpu=self.tester_gpu)
                handles.append(tester.run.remote())
            elif actor_name == 'replay_server':
                replay_server = ray_replay_server.options(name='replay_server').remote(config=self.cfg)
                handles.append(replay_server.run.remote())
            else:
                prefix, worker_id = actor_name.split('_')
                worker_id = int(worker_id)
                assert prefix == 'worker' and worker_id in range(1, self.num_workers + 1)
                worker = ray_worker.options(name=actor_name).remote(worker_id, hparams=self.cfg, use_gpu=self.worker_gpu)
                handles.append(worker.run.remote())
        if len(handles) > 0:
            ready_ids, remaining_ids = ray.wait(handles)
            # todo - find a good way to block the main program here, so ray will continue tracking all actors, restart etc.
            ray.get(ready_ids + remaining_ids, timeout=self.cfg.get('time_limit', 3600 * 48))
        print('finished')

    def debug_actor(self, actor_name):
        # kill the existing one if any
        try:
            actor = ray.get_actor(actor_name)
            # if actor exists, kill it
            print(f'killing the existing {actor_name}...')
            ray.kill(actor)

        except ValueError as e:
            # if actor_name doesn't exist, ray will raise a ValueError exception saying this
            print(e)

        print(f'instantiating {actor_name} locally for debug...')
        if actor_name == 'learner':
            learner = CutDQNLearner(hparams=self.cfg, use_gpu=self.learner_gpu, gpu_id=self.cfg.get('gpu_id', None), run_io=True)
            learner.run()
        elif actor_name == 'tester':
            tester = CutDQNWorker('Test', hparams=self.cfg, is_tester=True, use_gpu=self.tester_gpu, gpu_id=self.cfg.get('gpu_id', None))
            tester.run()
        elif actor_name == 'replay_server':
            replay_server = PrioritizedReplayServer(config=self.cfg)
            replay_server.run()
        else:
            prefix, worker_id = actor_name.split('_')
            worker_id = int(worker_id)
            assert prefix == 'worker' and worker_id in range(1, self.num_workers + 1)
            worker = CutDQNWorker(worker_id, hparams=self.cfg, use_gpu=self.worker_gpu, gpu_id=self.cfg.get('gpu_id', None))
            worker.run()

    def kill(self, actors=[]):
        """ kills the actors running in the current ray server
        :param actors: list of actors to kill. if not specified kills all actors."""
        actors = list(self.actors.keys()) if len(actors) == 0 else actors
        for actor_name in actors:
            try:
                actor = ray.get_actor(actor_name)
                ray.kill(actor)
                print(f'{actor_name} killed')
            except ValueError as e:
                # if actor_name doesn't exist, ray will raise a ValueError exception saying this
                print(e)

    def clean_zombie_actors(self):
        """ kills processes that use the current experiment ports """
        ports_file = os.path.join(self.cfg["rootdir"], self.cfg['run_id'], 'com_cfg.pkl')
        if os.path.exists(ports_file):
            with open(ports_file, 'rb') as f:
                ports = pickle.load(f)
            print('loaded ports from ', self.cfg['run_dir'])
            print('killing all processes running on ', ports)
            for port in ports.values():
                print(f'(port {port}) killing pid:')
                os.system('echo $(lsof -t -i:8080)')
                os.system('kill -9 $(lsof -t -i:8080)')

