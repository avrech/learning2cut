import ray
from distributed.replay_server import PrioritizedReplayServer
from distributed.cut_dqn_worker import CutDQNWorker
from distributed.cut_dqn_learner import CutDQNLearner
import os
import pickle
import wandb
import zmq
import pyarrow as pa
import psutil


class ApeXDQN:
    """ Apex-DQN controller for learning to cut
    This class spawns remote actors (replay server, learner, multiple workers and tester) as in the standard Apex paper.
    Here goes technical description of how the distributed part works. For algorithmic details refer to ???
    Implementation principles:
    ApexDQN:
        central controller, responsible for spawning, restarting, resuming and debugging the remote actors,
        and for aggregating logs from all actors and logging them to wandb.
    PrioritizedReplayServer:
        replay server class. receives transitions from workers, sends batches to learner, receives and updates priorities.
    CutDQNLearner:
        learner class. receives batches from replay server, backprops and returns priorities.
        updates periodically all workers with model parameters.
    CutDQNWorker:
        worker and tester class. generates data for training.
        a test worker evaluates periodically the model performance.

    Except the central ApexDQN controller, all those actors runs as Ray detached actors. Each actor can be restarted
    and debugged while the others keep running. Since ray starts a separate python driver for each actor,
    an actor can be restarted with potentially updated code, while the rest of actors are still running.
    Each entity has a ray unique name, which can be used for accessing the specific actor.
    Actors communicate via zmq sockets. The learner, the replay server and the central unit binds to tcp ports,
    making those ports exclusive for the current run.
    In order to allow multiple Apex instances on a single machine, a setup routine is executed at the beginning of
    each run, binding all actors to randomly selected free tcp ports. Those ports are saved to run_dir, and can
    be reused when restarting a part of the system.
    The pid and tcp ports of the latest run are saved to run_dir, allowing tracking and cleaning zombie processes.

    The communication setup goes as follows:
    1. ApexDQN binds to apex_port, starts learner and waits for learner tcp ports.
    2. Learner in its turn connects to apex_port,
       binds to replay_server_2_learner_port and learner_2_workers_pubsub_port,
       sends its port numbers to apex_port and waits for port numbers from the replay server.
    3. Replay server connects to learner and apex ports,
       binds to learner_2_replay_server_port, workers_2_replay_server_port and replay_server_2_workers_pubsub_port,
       sends port numbers tp apex and learner ports, and starts its main loop.
    4. Workers connect to learner replay and apex port numbers and start their main loop.
    5. Learner connects to learner_2_replay_server_port, and starts its main loop.
    6. ApexDQN saves all port numbers, process pids and ray server info to run dir
       and starts its main logging loop.

    When restarting a specific actor, those ports are loaded and reused.
    When restarting the entire system, this setup routine is repeated, overriding the exisiting configuration.
    """
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
        # set run dir
        run_id = self.cfg['run_id'] if self.cfg['resume'] else wandb.util.generate_id()
        self.cfg['run_id'] = run_id
        self.cfg['run_dir'] = run_dir = os.path.join(self.cfg['rootdir'], run_id)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        # initialize ray server
        self.init_ray()
        # apex controller socket for receiving logs
        self.apex_socket = None
        # logging
        self.step_counter = {actor_name: -1 for actor_name in self.actors.keys() if actor_name != 'replay_server'}
        self.pending_logs = []
        self.history = {}

        # reuse communication setting
        if cfg['restart']:
            assert len(self.get_ray_running_actors()) > 0, 'no running actors exist. run without --restart'
            with open(os.path.join(self.cfg['run_dir'], 'com_cfg.pkl'), 'rb') as f:
                self.cfg['com'] = pickle.load(f)
            print('loaded communication config from ', os.path.join(self.cfg['run_dir'], 'com_cfg.pkl'))

        # else:
        #     com_cfg = self.find_free_ports()
        #     # pickle ports to experiment dir
        #     with open(os.path.join(self.cfg['run_dir'], 'com_cfg.pkl'), 'wb') as f:
        #         pickle.dump(com_cfg, f)
        #     print('saved ports to ', os.path.join(self.cfg['run_dir'], 'com_cfg.pkl'))
        # self.cfg['com'] = com_cfg

    def init_ray(self):
        if self.cfg['restart']:
            # load ray server address from run_dir
            with open(os.path.join(self.cfg['run_dir'], 'ray_info.pkl'), 'rb') as f:
                ray_info = pickle.load(f)
            # connect to the existing ray server
            ray_info = ray.init(ignore_reinit_error=True, address=ray_info['redis_address'])
        else:
            if self.cfg['resume']:
                # check first that there are no processes running with the current run_id
                running_processes = self.get_actors_running_process()
                if any(running_processes.values()):
                    self.print('running processes found for this run_id:')
                    for actor_name, process in running_processes.items():
                        if process is not None:
                            print(f'{actor_name}: pid {process.pid}')
                    self.print('run again with --restart --force-restart to kill the existing processes')
                    exit(0)

            # create a new ray server.
            ray_info = ray.init()  # todo - do we need ignore_reinit_error=True to launch several ray servers concurrently?
        self.cfg['ray_info'] = ray_info
        # save ray info for reconnecting
        with open(os.path.join(self.cfg['run_dir'], 'ray_info.pkl'), 'wb') as f:
            pickle.dump(ray_info, f)

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

    # def spawn(self):
    #     """
    #     Instantiate all components as Ray detached Actors.
    #     Detached actors have global unique names, they run independently of the current python driver.
    #     Detached actors can be killed and restarted, with potentially updated code.
    #     Use case: when debugging/upgrading the learner/tester code, while the replay server keeps running.
    #     For reference see: https://docs.ray.io/en/master/advanced.html#dynamic-remote-parameters
    #     the "Detached Actors" section.
    #     In the setup process actors incrementally bind to random free ports,
    #     to allow multiple instances running on the same node.
    #     """
    #     # wrap base classes with ray.remote to make them remote "Actor"s
    #     ray_worker = ray.remote(num_gpus=int(self.worker_gpu), num_cpus=1)(CutDQNWorker)
    #     ray_tester = ray.remote(num_gpus=int(self.tester_gpu), num_cpus=1)(CutDQNWorker)
    #     ray_learner = ray.remote(num_gpus=int(self.learner_gpu), num_cpus=2)(CutDQNLearner)
    #     ray_replay_server = ray.remote(PrioritizedReplayServer)
    #
    #     # spawn all actors as detached actors with globally unique names.
    #     # those detached actors can be accessed from any driver connecting to the current ray server
    #     # using their global unique names.
    #     for n in range(1, self.num_workers + 1):
    #         self.actors[f'worker_{n}'] = ray_worker.options(name=f'worker_{n}').remote(n, hparams=self.cfg, use_gpu=self.worker_gpu)
    #     self.actors['tester'] = ray_tester.options(name='tester').remote('Test', hparams=self.cfg, use_gpu=self.tester_gpu, is_tester=True)
    #     # instantiate learner and run its io process in a background thread
    #     self.actors['learner'] = ray_learner.options(name='learner').remote(hparams=self.cfg, use_gpu=self.learner_gpu, run_io=True)
    #     self.actors['replay_server'] = ray_replay_server.options(name='replay_server').remote(config=self.cfg)

    def setup(self):
        """
        Instantiate all components as Ray detached Actors.
        Detached actors have global unique names, they run independently of the current python driver.
        Detached actors can be killed and restarted, with potentially updated code.
        Use case: when debugging/upgrading the learner/tester code, while the replay server keeps running.
        For reference see: https://docs.ray.io/en/master/advanced.html#dynamic-remote-parameters
        the "Detached Actors" section.
        In the setup process actors incrementally bind to random free ports,
        to allow multiple instances running on the same node.
        """
        assert 'com' not in self.cfg.keys()
        # open main logger socket for receiving logs from all actors
        context = zmq.Context()
        self.apex_socket = context.socket(zmq.PULL)
        apex_port = self.apex_socket.bind_to_random_port('tcp://127.0.0.1', min_port=10000, max_port=60000)
        self.cfg['com'] = {'apex_port': apex_port}
        self.print(f"binding to {apex_port} for receiving logs")

        # spawn learner
        self.print('spawning learner process')
        ray_learner = ray.remote(num_gpus=int(self.learner_gpu), num_cpus=2)(CutDQNLearner)
        # instantiate learner and run its io process in a background thread
        self.actors['learner'] = ray_learner.options(name='learner').remote(hparams=self.cfg, use_gpu=self.learner_gpu, run_io=True, run_setup=True)
        # wait for learner's com config
        learner_msg = self.apex_socket.recv()
        topic, body = pa.deserialize(learner_msg)
        assert topic == 'learner_com_cfg'
        for k, v in body:
            self.cfg['com'][k] = v

        # spawn replay server
        self.print('spawning replay server process')
        ray_replay_server = ray.remote(PrioritizedReplayServer)
        self.actors['replay_server'] = ray_replay_server.options(name='replay_server').remote(config=self.cfg, run_setup=True)
        # todo go to replay_server, connect to apex port. bind to others, send com config, and start run
        # wait for replay_server's com config
        replay_server_msg = self.apex_socket.recv()
        topic, body = pa.deserialize(replay_server_msg)
        assert topic == 'replay_server_com_cfg'
        for k, v in body:
            self.cfg['com'][k] = v

        # spawn workers and tester
        self.print('spawning workers and tester processes')
        ray_worker = ray.remote(num_gpus=int(self.worker_gpu), num_cpus=1)(CutDQNWorker)
        ray_tester = ray.remote(num_gpus=int(self.tester_gpu), num_cpus=1)(CutDQNWorker)
        for n in range(1, self.num_workers + 1):
            self.actors[f'worker_{n}'] = ray_worker.options(name=f'worker_{n}').remote(n, hparams=self.cfg, use_gpu=self.worker_gpu)
        self.actors['tester'] = ray_tester.options(name='tester').remote('Test', hparams=self.cfg, use_gpu=self.tester_gpu, is_tester=True)

        # pickle com config to experiment dir
        with open(os.path.join(self.cfg['run_dir'], 'com_cfg.pkl'), 'wb') as f:
            pickle.dump(self.cfg['com'], f)
        self.print(f'saving communication config to {os.path.join(self.cfg["run_dir"], "com_cfg.pkl")}')

        # initialize wandb logger
        # todo wandb
        self.print('initializing wandb')
        wandb_config = self.cfg.copy()
        wandb_config.pop('datasets')
        wandb_config.pop('com')
        wandb_config.pop('ray_info')
        wandb.init(resume='allow',  # hparams['resume'],
                   id=self.cfg['run_id'],
                   project=self.cfg['project'],
                   config=wandb_config)

        # save pid to run_dir
        pid = os.getpid()
        pid_file = os.path.join(self.cfg["run_dir"], 'apex_pid.txt')
        self.print(f'[Apex] saving pid {pid} to {pid_file}')
        with open(pid_file, 'w') as f:
            f.writelines(str(pid) + '\n')
        self.print('setup finished')

    def train(self):
        print("[Apex] running logger loop")
        # ready_ids, remaining_ids = ray.wait([actor.run.remote() for actor in self.actors.values()])
        for actor in self.actors.values():
            actor.run.remote()
        self.wandb_loop()

    def wandb_loop(self):
        # todo - loop here wandb logs - send logs from all actors, recv here and wandb.log
        while True:
            # receive message
            packet = self.apex_socket.recv()
            topic, sender, body = pa.deserialize(packet)
            assert topic == 'log'
            # put things into a dictionary
            log_dict = {}
            for k, v in body:
                if type(v) == tuple and v[0] == 'fig':
                    log_dict[k] = wandb.Image(v[1], caption=k)
                else:
                    log_dict[k] = v
            global_step = log_dict.pop('global_step')
            # increment sender step counter
            self.step_counter[sender] = global_step
            # update history
            if global_step in self.history.keys():
                self.history[global_step].update(log_dict)
            else:
                self.history[global_step] = log_dict
            # push to pending logs
            if not self.pending_logs or global_step > self.pending_logs[-1]:
                self.pending_logs.append(global_step)
            # if all actors finished a certain step, log to wandb
            while self.pending_logs and all([self.pending_logs[0] <= cnt for cnt in self.step_counter.values()]):
                step = self.pending_logs.pop(0)
                log_dict = self.history.pop(step)
                wandb.log(log_dict, step=step)
            if len(self.pending_logs) > 1000:
                self.print('some actor is dead. restart to continue logging.')

        # ray.get(ready_ids + remaining_ids, timeout=self.cfg.get('time_limit', 3600*48))

    def get_actors_running_process(self, actors=None):
        actors = actors if actors is not None else list(self.actors.keys()) + ['apex']
        running_actors = {}
        for actor_name in actors:
            # read existing pid from run dir
            with open(os.path.join(self.cfg['run_dir'], actor_name + '_pid.txt'), 'r') as f:
                pid = int(f.readline().strip())
            try:
                os.kill(pid, 0)
            except OSError:
                print(f'{actor_name} process does not exist')
                running_actors[actor_name] = None
            else:
                print(f'found {actor_name} process (pid {pid})')
                running_actors[actor_name] = psutil.Process(pid)
        return running_actors

    def get_ray_running_actors(self, actors=None):
        actors = actors if actors is not None else list(self.actors.keys())
        running_actors = {}
        for actor_name in actors:
            if actor_name == 'apex':
                continue
            try:
                actor = ray.get_actor(actor_name)
                running_actors[actor_name] = actor
            except ValueError as e:
                # if actor_name doesn't exist, ray will raise a ValueError exception saying this
                print(e)
                running_actors[actor_name] = None
        return running_actors

    def restart(self, actors=[], force_restart=False):
        """ restart actors as remote entities """
        actors = list(self.actors.keys()) + ['apex'] if len(actors) == 0 else actors
        # move 'apex' to the end of list
        if 'apex' in actors:
            actors.append(actors.pop(actors.index('apex')))

        ray_running_actors = self.get_ray_running_actors(actors)
        actor_processes = self.get_actors_running_process(actors)

        ray_worker = ray.remote(num_gpus=int(self.worker_gpu), num_cpus=1)(CutDQNWorker)
        ray_tester = ray.remote(num_gpus=int(self.tester_gpu), num_cpus=1)(CutDQNWorker)
        ray_learner = ray.remote(num_gpus=int(self.learner_gpu), num_cpus=2)(CutDQNLearner)
        ray_replay_server = ray.remote(PrioritizedReplayServer)
        handles = []
        # restart all actors
        for actor_name, actor_process in actor_processes.items():
            # actor_process = running_actors[actor_name]
            if actor_process is not None:
                if force_restart:
                    print(f'killing {actor_name} process (pid {actor_process.pid})')
                    if actor_name == 'apex':
                        actor_process.kill()
                    else:
                        ray.kill(ray_running_actors[actor_name])
                else:
                    print(f'{actor_name} is running (pid {actor_process.pid}). '
                          f'use --force-restart for killing running actors and restarting new ones.')
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
            elif 'worker' in actor_name:
                prefix, worker_id = actor_name.split('_')
                worker_id = int(worker_id)
                assert prefix == 'worker' and worker_id in range(1, self.num_workers + 1)
                worker = ray_worker.options(name=actor_name).remote(worker_id, hparams=self.cfg, use_gpu=self.worker_gpu)
                handles.append(worker.run.remote())
            elif actor_name == 'apex':
                # bind to apex_port
                context = zmq.Context()
                self.apex_socket = context.socket(zmq.PULL)
                self.apex_socket.bind(f'tcp://127.0.0.1:{self.cfg["com"]["apex_port"]}')
                # initialize wandb logger
                self.print('initializing wandb')
                wandb_config = self.cfg.copy()
                wandb_config.pop('datasets')
                wandb_config.pop('com')
                wandb_config.pop('ray_info')
                wandb.init(resume='allow',  # hparams['resume'],
                           id=self.cfg['run_id'],
                           project=self.cfg['project'],
                           config=wandb_config)

                # save pid to run_dir
                pid = os.getpid()
                pid_file = os.path.join(self.cfg["run_dir"], 'apex_pid.txt')
                self.print('saving pid {pid} to {pid_file}')
                with open(pid_file, 'w') as f:
                    f.writelines(str(pid) + '\n')
                self.print('starting wandb loop')
                self.wandb_loop()

        # # todo - skip this when debugging an actor
        # if len(handles) > 0:
        #     ready_ids, remaining_ids = ray.wait(handles)
        #     # todo - find a good way to block the main program here, so ray will continue tracking all actors, restart etc.
        #     ray.get(ready_ids + remaining_ids, timeout=self.cfg.get('time_limit', 3600 * 48))
        print('finished')

    def run_debug(self, actor_name):
        # spawn all the other actors as usual
        rest_of_actors = [name for name in self.actors.keys() if name != actor_name]
        self.restart(actors=rest_of_actors)

        # debug actor locally
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

    def run(self):
        self.setup()
        self.train()

    # def kill(self, actors=[]):
    #     """ kills the actors running in the current ray server
    #     :param actors: list of actors to kill. if not specified kills all actors."""
    #     actors = list(self.actors.keys()) if len(actors) == 0 else actors
    #     for actor_name in actors:
    #         try:
    #             actor = ray.get_actor(actor_name)
    #             ray.kill(actor)
    #             print(f'{actor_name} killed')
    #         except ValueError as e:
    #             # if actor_name doesn't exist, ray will raise a ValueError exception saying this
    #             print(e)
    #
    # def clean_zombie_actors(self):
    #     """ kills processes that use the current experiment ports """
    #     ports_file = os.path.join(self.cfg["rootdir"], self.cfg['run_id'], 'com_cfg.pkl')
    #     if os.path.exists(ports_file):
    #         with open(ports_file, 'rb') as f:
    #             ports = pickle.load(f)
    #         print('loaded ports from ', self.cfg['run_dir'])
    #         print('killing all processes running on ', ports)
    #         for port in ports.values():
    #             print(f'(port {port}) killing pid:')
    #             os.system('echo $(lsof -t -i:8080)')
    #             os.system('kill -9 $(lsof -t -i:8080)')

    def print(self, expr):
        print('[Apex] ', expr)