from distributed.worker import DQNWorker
from distributed.dqn_learner import DQNLearner
from distributed.per_server import PrioritizedReplayBufferServer
from distributed.param_server import ParameterServer
import argparse
import yaml
import os


def test_distributed_functionality(self):
    """
    Tests the following functionality:
    1. Worker -> PER packets:
        a. store transitions in local buffer.
        b. encode local buffer periodically and send to PER.
        c. decode worker's packet and push transitions into the PER storage
    2. PER <-> Learner packets:
        a. encode transitions batch and send to Learner
        b. decode PER->Learner packet and perform SGD
        c. send back encoded new_priorities to PER, decode on PER side, and update priorities

    Learner -> ParamServer -> Worker packets are not tested, since we didn't touch this type of packets.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='results',
                        help='path to save results')
    parser.add_argument('--datadir', type=str, default='data/maxcut',
                        help='path to generate/read data')
    parser.add_argument('--configfile', type=str, default='experiment_config.yaml',
                        help='general experiment settings')
    parser.add_argument('--resume-training', action='store_true',
                        help='set to load the last training status from checkpoint file')
    parser.add_argument('--mixed-debug', action='store_true',
                        help='set for mixed python/c debugging')
    parser.add_argument('--gpu-id', type=int, default=None,
                        help='gpu id to use if available')

    args = parser.parse_args()
    if args.mixed_debug:
        import ptvsd
        port = 3000
        # ptvsd.enable_attach(secret='my_secret', address =('127.0.0.1', port))
        ptvsd.enable_attach(address=('127.0.0.1', port))
        ptvsd.wait_for_attach()

    with open(args.configfile) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        hparams[k] = v
    if hparams.get('debug_cuda', False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # # set logdir according to hparams
    # relative_logdir = f"lr_{hparams['lr']}-nstep_{hparams['nstep_learning']}-credit_{hparams['credit_assignment']}-gamma_{hparams['gamma']}-obj_{hparams['dqn_objective']}"
    # hparams['logdir'] = os.path.join(hparams['logdir'], relative_logdir)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)



    workers = [DQNWorker(worker_id=worker_id, cfg=hparams, common_config=hparams, worker_brain=None) for worker_id in range(2)]
    learner = DQNLearner(hparams=hparams)
    per_server = PrioritizedReplayBufferServer(buffer_cfg=hparams, comm_cfg=hparams)
    param_server = ParameterServer()

    for worker in workers:
        worker.initialize_training()
        worker.load_datasets()

    while True:
        # WORKER SIDE
        for worker in workers:
            local_buffer = worker.collect_data()

        # self.local_buffer is full enough. encode Worker->PER packet and send
        worker_to_per_message = DQNWorker.pack_message_to_per(self.local_buffer)  # todo serialize
        # flush the local buffer
        self.local_buffer = []
        # the packet should be sent here and received on the listening socket.

        # PER SERVER SIDE
        # worker2per_packet from worker should be received here.
        # unpack Worker->PER packet and push into PER
        buffer = PERServer.unpack_message_from_worker(worker_to_per_message)
        # push into memory
        self.memory.add_buffer(buffer)

        # encode PER->Learner packet and send to Learner
        if len(self.memory) >= self.batch_size:
            # todo sample with beta
            beta = self.priority_beta_end - (self.priority_beta_end - self.priority_beta_start) * math.exp(
                -1. * self.num_sgd_steps_done / self.priority_beta_decay)
            transitions, weights, idxes = self.memory.sample(self.batch_size, beta)

            per_to_learner_message = PERServer.pack_message_to_leaner((transitions, weights, idxes))

            # the message should be sent here and received on the Learner side

            # LEARNER SIDE
            unpacked_transitions, unpacked_weights, unpacked_idxes = DQNLearner.unpack_message_from_per(
                per_to_learner_message)
            # perform sgd step and return new priorities to per server
            new_priorities = self.sgd_step(transitions=unpacked_transitions,
                                           importance_sampling_correction_weights=unpacked_weights)
            # send back to per the new priorities together with the corresponding idxes
            learner_to_per_message = DQNLearner.pack_message_to_per((new_priorities, unpacked_idxes))
            # the message should be sent here and received on the per side

            # PER SERVER SIDE
            new_priorities_from_learner, idxes_from_learner = PERServer.unpack_message_from_learner(
                learner_to_per_message)
            # update priorities
            self.memory.update_priorities(idxes_from_learner, new_priorities_from_learner)

            # LEARNER SIDE
            # pack new params to parameter server and send
            learner_to_param_server_message = DQNLearner.pack_message_to_param_server()

            self.num_policy_updates += 1

        # decode packet on Learner and perform SGD step

        # send back new_priorities to PER, and update

        # # perform 1 optimization step
        # self.optimize_model()

        # this part is the same as in train_single_thread()
        if self.num_sgd_steps_done % hparams.get('target_update_interval', 1000) == 0:
            self.update_target()

        if self.num_policy_updates > 0:
            global_step = self.num_policy_updates
            if global_step % hparams.get('log_interval', 100) == 0:
                self.log_stats()

            # evaluate periodically
            self.evaluate()

            if global_step % hparams.get('checkpoint_interval', 100) == 0:
                self.save_checkpoint()

    return 0
