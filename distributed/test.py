from distributed.worker import GDQNWorker
from distributed.dqn_learner import DQNLearner
from distributed.per_server import PrioritizedReplayBufferServer
from distributed.param_server import ParameterServer
import argparse
import yaml
import os
import math

if __name__ == '__main__':
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
    parser.add_argument('--logdir', type=str, default='results/distributed_functionality_test',
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



    workers = [GDQNWorker(worker_id=worker_id, cfg=hparams, common_config=hparams, worker_brain=None) for worker_id in range(2)]
    test_worker = GDQNWorker(worker_id='Tester', cfg=hparams, common_config=hparams, worker_brain=None)
    learner = DQNLearner(hparams=hparams)
    # todo ensure workers and learner are assigned correct is_<role>
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
            worker_to_per_message = GDQNWorker.pack_message_to_per(local_buffer)  # todo serialize
            # flush the local buffer
            worker.local_buffer = []
            # the packet should be sent here and received on the listening socket.

            # PER SERVER SIDE
            # worker2per_packet from worker should be received here.
            # unpack Worker->PER packet and push into PER
            unpacked_buffer = per_server.unpack_message_from_worker(worker_to_per_message)
            # push into memory
            per_server.add_buffer(unpacked_buffer)

        # encode PER->Learner packet and send to Learner
        if len(per_server.storage) >= per_server.batch_size:
            # todo sample with beta
            beta = per_server.priority_beta_end - (per_server.priority_beta_end - per_server.priority_beta_start) * math.exp(
                -1. * per_server.num_sgd_steps_done / per_server.priority_beta_decay)
            transitions, weights, idxes = per_server.sample(self.batch_size, beta)

            per_to_learner_message = per_server.pack_message_to_leaner((transitions, weights, idxes))

            # the message should be sent here and received on the Learner side

            # LEARNER SIDE
            # decode packet on Learner and perform SGD step
            unpacked_transitions, unpacked_weights, unpacked_idxes = learner.unpack_message_from_per(per_to_learner_message)
            # perform sgd step and return new priorities to per server
            new_priorities = learner.sgd_step(transitions=unpacked_transitions, importance_sampling_correction_weights=unpacked_weights)
            # send back to per the new priorities together with the corresponding idxes
            learner_to_per_message = DQNLearner.pack_message_to_per((new_priorities, unpacked_idxes))
            # the message should be sent here and received on the per side

            # PER SERVER SIDE
            new_priorities_from_learner, idxes_from_learner = per_server.unpack_message_from_learner(learner_to_per_message)
            # update priorities
            per_server.update_priorities(idxes_from_learner, new_priorities_from_learner)

        if learner.num_sgd_steps_done % hparams['param_update_interval'] == 0:
            # LEARNER SIDE
            # pack new params to parameter server and send
            learner_to_param_server_message = learner.pack_message_to_param_server()
            learner.num_param_updates += 1

            # PARAM SERVER SIDE
            new_params = param_server.unpack_message_from_learner(learner_to_param_server_message)
            param_server.update_params(new_params) # todo - verify
            param_server.num_param_updates += 1
            param_server_to_worker_message = param_server.pack_message_to_worker
            # the new params should be broadcasted here to all workers.

            # WORKER SIDE
            for worker in workers:
                unpacked_new_params = worker.unpack_message_from_param_server(param_server_to_worker_message)
                worker.update_params(unpacked_new_params)
                worker.num_param_updates += 1
            unpacked_new_params = test_worker.unpack_message_from_param_server(param_server_to_worker_message)
            test_worker.update_params(unpacked_new_params)
            test_worker.num_param_updates += 1

        # update learner target policy periodically
        if learner.num_sgd_steps_done % hparams.get('target_update_interval', 1000) == 0:
            learner.update_target()

        global_step = learner.num_param_updates
        if learner.num_param_updates > 0:
            if global_step % hparams.get('log_interval', 100) == 0:
                learner.log_stats()

        # TEST WORKER SIDE
        test_worker.evaluate()
        if global_step % hparams.get('checkpoint_interval', 100) == 0:
            learner.save_checkpoint()
            per_server.save_checkpoint()
            test_worker.save_checkpoint()
            for worker in workers:
                worker.save_checkpoint()

    return 0
