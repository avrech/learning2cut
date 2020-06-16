from distributed.worker import GDQNWorker
from distributed.dqn_learner import GDQNLearner
from distributed.per_server import PrioritizedReplayServer
import argparse
import yaml
import os

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
    parser.add_argument('--logdir', type=str, default='unittest_results',
                        help='path to save results')
    parser.add_argument('--datadir', type=str, default='../experiments/dqn/data/maxcut',
                        help='path to generate/read data')
    parser.add_argument('--configfile', type=str, default='../experiments/dqn/test_config.yaml',
                        help='general experiment settings')
    parser.add_argument('--resume-training', action='store_true',
                        help='set to load the last training status from checkpoint file')
    parser.add_argument('--mixed-debug', action='store_true',
                        help='set for mixed python/c debugging')
    parser.add_argument('--gpu-id', type=int, default=None,
                        help='gpu id to use if available')

    args = parser.parse_args()
    with open(args.configfile) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        hparams[k] = v
    if hparams.get('debug_cuda', False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # modify hparams to fit workers and learner
    workers = [GDQNWorker(worker_id=worker_id, hparams=hparams, use_gpu=False) for worker_id in range(2)]
    test_worker = GDQNWorker(worker_id='Tester', hparams=hparams, is_tester=True, use_gpu=False)
    learner = GDQNLearner(hparams=hparams, use_gpu=True, gpu_id=args.gpu_id)
    replay_server = PrioritizedReplayServer(config=hparams)

    for worker in workers:
        worker.initialize_training()
        worker.load_datasets()
    test_worker.initialize_training()
    test_worker.load_datasets()
    learner.initialize_training()

    while True:
        # WORKER SIDE
        for worker in workers:
            local_buffer = worker.collect_data()

            # local_buffer is full enough. pack the local buffer and send packet
            replay_data_packet = GDQNWorker.pack_replay_data(local_buffer)  # todo serialize
            # the packet should be sent here and received on the listening socket.

            # PER SERVER SIDE
            # a packet from worker should be received here.
            # unpack replay data and push into PER
            unpacked_buffer = replay_server.unpack_replay_data(replay_data_packet)
            # push into memory
            replay_server.add_data_list(unpacked_buffer)

        # encode PER->Learner packet and send to Learner
        if len(replay_server) >= replay_server.batch_size:
            batch_packet = replay_server.get_batch_packet()
            # batch_packet should be sent here and received on the Learner side

            # LEARNER SIDE
            # unpack batch_packet
            batch_transitions, batch_weights, batch_idxes, batch_ids = learner.unpack_batch_packet(batch_packet)
            # perform sgd step and return new priorities to replay server
            batch_new_priorities = learner.sgd_step(transitions=batch_transitions, importance_sampling_correction_weights=batch_weights)
            # send back to per the new priorities together with the corresponding idxes
            new_priorities_packet = GDQNLearner.pack_priorities((batch_idxes, batch_new_priorities, batch_ids))
            # priorities_packet should be sent here and received on the per side

            # PER SERVER SIDE
            idxes, new_priorities, data_ids = replay_server.unpack_priorities(new_priorities_packet)
            # update priorities
            replay_server.update_priorities(idxes, new_priorities, data_ids=data_ids)

        if learner.num_sgd_steps_done > 0 and learner.num_sgd_steps_done % hparams['param_update_interval'] == 0:
            # LEARNER SIDE
            # pack new params and broadcast to all workers
            public_params_packet = learner.get_params_packet()
            learner.num_param_updates += 1
            # the new params should be broadcasted here to all workers.

            # WORKER SIDE
            for worker in workers:
                worker.synchronize_params(public_params_packet)
                worker.num_param_updates += 1
            test_worker.synchronize_params(public_params_packet)
            test_worker.num_param_updates += 1
            received_new_params = True
        else:
            received_new_params = False

        # update learner target policy periodically
        if learner.num_sgd_steps_done > 0 and learner.num_sgd_steps_done % hparams.get('target_update_interval', 1000) == 0:
            learner.update_target()

        global_step = learner.num_param_updates
        if global_step > 0 and global_step % hparams.get('log_interval', 100) == 0:
            learner.log_stats()

        # TEST WORKER SIDE
        if received_new_params:
            test_worker.evaluate()
            # frequently checkpoint all components
            learner.save_checkpoint()
            # todo replay_server.save_checkpoint()
            test_worker.save_checkpoint()
            for worker in workers:
                worker.save_checkpoint()

