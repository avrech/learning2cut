from distributed.cut_dqn_worker import CutDQNWorker
from distributed.cut_dqn_learner import CutDQNLearner
from distributed.replay_server import PrioritizedReplayServer
import argparse
import yaml
import os

if __name__ == '__main__':
    """
    Tests the following functionality:
    1. Workers -> ReplayBuffer packets:
        a. store transitions in local buffer on the worker side.
        b. send replay data packet to repay server via zmq socket.
        c. receive replay data on replay server socket, and push into memory. 
    2. ReplayBuffer <-> Learner packets:
        a. send batches to learner up to max_pending_requests. 
        b. receive batch on learner side and push to queue.
        c. pull beach from learner queue, perform SGD and push new priorities from queue.
        d. pull new priorities from queue and send to replay server.
        e. receive new priorities on replay server side and update.  
    3. Learner ->  Workers packets
        a. publish new params to workers. 
        b. each worker receives new params and update policy.
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
    workers = [CutDQNWorker(worker_id=worker_id, hparams=hparams, use_gpu=False) for worker_id in range(1, hparams['num_workers']+1)]
    test_worker = CutDQNWorker(worker_id='Tester', hparams=hparams, is_tester=True, use_gpu=False)
    learner = CutDQNLearner(hparams=hparams, use_gpu=True, gpu_id=args.gpu_id)
    replay_server = PrioritizedReplayServer(config=hparams)

    for worker in workers:
        worker.initialize_training()
        worker.load_datasets()
    test_worker.initialize_training()
    test_worker.load_datasets()
    learner.initialize_training()

    while True:
        # WORKER SIDE
        # collect data and send to replay server, one packet per worker
        for worker in workers:
            replay_data = worker.collect_data()
            # local_buffer is full enough. pack the local buffer and send packet
            worker.send_replay_data(replay_data)

        # REPLAY SERVER SIDE
        # try receiving up to num_workers replay data packets,
        # each received packets is unpacked and pushed into memory
        replay_server.recv_replay_data()
        # send batches to learner
        replay_server.send_batches()
        # wait for new priorities
        # (in the real application - receive new replay data from workers in the meanwhile)

        # LEARNER SIDE
        # in the real application, the learner receives batches and sends back new priorities in a separate thread,
        # while pulling processing the batches in another asynchronous thread.
        # here we alternate receiving a batch, processing and sending back priorities,
        # until no more waiting batches exist.
        while learner.recv_batch(blocking=False):
            # receive batch from replay server and push into learner.replay_data_queue
            # pull batch from queue, process and push new priorities to learner.new_priorities_queue
            learner.optimize_model()
            # push new params to learner.new_params_queue periodically
            learner.prepare_new_params_to_workers()
            # send back the new priorities
            learner.send_new_priorities()
            # publish new params if any available in the new_params_queue
            learner.publish_params()

        # PER SERVER SIDE
        # receive all the waiting new_priorities packets and update memory
        replay_server.recv_new_priorities()
        if replay_server.num_sgd_steps_done > 0 and replay_server.num_sgd_steps_done % replay_server.checkpoint_interval == 0:
            replay_server.save_checkpoint()

        # WORKER SIDE
        # subscribe new_params from learner
        for worker in workers:
            if worker.recv_new_params():
                worker.log_stats(global_step=worker.num_param_updates - 1)

        # TEST WORKER
        # if new params have been published, evaluate the new policy
        if test_worker.recv_new_params():
            test_worker.evaluate()
            test_worker.save_checkpoint()


