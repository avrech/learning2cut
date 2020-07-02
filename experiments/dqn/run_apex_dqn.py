from distributed.apex_dqn import ApeXDQN
import argparse
import yaml
import os
import cProfile
import pstats
import io


if __name__ == '__main__':
    """
    Run Ape-X DQN
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='results/apex',
                        help='path to save results')
    parser.add_argument('--datadir', type=str, default='data/maxcut',
                        help='path to generate/read data')
    parser.add_argument('--configfile', type=str, default='experiment_config.yaml',
                        help='general experiment settings')
    parser.add_argument('--resume-training', action='store_true',
                        help='set to load the last training status from checkpoint file')
    parser.add_argument('--gpu-id', type=int, default=None,
                        help='gpu id to use if available')
    parser.add_argument('--use-gpu', action='store_true',
                        help='use gpu for learner')
    parser.add_argument('--profile', action='store_true',
                        help='run with cProfile')
    parser.add_argument('--time-limit', type=float, default=3600*24,
                        help='global time limit in seconds')
    parser.add_argument('--restart', action='store_true',
                        help='restart Ape-X detached actors. '
                             'if no actor name is specified, restarts only failed actors.'
                             'a running actor will be restarted only if --force-restart is enabled.'
                             'NOTE: while restarting profiling is not optional, and the --profile flag is ignored')
    parser.add_argument('--restart-actors', nargs='+', type=str, default=[],
                        help='[optional] list of actors to restart'
                             'options: replay_buffer, learner, tester, worker_<id> (id=0,1,2,...)')
    parser.add_argument('--force-restart', action='store_true',
                        help='force restart of detached actors')

    args = parser.parse_args()
    with open(args.configfile) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        config[k] = v
    if config.get('debug_cuda', False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    def main():
        apex = ApeXDQN(cfg=config, use_gpu=args.use_gpu)
        apex.spawn()
        apex.train()

    if args.restart:
        apex = ApeXDQN(cfg=config, use_gpu=args.use_gpu)
        apex.restart(actors=args.restart_actors, force_restart=args.force_restart)

    elif args.profile:
        # profile main() and print stats to readable file
        pr = cProfile.Profile()
        pr.enable()

        main()

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        with open(os.path.join(args.logdir, 'profile_pstats.txt'), 'w+') as f:
            f.write(s.getvalue())
        print('Saved pstats to ', os.path.join(args.logdir, 'profile_pstats.txt'))

    else:
        main()
