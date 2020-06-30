from distributed.apex_dqn import ApeXDQN
import argparse
import yaml
import os
import ray
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
        ray.init(ignore_reinit_error=True)
        apex = ApeXDQN(cfg=config, use_gpu=args.use_gpu)
        apex.spawn()
        apex.train()

    if args.profile:
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
