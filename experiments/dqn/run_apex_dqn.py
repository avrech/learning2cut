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
    parser.add_argument('--configfile', type=str, default='configs/experiment_config.yaml',
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
    parser.add_argument('--debug-actor', type=str, default=None,
                        help='allows debugging the specified actor locally while the rest of actors run remotely.'
                             'only one actor can be debugged at a time.'
                             'if there is already a running ray server, set --restart to connect to this instance.'
                             'in this case the other actors will not be affected.'
                             'after debugging, the specified actor can be killed and restarted as usual'
                             'using --restart --restart-actors <actor_name> --force-restart')

    args = parser.parse_args()
    with open(args.configfile) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        config[k] = v
    if config.get('debug_cuda', False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if args.restart:
        # connect to the existing ray server
        ray.init(ignore_reinit_error=True, address='auto')
    else:
        # create a new ray server.
        ray.init()  # todo - do we need ignore_reinit_error=True to launch several ray servers concurrently?

    # instantiate apex launcher
    apex = ApeXDQN(cfg=config, use_gpu=args.use_gpu)

    def main():
        apex.spawn()
        apex.train()

    if args.debug_actor is not None:
        # spawn all the other actors as usual
        all_actor_names = apex.actors.copy()
        debug_actor = all_actor_names.pop(args.debug_actor)
        rest_of_actors = list(all_actor_names.keys())
        apex.restart(actors=rest_of_actors)

        # run the debugged actor locally
        apex.debug_actor(actor_name=args.debug_actor)

    elif args.restart:
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
