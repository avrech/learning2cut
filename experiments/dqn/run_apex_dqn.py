from distributed.apex_dqn import ApeXDQN
import argparse
import yaml
import os
import ray
import cProfile
import pstats
import io
import wandb
import pickle


if __name__ == '__main__':
    """
    Run Ape-X DQN
    """
    from experiments.dqn.default_parser import parser, get_hparams
    # parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true',
                        help='run with cProfile')
    parser.add_argument('--time-limit', type=float, default=3600*24,
                        help='global time limit in seconds')
    parser.add_argument('--restart', action='store_true',
                        help='restart actors. '
                             'by default, restarts only failed actors. use --restart-actors for restarting specific actors. '
                             'a running actor will be restarted only if --force-restart is set.')
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
    parser.add_argument('--kill-actors', nargs='+', type=str, default=[],
                        help='list of actors to kill. if not specified, kills all running actors')
    parser.add_argument('--kill', action='store_true',
                        help='kill all running actors or a specified lise of actors in the currently running ray server')
                        # todo support multiple ray servers running in parallel. how to link to the correct one.

    args = parser.parse_args()
    config = get_hparams(args)
    assert (not args.restart) or (args.resume and args.run_id is not None), 'provide wandb run_id for resuming'
    run_id = args.run_id if args.resume else wandb.util.generate_id()
    config['run_id'] = run_id
    config['run_dir'] = run_dir = os.path.join(args.rootdir, run_id)

    if config.get('debug_cuda', False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if not os.path.exists(args.rootdir):
        os.makedirs(args.rootdir)

    if (args.restart or args.kill):
        # load ray server info from run_dir
        with open(os.path.join(run_dir, 'ray_info.pkl'), 'rb') as f:
            ray_info = pickle.load(f)
        # connect to the existing ray server
        ray_info = ray.init(ignore_reinit_error=True, address=ray_info['redis_address'])
    else:
        # create a new ray server.
        ray_info = ray.init()  # todo - do we need ignore_reinit_error=True to launch several ray servers concurrently?

    if not os.path.exists(config['run_dir']):
        os.makedirs(config['run_dir'])
    # save ray server info for reconnecting
    with open(os.path.join(run_dir, 'ray_info.pkl'), 'wb') as f:
        pickle.dump(ray_info, f)

    # instantiate apex launcher
    apex = ApeXDQN(cfg=config, use_gpu=args.use_gpu)

    def main():
        apex.spawn()
        apex.train()

    if args.kill:
        apex.kill(args.kill_actors)

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
        with open(os.path.join(args.rootdir, 'profile_pstats.txt'), 'w+') as f:
            f.write(s.getvalue())
        print('Saved pstats to ', os.path.join(args.rootdir, 'profile_pstats.txt'))

    else:
        main()
