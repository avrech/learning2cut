import os
from agents.cut_dqn_agent import CutDQNAgent


if __name__ == '__main__':
    import argparse
    import yaml
    from experiments.dqn.default_parser import parser, get_hparams
    # parser = argparse.ArgumentParser()
    parser.add_argument('--mixed-debug', action='store_true',
                        help='set for mixed python/c debugging')

    args = parser.parse_args()

    # get hparams for all modules
    hparams = get_hparams(args)
    run_id = args.run_id
    hparams['run_id'] = run_id
    hparams['run_dir'] = os.path.join(args.rootdir, run_id)
    test_agent = CutDQNAgent(hparams=hparams, use_gpu=args.use_gpu, gpu_id=args.gpu_id)
    test_agent.test()