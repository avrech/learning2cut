""" DQN
In this experiment cycle inequalities are added on the fly,
and DQN agent select which cuts to apply.
The model optimizes for the dualbound integral.
"""
import os
from agents.cut_dqn_agent import CutDQNAgent
import wandb


if __name__ == '__main__':
    import argparse
    import yaml
    from experiments.dqn.default_parser import parser, get_hparams
    # parser = argparse.ArgumentParser()
    parser.add_argument('--mixed-debug', action='store_true',
                        help='set for mixed python/c debugging')

    args = parser.parse_args()

    # enable python+C debugging with vscode
    if args.mixed_debug:
        import ptvsd
        port = 3000
        # ptvsd.enable_attach(secret='my_secret', address =('127.0.0.1', port))
        ptvsd.enable_attach(address=('127.0.0.1', port))
        ptvsd.wait_for_attach()

    # get hparams for all modules
    hparams = get_hparams(args)

    # set cuda debug mode
    if hparams.get('debug_cuda', False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # # set logdir according to hparams
    # relative_logdir = f"lr_{hparams['lr']}-nstep_{hparams['nstep_learning']}-credit_{hparams['credit_assignment']}-gamma_{hparams['gamma']}-obj_{hparams['dqn_objective']}"
    # hparams['logdir'] = os.path.join(hparams['logdir'], relative_logdir)
    run_id = args.run_id if args.resume else wandb.util.generate_id()
    hparams['run_id'] = run_id
    hparams['run_dir'] = os.path.join(args.rootdir, run_id)
    if not os.path.exists(hparams['run_dir']):
        os.makedirs(hparams['run_dir'])

    # todo wandb
    wandb_config = hparams.copy()
    wandb_config.pop('datasets')
    wandb.init(resume=args.resume,
               id=run_id,
               project=args.project,
               config=wandb_config,
               tags=args.tags
               )

    dqn_single_thread = CutDQNAgent(hparams=hparams, use_gpu=args.use_gpu, gpu_id=args.gpu_id)
    dqn_single_thread.train()
