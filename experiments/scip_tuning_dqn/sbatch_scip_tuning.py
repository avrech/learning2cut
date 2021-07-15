# submit automatically full node jobs to niagara/graham according to specified hparams
import os
from argparse import ArgumentParser
from itertools import product
import time
parser = ArgumentParser()
parser.add_argument('--cluster', type=str, default='niagara', help='graham | niagara')
parser.add_argument('--hours', type=str, default='6', help='0<hours<24')
parser.add_argument('--tag', type=str, default='v1', help='experiment tag for wandb')
parser.add_argument('--gpu', action='store_true', help='use gpu')
parser.add_argument('--test', action='store_true', help='test run')
parser.add_argument('--run_ids', type=str, nargs='+', default=[], help='run_ids to test')
args = parser.parse_args()
assert 0 < int(args.hours) < 24

if args.cluster == 'niagara':
    outfiles_dir = f"{os.environ['SCRATCH']}/learning2cut/scip_tuning/results/{args.tag}/outfiles/"
elif args.cluster == 'graham':
    outfiles_dir = f'{args.tag}_outfiles'
if not os.path.exists(outfiles_dir):
    os.makedirs(outfiles_dir)


def submit_job(config):
    sbatch_file = f'sbatch_{args.cluster}_{"_".join(k + str(v) for k, v in config.items())}.sh'
    with open(sbatch_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --time={args.hours.zfill(2)}:00:00\n")
        fh.writelines(f"#SBATCH --account=def-alodi\n")
        fh.writelines(
            f"#SBATCH --job-name={config.get('problem', 'MAXCUT')}-{config.get('scip_env', 'tuning_ccmab')}\n")
        fh.writelines(f"#SBATCH --nodes=1\n")
        fh.writelines(f"#SBATCH --mem=0\n")
        fh.writelines(f"#SBATCH --output={outfiles_dir}/{sbatch_file.split('.')[0]}-%j.out\n")
        if args.cluster == 'niagara':
            # fh.writelines(f"#SBATCH --output={os.environ['SCRATCH']}/learning2cut/scip_tuning/results/{args.tag}/{sbatch_file.split('.')[0]}-%j.out\n")  #/scratch/a/alodi/avrech/learning2cut/scip_tuning/results/{args.tag}/{sbatch_file.split('.')[0]}-%j.out\n")
            # fh.writelines(f"#SBATCH --output={args.tag}_outfiles/{sbatch_file.split('.')[0]}-%j.out\n")
            fh.writelines(f"#SBATCH --cpus-per-task=40\n")
            fh.writelines(f"#SBATCH --ntasks-per-node=1\n")
            # load modules and activate virtualenv
            fh.writelines(f"module load NiaEnv/2018a\n")
            fh.writelines(f"module load python\n")
            fh.writelines(f"source $HOME/server_bashrc\n")
            fh.writelines(f"source $HOME/venv/bin/activate\n")

        elif args.cluster == 'graham':
            # fh.writelines(f"#SBATCH --output={args.tag}_outfiles/{sbatch_file.split('.')[0]}-%j.out\n")
            fh.writelines(f"#SBATCH --cpus-per-task=32\n")
            fh.writelines(f"#SBATCH --ntasks-per-node=1\n")
            if args.gpu:
                fh.writelines(f"# SBATCH --gres=gpu:1\n")
        else:
            raise ValueError
        # command
        fh.writelines(f"srun python run_scip_tuning_dqn.py ")
        fh.writelines(f"  --configfile configs/scip_tuning_dqn.yaml ")
        fh.writelines(f"  --rootdir $SCRATCH/learning2cut/scip_tuning/results/{args.tag} ")
        fh.writelines(f"  --datadir $SCRATCH/learning2cut/data ")
        fh.writelines(f"  --data_config ../../data/{config.get('problem', 'MAXCUT').lower()}_data_config.yaml ")
        fh.writelines(f"  --tags {args.tag} ")
        if config.get('overfit', False):
            if config['problem'] == 'MAXCUT':
                fh.writelines(f"  --overfit validset_40_50 validset_60_70 ")
            else:
                fh.writelines(f"  --overfit validset_100_110 validset_150_160 ")

        if args.cluster == 'niagara':
            fh.writelines(f"  --num_workers 56 ")
        else:
            fh.writelines(f"  --num_workers 28 ")
        fh.writelines(f"  --wandb_offline True ")
        fh.writelines(f"  --eps_decay {250 if 'ccmab' in config.get('scip_env', 'tuning_ccmab') else 8000} ")
        fh.writelines(f"  --eps_end 0.1 ")
        fh.writelines(f"  --replay_buffer_capacity 20000 ")
        fh.writelines(f"  --local_buffer_size 10 ")
        fh.writelines(f"  --replay_buffer_minimum_size 1000 ")
        for k, v in config.items():
            fh.writelines(f"  --{k} {v}")
        # fh.writelines(f"  --problem {config['problem']} ")
        # fh.writelines(f"  --scip_env {scip_env} ")
        # fh.writelines(f"  --encoder_lp_conv_layers {encoder_lp_conv_layers} ")
        # fh.writelines(f"  --seed {seed} ")
        if args.test:
            fh.writelines(f"  --test \n")

    print(f'submitting {sbatch_file}')
    os.system(f'sbatch {sbatch_file}')
    time.sleep(0.5)


if args.test:
    for run_id in args.run_ids:
        submit_job({'run_id': run_id})
    exit(0)

search_space_mdp = {
    'problem': ['MAXCUT'],  #, 'MVC'],
    'scip_env': ['tuning_mdp'], #'tuning_ccmab'],
    'reward_func': ['db_aucXslope'],  #, 'db_auc'],  # 'db_slopeXdiff',
    'encoder_lp_conv_layers': [2],
    'conditional_q_heads': [True, False],
    'fix_training_scip_seed': [223],
    'seed': [11, 21, 31, 41, 51]
}

search_space_ccmab = {
    'problem': ['MAXCUT'],  #, 'MVC'],
    'scip_env': ['tuning_ccmab'], #'tuning_ccmab'],
    'norm_reward': [True, False],  # 'db_aucXslope',
    'square_reward': [True],
    'encoder_lp_conv_layers': [2],
    'conditional_q_heads': [True, False],
    'fix_training_scip_seed': [223],
    'seed': [11, 21, 31, 41, 51]
}
for search_space in [search_space_mdp]: #, search_space_ccmab]:
    cfgs = list(product(*search_space.values()))
    for cfg in cfgs:
        config = {k : str(v) for k, v in zip(search_space.keys(), cfg)}
        submit_job(config)
