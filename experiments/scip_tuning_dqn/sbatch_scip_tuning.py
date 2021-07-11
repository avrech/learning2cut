# submit automatically full node jobs to niagara/graham according to specified hparams
import os
from argparse import ArgumentParser
import time
parser = ArgumentParser()
parser.add_argument('--cluster', type=str, default='niagara', help='graham | niagara')
parser.add_argument('--hours', type=str, default='6', help='0<hours<24')
parser.add_argument('--tag', type=str, default='v1', help='experiment tag for wandb')
parser.add_argument('--gpu', action='store_true', help='use gpu')
args = parser.parse_args()
assert 0 < int(args.hours) < 24
if args.cluster == 'graham' and not os.path.exists(f'{args.tag}_outfiles'):
    os.makedirs(f'{args.tag}_outfiles')
for problem in ['MAXCUT', 'MVC']:
    for scip_env in ['tuning_ccmab', 'tuning_mdp']:
        for scip_seed in [0, 223]:
            for encoder_lp_conv_layers in [1, 2]:
                for seed in [11, 21, 31]:
                    sbatch_file = f'sbatch_{args.cluster}_{problem}_{scip_env}_seed{seed}_scip_seed{scip_seed}_nlayers{encoder_lp_conv_layers}.sh'
                    with open(sbatch_file, 'w') as fh:
                        fh.writelines("#!/bin/bash\n")
                        fh.writelines(f"#SBATCH --time={args.hours.zfill(2)}:00:00\n")
                        fh.writelines(f"#SBATCH --account=def-alodi\n")
                        fh.writelines(f"#SBATCH --job-name=ccmab-overfit\n")
                        fh.writelines(f"#SBATCH --nodes=1\n")
                        fh.writelines(f"#SBATCH --mem=0\n")
                        if args.cluster == 'niagara':
                            fh.writelines(f"#SBATCH --output=/scratch/a/alodi/avrech/learning2cut/tuning/{sbatch_file.split('.')[0]}-%j.out\n")
                            fh.writelines(f"#SBATCH --cpus-per-task=40\n")
                            fh.writelines(f"#SBATCH --ntasks-per-node=1\n")
                            # load modules and activate virtualenv
                            fh.writelines(f"module load NiaEnv/2018a\n")
                            fh.writelines(f"module load python\n")
                            fh.writelines(f"source $HOME/server_bashrc\n")
                            fh.writelines(f"source $HOME/venv/bin/activate\n")

                        elif args.cluster == 'graham':
                            fh.writelines(f"#SBATCH --output={args.tag}_outfiles/{sbatch_file.split('.')[0]}-%j.out\n")
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
                        fh.writelines(f"  --data_config ../../data/{problem.lower()}_data_config.yaml ")
                        fh.writelines(f"  --problem {problem} ")
                        fh.writelines(f"  --tags {args.tag} ")
                        if scip_seed:
                            if problem == 'MAXCUT':
                                fh.writelines(f"  --overfit validset_40_50 validset_60_70 ")
                            else:
                                fh.writelines(f"  --overfit validset_100_110 validset_150_160 ")
                        if args.cluster == 'niagara':
                            fh.writelines(f"  --num_workers 56 ")
                        else:
                            fh.writelines(f"  --num_workers 28 ")
                        fh.writelines(f"  --wandb_offline True ")
                        fh.writelines(f"  --eps_decay {300 if 'ccmab' in scip_env else 10000} ")
                        fh.writelines(f"  --eps_end 0.1 ")
                        fh.writelines(f"  --scip_env {scip_env} ")
                        fh.writelines(f"  --replay_buffer_capacity 10000 ")
                        fh.writelines(f"  --local_buffer_size 10 ")
                        fh.writelines(f"  --replay_buffer_minimum_size 1000 ")
                        fh.writelines(f"  --encoder_lp_conv_layers {encoder_lp_conv_layers} ")
                        fh.writelines(f"  --seed {seed} ")
                        fh.writelines(f"  --fix_training_scip_seed {scip_seed} \n")

                    print(f'submitting {sbatch_file}')
                    os.system(f'sbatch {sbatch_file}')
                    time.sleep(1)
