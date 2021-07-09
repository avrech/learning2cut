# submit automatically full node jobs to niagara/graham according to specified hparams
import os
from argparse import ArgumentParser
import time
parser = ArgumentParser()
parser.add_argument('--cluster', type=str, default='graham', help='graham | niagara')
parser.add_argument('--hours', type=str, default='6', help='0<hours<24')
parser.add_argument('--gpu', action='store_true', help='use gpu')
args = parser.parse_args()
assert 0 < int(args.hours) < 24

for problem in ['MAXCUT', 'MVC']:
    for scip_env in ['cut_selection_mdp']:
        for overfit in [True, False]:
            for square_reward in [True, False]:
                sbatch_file = f'sbatch_{args.cluster}_{problem}_{scip_env}_overfit{overfit}_square_reward{square_reward}.sh'
                with open(sbatch_file, 'w') as fh:
                    fh.writelines("#!/bin/bash\n")
                    fh.writelines(f"#SBATCH --time={args.hours.zfill(2)}:00:00\n")
                    fh.writelines(f"#SBATCH --account=def-alodi\n")
                    fh.writelines(f"#SBATCH --job-name=cut_sel\n")
                    if args.cluster == 'niagara':
                        fh.writelines(f"#SBATCH --mem=0\n")
                        fh.writelines(f"#SBATCH --nodes=1\n")
                        fh.writelines(f"#SBATCH --output=/scratch/a/alodi/avrech/learning2cut/tuning/{sbatch_file.split('.')[0]}-%j.out\n")
                        fh.writelines(f"#SBATCH --cpus-per-task=40\n")
                        fh.writelines(f"#SBATCH --ntasks-per-node=1\n")
                        # load modules and activate virtualenv
                        fh.writelines(f"module load NiaEnv/2018a\n")
                        fh.writelines(f"module load python\n")
                        fh.writelines(f"source $HOME/server_bashrc\n")
                        fh.writelines(f"source $HOME/venv/bin/activate\n")

                    elif args.cluster == 'graham':
                        fh.writelines(f"#SBATCH --output={sbatch_file.split('.')[0]}-%j.out\n")
                        if args.gpu:
                            fh.writelines(f"#SBATCH --gres=gpu:1\n")
                            fh.writelines(f"#SBATCH --mem=60G\n")
                            fh.writelines(f"#SBATCH --cpus-per-task=16\n")
                        else:
                            fh.writelines(f"#SBATCH --mem=0\n")
                            fh.writelines(f"#SBATCH --nodes=1\n")
                            fh.writelines(f"#SBATCH --cpus-per-task=32\n")
                            fh.writelines(f"#SBATCH --ntasks-per-node=1\n")
                    else:
                        raise ValueError
                    # command
                    fh.writelines(f"srun python run_cut_selection_dqn.py ")
                    fh.writelines(f"  --configfile configs/exp5.yaml ")
                    fh.writelines(f"  --rootdir $SCRATCH/learning2cut/cut_selection/results/ ")
                    fh.writelines(f"  --datadir $SCRATCH/learning2cut/data ")
                    fh.writelines(f"  --data_config ../../data/{problem.lower()}_data_config.yaml ")
                    fh.writelines(f"  --problem {problem} ")
                    fh.writelines(f"  --tags cut_selection v0 ")
                    if overfit:
                        if problem == 'MAXCUT':
                            fh.writelines(f"  --overfit validset_40_50 validset_60_70 ")
                        else:
                            fh.writelines(f"  --overfit validset_100_110 validset_150_160 ")
                    if args.cluster == 'niagara':
                        fh.writelines(f"  --num_workers 56 ")
                    else:
                        if args.gpu:
                            fh.writelines(f"  --num_workers 12 ")
                        else:
                            fh.writelines(f"  --num_workers 28 ")

                    fh.writelines(f"  --wandb_offline True ")
                    # fh.writelines(f"  --eps_decay 1000 ")
                    fh.writelines(f"  --eps_end 0.1 ")
                    fh.writelines(f"  --scip_env {scip_env} ")
                    fh.writelines(f"  --replay_buffer_capacity 9000 ")
                    fh.writelines(f"  --local_buffer_size 10 ")
                    fh.writelines(f"  --replay_buffer_minimum_size 1000 ")
                    # fh.writelines(f"  --conditional_q_heads {cond_q} ")
                    # fh.writelines(f"  --norm_reward {norm_reward} ")
                    fh.writelines(f"  --square_reward {square_reward} \n")

                print(f'submitting {sbatch_file}')
                os.system(f'sbatch {sbatch_file}')
                time.sleep(1)
