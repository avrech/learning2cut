# submit automatically full node jobs to niagara/graham according to specified hparams
import os
from argparse import ArgumentParser
import time
parser = ArgumentParser()
parser.add_argument('--cluster', type=str, default='niagara', help='graham | niagara')
parser.add_argument('--hours', type=str, default='6', help='0<hours<24')
parser.add_argument('--gpu', action='store_true', help='use gpu')
args = parser.parse_args()
assert 0 < int(args.hours) < 24

for problem in ['MAXCUT', 'MVC']:
    for scip_env in ['tuning_ccmab', 'tuning_mdp']:
        for overfit in [True, False]:
            for cond_q in [True, False]:
                for norm_reward in [True, False]:
                    for square_reward in [True, False]:
                        if 'mdp' in scip_env and norm_reward:
                            # not supported yet
                            continue
                        sbatch_file = f'sbatch_{args.cluster}_{problem}_{scip_env}_overfit{overfit}_cond_q{cond_q}_norm_reward{norm_reward}_square_reward{square_reward}.sh'
                        with open(sbatch_file, 'w') as fh:
                            fh.writelines("#!/bin/bash\n")
                            fh.writelines(f"#SBATCH --time={args.hours.zfill(2)}:00:00\n")
                            fh.writelines(f"#SBATCH --account=def-alodi\n")
                            fh.writelines(f"#SBATCH --job-name=ccmab-overfit\n")
                            fh.writelines(f"#SBATCH --mem=0\n")
                            fh.writelines(f"#SBATCH --nodes=1\n")
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
                                fh.writelines(f"#SBATCH --output={sbatch_file.split('.')[0]}-%j.out\n")
                                fh.writelines(f"#SBATCH --cpus-per-task=32\n")
                                fh.writelines(f"#SBATCH --ntasks-per-node=1\n")
                                if args.gpu:
                                    fh.writelines(f"# SBATCH --gres=gpu:1\n")
                            else:
                                raise ValueError
                            # command
                            fh.writelines(f"srun python run_scip_tuning_dqn.py \\ \n")
                            fh.writelines(f"  --configfile configs/scip_tuning_dqn.yaml \\ \n")
                            fh.writelines(f"  --rootdir $SCRATCH/learning2cut/scip_tuning/results/{sbatch_file.split('.')[0]} \\ \n")
                            fh.writelines(f"  --datadir $SCRATCH/learning2cut/data \\ \n")
                            fh.writelines(f"  --data_config ../../data/{problem.lower()}_data_config.yaml \\ \n")
                            fh.writelines(f"  --problem {problem} \\ \n")
                            fh.writelines(f"  --tags tuning v0 \\ \n")
                            if overfit:
                                if problem == 'MAXCUT':
                                    fh.writelines(f"  --overfit validset_40_50 validset_60_70 validset_90_100 \\ \n")
                                else:
                                    fh.writelines(f"  --overfit validset_100_110 validset_150_160 validset_200_210 \\ \n")
                            if args.cluster == 'niagara':
                                fh.writelines(f"  --num_workers 76 \\ \n")
                            else:
                                fh.writelines(f"  --num_workers 28 \\ \n")
                            fh.writelines(f"  --wandb_offline True \\ \n")
                            fh.writelines(f"  --eps_decay 200 \\ \n")
                            fh.writelines(f"  --eps_end 0.1 \\ \n")
                            fh.writelines(f"  --scip_env {scip_env} \\ \n")
                            fh.writelines(f"  --replay_buffer_capacity 10000 \\ \n")
                            fh.writelines(f"  --local_buffer_size 10 \\ \n")
                            fh.writelines(f"  --replay_buffer_minimum_size 1000 \\ \n")
                            fh.writelines(f"  --conditional_q_heads {cond_q} \\ \n")
                            fh.writelines(f"  --norm_reward {norm_reward} \\ \n")
                            fh.writelines(f"  --square_reward {square_reward} \n")

                        print(f'submitting {sbatch_file}')
                        os.system(f'sbatch {sbatch_file}')
                        time.sleep(1)
