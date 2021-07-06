#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=def-alodi
#SBATCH --output=/scratch/a/alodi/avrech/learning2cut/tuning/ccmab-overfit-%j.out
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=ccmab-overfit

# load modules and activate virtualenv
module load NiaEnv/2018a
module load python
source $HOME/server_bashrc
source $HOME/venv/bin/activate

srun python run_scip_tuning_dqn.py \
  --configfile configs/scip_tuning_ccmab_maxcut_overfit_40_50.yaml \
  --rootdir $SCRATCH/learning2cut/tuning/ccmab/results/maxcut-overfit \
  --datadir $SCRATCH/learning2cut/data \
  --data_config ../../data/maxcut_data_config.yaml \
  --problem MAXCUT \
  --tags tuning maxcut ccmab overfit \
  --overfit validset_40_50 validset_60_70 validset_90_100 \
  --num_workers 36 \
  --wandb_offline \
  --eps_decay 500 \
  --eps_end 0.1 \
  --conditional_q_heads True
