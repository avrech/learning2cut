#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=def-alodi
#SBATCH --output=ccmab-overfit-%j.out
#SBATCH --mem=120G
#SBATCH --cpus-per-task=19
#SBATCH --job-name=ccmab-overfit-%j
#SBATCH --gres=gpu:1
srun python run_scip_tuning_dqn.py \
  --configfile configs/scip_tuning_ccmab_maxcut_overfit_40_50.yaml \
  --rootdir $SCRATCH/learning2cut/results/scip_tuning_ccmab_maxcut_overfit_40_50 \
  --datadir $SCRATCH/learning2cut/data \
  --data_config ../../data/maxcut_data_config.yaml \
  --problem MAXCUT \
  --tags tuning maxcut ccmab overfit \
  --use-gpu \
  --num_workers 15 \
  --wandb_offline \
  --eps_decay 500 \
  --eps_end 0.1
