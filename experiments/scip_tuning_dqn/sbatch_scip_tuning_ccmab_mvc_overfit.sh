#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=def-alodi
#SBATCH --output=apex-cut_selection_dqn-%j.out
#SBATCH --mem=120G
#SBATCH --cpus-per-task=19
#SBATCH --job-name=apex-cut_selection_dqn-%j
#SBATCH --gres=gpu:1
#SBATCH --mail-user=avrech@campus.tecnion.ac.il
python run_scip_tuning_dqn.py \
  --configfile configs/scip_tuning_ccmab_maxcut_overfit_40_50.yaml \
  --rootdir $SCRATCH/learning2cut/results/scip_tuning_ccmab_mvc_overfit \
  --datadir $SCRATCH/learning2cut/data \
  --data_config ../../data/mvc_data_config.yaml \
  --tags tuning maxcut ccmab overfit \
  --use-gpu \
  --num_workers 15 \
  --wandb_offline \
  --problem MVC \
  --overfit validset_100_110 validset_150_160 validset_200_210
