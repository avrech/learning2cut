#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=def-alodi
#SBATCH --output=apex-cut_selection_dqn-%j.out
#SBATCH --mem=120G
#SBATCH --cpus-per-task=19
#SBATCH --job-name=apex-cut_selection_dqn-%j
#SBATCH --gres=gpu:1
#SBATCH --mail-user=avrech@campus.tecnion.ac.il
python run_scip_tuning_dqn.py \
  --configfile configs/scip_tuning_mdp.yaml \
  --rootdir $SCRATCH/learning2cut/results/scip_tuning_mdp \
  --datadir $SCRATCH/learning2cut/data \
  --data_config configs/maxcut_data_config.yaml \
  --problem MAXCUT \
  --tags exp5 MAXCUT \
  --use-gpu \
  --num_workers 15 \
  --wandb_offline
