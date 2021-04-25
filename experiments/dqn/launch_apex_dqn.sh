#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=def-alodi
#SBATCH --output=apex-dqn-%j.out
#SBATCH --mem=120G
#SBATCH --cpus-per-task=19
#SBATCH --job-name=apex-dqn-%j
#SBATCH --gres=gpu:1
#SBATCH --mail-user=avrech@campus.tecnion.ac.il
python run_apex_dqn.py \
  --configfile configs/exp5.yaml \
  --rootdir $SCRATCH/dqn/results/exp5 \
  --datadir $SCRATCH/dqn/data \
  --data_config configs/mvc_data_config.yaml \
  --problem MVC \
  --tags exp5 MVC \
  --use-gpu \
  --num_workers 15 \
  --wandb_offline
