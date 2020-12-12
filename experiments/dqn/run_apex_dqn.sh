#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=def-alodi
#SBATCH --output=apex-dqn-%j.out
#SBATCH --mem=64G
#SBATCH --cpus-per-task=13
#SBATCH --job-name=apex-dqn-%j
#SBATCH --gres=gpu:1
#SBATCH --mail-user=avrech@campus.tecnion.ac.il
python run_apex_dqn.py \
  --configfile configs/exp3-demoLossOnly.yaml \
  --rootdir $SCRATCH/dqn/results/exp3 \
  --datadir $SCRATCH/dqn/data \
  --tags exp3 \
  --use-gpu \
  --num_workers 10
