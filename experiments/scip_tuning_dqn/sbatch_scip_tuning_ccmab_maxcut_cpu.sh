#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=def-alodi
#SBATCH --output=ccmab-%j.out
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --cpus-per-task=32
#SBATCH --job-name=ccmab-%j
srun python run_scip_tuning_dqn.py \
  --configfile configs/scip_tuning_ccmab.yaml \
  --rootdir $SCRATCH/learning2cut/results/scip_tuning_ccmab_maxcut \
  --datadir $SCRATCH/learning2cut/data \
  --data_config ../../data/maxcut_data_config.yaml \
  --problem MAXCUT \
  --tags tuning maxcut ccmab \
  --num_workers 28 \
  --wandb_offline \
  --eps_decay 200 \
  --eps_end 0.1 \
  --replay_buffer_minimum_size 300 \
  --local_buffer_size 10 \
  --conditional_q_heads True