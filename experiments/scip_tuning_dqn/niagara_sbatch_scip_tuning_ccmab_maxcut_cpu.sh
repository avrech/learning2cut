#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=def-alodi
#SBATCH --output=/scratch/a/alodi/avrech/learning2cut/tuning/ccmab-%j.out
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=ccmab

# load modules and activate virtualenv
module load NiaEnv/2018a
module load python
source $HOME/server_bashrc
source $HOME/venv/bin/activate

srun python run_scip_tuning_dqn.py \
  --configfile configs/scip_tuning_ccmab.yaml \
  --rootdir $SCRATCH/learning2cut/tuning/ccmab/results/maxcut \
  --datadir $SCRATCH/learning2cut/data \
  --data_config ../../data/maxcut_data_config.yaml \
  --problem MAXCUT \
  --tags tuning maxcut ccmab \
  --num_workers 76 \
  --wandb_offline \
  --eps_decay 200 \
  --eps_end 0.1 \
  --replay_buffer_minimum_size 300 \
  --local_buffer_size 10 \
  --conditional_q_heads True