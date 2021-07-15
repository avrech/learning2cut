#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --account=def-alodi
#SBATCH --output=/scratch/a/alodi/avrech/learning2cut/tuning/test-{$1}%j.out
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=test-{$1}

# load modules and activate virtualenv
module load NiaEnv/2018a
module load python
source $HOME/server_bashrc
source $HOME/venv/bin/activate

srun python run_scip_tuning_dqn.py \
  --configfile configs/scip_tuning_ccmab.yaml \
  --rootdir $2 \
  --datadir $SCRATCH/learning2cut/data \
  --data_config ../../data/maxcut_data_config.yaml \
  --problem MAXCUT \
  --num_workers 76 \
  --wandb_offline \
  --test \
  --run_id $1