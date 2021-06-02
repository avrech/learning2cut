#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-alodi
#SBATCH --output=/scratch/a/alodi/avrech/learning2cut/tuning/job.out
#SBATCH --job-name=hyperopt
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40

# load modules and activate virtualenv
module load NiaEnv/2018a
module load python
source $HOME/server_bashrc
source $HOME/venv/bin/activate

# generate dataset
srun python scip_hparam_tuning.py --datadir $SCRATCH/learning2cut/data --num_trials 1000



