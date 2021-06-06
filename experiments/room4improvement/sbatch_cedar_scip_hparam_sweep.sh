#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-alodi
#SBATCH --output=hyperopt-%j.out
#SBATCH --job-name=hyperopt
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32

srun python scip_hparam_sweep.py --datadir $SCRATCH/learning2cut/data --num_trials 1000 --rootdir $SCRATCH/learning2cut/tuning
