#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=16
srun python run_adaptive_policy_experiment.py