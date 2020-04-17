#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-alodi
#SBATCH --output=graham-job-%j.out
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=64

#SBATCH --mail-user=avrech@campus.tecnion.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python run_adaptive_policy_experiment.py
