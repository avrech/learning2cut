#!/bin/bash
#SBATCH --time=00:5:00
#SBATCH --account=def-alodi
#SBATCH --output=graham-output/%j.out
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=16

#SBATCH --mail-user=avrech@campus.tecnion.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python run_adaptive_policy_experiment.py