#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-alodi
#SBATCH --output=imitation-genexamples-%j.out
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --job-name=imitation-genexamples-%j
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=avrech@campus.tecnion.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# load modules and activate virtualenv

# run the experiment

srun python run_experiment.py --experiment imitation --configfile imitation/datagen_config.yaml --logdir imitation/results/test --datadir imitation/data --mp mp
# srun python run_adaptive_policy_experiment.py
