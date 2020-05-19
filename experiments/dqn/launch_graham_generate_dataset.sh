#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=def-alodi
#SBATCH --output=dqn0-%j.out
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=generate_dataset-%j
#SBATCH --mail-user=avrech@campus.tecnion.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

nvidia-smi

# load modules and activate virtualenv

# run the experiment

srun python generate_dataset.py --logdir $SCRATCH/dqn/results --datadir $SCRATCH/dqn/data --workerid --nworkers 10