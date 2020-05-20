#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=def-alodi
#SBATCH --output=dqn0-%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=dqn0-%j
#SBATCH --gres=gpu:1
#SBATCH --mail-user=avrech@campus.tecnion.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

nvidia-smi

# load modules and activate virtualenv

# run the experiment

python experiment.py --configfile experiment_config.yaml --logdir $SCRATCH/dqn/results/lr0.001-gamma0.99-nstep1 --datadir $SCRATCH/dqn/data --resume-training
