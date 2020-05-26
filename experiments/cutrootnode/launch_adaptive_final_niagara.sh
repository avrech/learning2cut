#!/bin/bash
#SBATCH --time=00:16:00
#SBATCH --account=def-alodi
#SBATCH --output=/scratch/a/alodi/avrech/cutrootnode/gsize100/adaptive-final-%j.out
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --mail-user=avrech@campus.tecnion.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
source $HOME/login_niagara_bashrc
cd cutrootnode
# srun python experiment.py --log_dir /scratch/a/alodi/avrech/cutrootnode/gsize100/results/adaptive_final --data_dir /gpfs/fs1/home/a/alodi/avrech/learning2cut/experiments/cutrootnode/data --starting_policies_abspath /scratch/a/alodi/avrech/cutrootnode/gsize100/adaptive/starting_policies.pkl

python experiment.py --log_dir $SCRATCH/cutrootnode/gsize100/results/adaptive-final --starting_policies_abspath $SCRATCH/cutrootnode/gsize100/adaptive/starting_policies.pkl

