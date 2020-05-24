#!/bin/bash
#SBATCH --time=00:16:00
#SBATCH --account=def-alodi
#SBATCH --output=/scratch/a/alodi/avrech/cutrootnode/gsize50/results/experts-%j.out
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40

#SBATCH --mail-user=avrech@campus.tecnion.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
source $HOME/login_niagara_bashrc
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd cutrootnode 

srun python run_experiment.py --logdir /scratch/a/alodi/avrech/cutrootnode/gsize50/results/experts --configfile experts_config.yaml

# python experiment.py --log_dir $SCRATCH/cutrootnode/gsize50/adaptive-final --starting_policies_abspath $SCRATCH/cutrootnode/gsize50/adaptive-final/starting_policies.pkl

