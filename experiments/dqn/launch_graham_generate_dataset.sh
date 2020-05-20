#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=def-alodi
#SBATCH --output=gen-trainset-%j.out
#SBATCH --job-name=generate_dataset-%j
#SBATCH --mail-user=avrech@campus.tecnion.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# load modules and activate virtualenv

#module load python\n')
#source $HOME/server_bashrc\n')
#source $HOME/venv/bin/activate\n')

# generate dataset
srun python generate_dataset.py --configfile trainset_config.yaml --datadir $SCRATCH/dqn/data --mp --nworkers 32



