#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=def-alodi
#SBATCH --output=cutoff-%j.out
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --job-name=cutoff-%j
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=avrech@campus.tecnion.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load python
source $HOME/server_bashrc
source $HOME/venv/bin/activate
srun python run_experiment.py --logdir $SCRATCH/cutoff/results --configfile cutoff_config.yaml --datadir $SCRATCH/cutoff/data --mp ray
