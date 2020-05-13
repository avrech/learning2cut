#!/bin/bash
#SBATCH --time=00:35:00
#SBATCH --account=def-alodi
#SBATCH --output=generate_examples_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --mail-user=avrech@campus.tecnion.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python run_experiment.py --experiment imitation --configfile imitation/imitation_config.yaml --logdir imitation/results --datadir imitation/data
