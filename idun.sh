#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=eirre
#SBATCH --time=0-00:15:00     # 0 days and 15 minutes limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=1     # 2 CPU cores
#SBATCH --mem=5G              # 5 gigabytes memory
#SBATCH --output=log.txt    # Log file

module purge
module load Python/3.11.5-GCCcore-13.2.0
module list
poetry run python demo
