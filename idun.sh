#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=eirre
#SBATCH --time=0-00:15:00     # 0 days and 15 minutes limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --gres=gpu:1          # 1 GPU
#SBATCH --output=log.txt      # Log file

module purge
module load Python/3.11.5-GCCcore-13.2.0
module list

pip install poetry
poetry install

poetry run wandb login

poetry run python demo
