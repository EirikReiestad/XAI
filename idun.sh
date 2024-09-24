#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=eirre
#SBATCH --time=0-24:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --output=log.txt

module purge
module load Python/3.11.5-GCCcore-13.2.0
module list

pip install poetry
poetry install

poetry run wandb login

poetry run python -O demo
