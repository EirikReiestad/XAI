#!/bin/sh
#SBATCH --account="ie-idi"
#SBATCH --partition=GPUQ
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name="multi-agent-rl"
#SBATCH --output=srun.out
#SBATCH --error=srun.err

if [ ! -f ./clean.sh ]; then
  echo "No clean script found"
else
  echo "Cleaning up"
  sh ./clean.sh
fi

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}" || exit 1
echo "Running from this directory: $(pwd)"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The jo was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Python/3.11.5-GCCcore-13.2.0
module list

pip install poetry
poetry install

poetry run wandb login

poetry run python -O demo

uname -a
