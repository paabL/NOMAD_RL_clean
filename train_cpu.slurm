#!/bin/bash
#SBATCH --account=def-kuzhang
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=nomad_rc5_cpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load python/3.11
source ~/envs/nomad_rl/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "hostname: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

python -c "import os, platform; print('python ok'); print('host:', platform.node()); print('cpus:', os.cpu_count())"

python -u NOMAD_RC5/training.py