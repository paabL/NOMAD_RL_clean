#!/bin/bash
#SBATCH --account=def-kuzhang
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --job-name=nomad_rc5_cpu_subproc
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load python/3.11
source ~/envs/nomad_rl/bin/activate

# Très important avec SubprocVecEnv:
# on veut que le parallélisme vienne des processes d'envs,
# pas de threads internes BLAS/OpenMP.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

python - <<'PY'
import os, platform
print("python ok")
print("host:", platform.node())
print("cpus visible:", os.cpu_count())
PY

python -u NOMAD_RC5/training.py