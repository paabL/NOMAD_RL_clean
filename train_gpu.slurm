#!/bin/bash
#SBATCH --account=def-kuzhang_gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000M
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --job-name=nomad_rc5_gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load python/3.11
source ~/envs/nomad_rl/bin/activate

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

CONFIG=${1:-configs/rc5_rorqual.json}

echo "hostname: $(hostname)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "PWD=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "config: $CONFIG"

nvidia-smi

python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"

srun python -u -m NOMAD_RC5.training "$CONFIG"
