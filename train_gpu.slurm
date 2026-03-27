#!/bin/bash
#SBATCH --account=def-toncompte
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --job-name=nomad_rc5_gpu
#SBATCH --output=slurm-%j.out

module load python/3.11
source ~/envs/nomad_rl/bin/activate

cd ~/NOMAD_RL_clean

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count())"
python NOMAD_RC5/training_gpu.py
