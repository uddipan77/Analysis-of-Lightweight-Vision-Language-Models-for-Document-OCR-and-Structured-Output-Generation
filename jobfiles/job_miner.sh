#!/bin/bash -l
#
# SLURM Job Script for Miner Fine-tuning
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:a100:1                 # Request 1 NVIDIA V100 GPU
#SBATCH --partition=a100                  # Specify the GPU partition
#SBATCH --time=24:00:00                   # Maximum runtime of 5 hours
#SBATCH --export=NONE
#SBATCH --job-name=miner_inven_updated
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/miner/%j/inven_updated.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/miner/%j/inven_updated.err

# ===============================
# Environment Configuration
# ===============================

# Create log directory BEFORE job execution
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/miner/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
unset SLURM_EXPORT_ENV

module load python/3.12-conda
module load cuda/11.8.0

conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/mineruocr

# ===============================
# Navigate to Script Directory
# ===============================
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/miner/finetune/
# Run the Training Script
python3 inven_updated.py
# ===============================
