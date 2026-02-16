#!/bin/bash

# ===============================
# SLURM Directives
# ===============================
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=15:00:00
#SBATCH --job-name=florence_schmuck_new
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/Florence2/%j/schmuck_new.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/Florence2/%j/schmuck_new.err

# ===============================
# Environment Setup
# ===============================
# Set HTTP and HTTPS proxies if required
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# Create log directory for this job
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/Florence2/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# Enhanced environment configuration for CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Load necessary modules
module load python/3.12-conda
module load cuda/11.8.0

# Activate conda environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/global_env1

# Make sure your env's lib/ is prioritized
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Ignore any ~/.local site-packages
export PYTHONNOUSERSITE=1

# Set CUDA memory allocation configuration
export CUDA_VISIBLE_DEVICES=0

# Navigate to script directory
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/florence2/finetune_with_image_text/

# ===============================
# Run the fine-tuning script
# ===============================

echo "========================================"
echo "Florence-2 OCR Fine-tuning Job Configuration:"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Log Directory: $LOG_DIR"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "Job started at: $(date)"
echo "========================================"

# Run the Florence-2 fine-tuning with appropriate arguments
python3 schmuck_new.py
    

echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"
