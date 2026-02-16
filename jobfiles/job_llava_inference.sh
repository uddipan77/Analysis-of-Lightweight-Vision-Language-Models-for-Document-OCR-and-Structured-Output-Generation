#!/bin/bash

# ===============================
# SLURM Directives
# ===============================
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=02:00:00
#SBATCH --job-name=llava_inference
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/LLAVA/%j/llava_inference.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/LLAVA/%j/llava_inference.err

# ===============================
# Environment Setup
# ===============================
# Set HTTP and HTTPS proxies if required
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# Create log directory for this job
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/LLAVA/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
# Enhanced environment configuration for CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512


# Load necessary modules
module load python/3.12-conda
module load cuda/11.8.0

# Activate conda environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/uddipan_thesis

# Navigate to script directory
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/LLAVA/inference

# ===============================
# Run the inference script
# ===============================

echo "Starting LLaVA inference..."
echo "Job ID: $SLURM_JOB_ID"
echo "Log Directory: $LOG_DIR"
echo "Job started at: $(date)"

python3 code2.py

echo "Job completed at: $(date)"
