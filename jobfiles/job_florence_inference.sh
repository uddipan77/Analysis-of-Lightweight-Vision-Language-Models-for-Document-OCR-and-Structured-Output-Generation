#!/bin/bash

# ===============================
# SLURM Directives
# ===============================
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=24:00:00
#SBATCH --job-name=Florence2_schmuck_new
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

# Set the CUDA memory allocation configuration for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Load necessary modules
module load python/3.12-conda
module load cuda/11.8.0

# Activate conda environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/global_env1

# Navigate to script directory
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/florence2/finetune_with_image_text/

# ===============================
# Run the inference script
# ===============================

echo "Starting Florence 2 inference..."
echo "Job ID: $SLURM_JOB_ID"
echo "Log Directory: $LOG_DIR"
echo "Job started at: $(date)"

python3 schmuck_new.py

echo "Job completed at: $(date)"
