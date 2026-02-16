#!/bin/bash

# ===============================
# SLURM Directives
# ===============================
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=03:00:00
#SBATCH --job-name=internvl3_inference
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/INTERNVL3/%j/inference_1B.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/INTERNVL3/%j/inference_1B.err

# ===============================
# Environment Setup
# ===============================
# Set HTTP and HTTPS proxies if required
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# Create log directory for this job
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/INTERNVL3/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# Load necessary modules
module load python/3.12-conda
module load cuda/11.8.0

# Activate conda environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/uddipan_thesis

# Navigate to script directory
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/internvl3/inference

# ===============================
# Run the inference script
# ===============================

echo "Starting InternVL3-1B inference..."
echo "Job ID: $SLURM_JOB_ID"
echo "Log Directory: $LOG_DIR"
echo "Job started at: $(date)"

python3 inference_1B.py

echo "Job completed at: $(date)"
