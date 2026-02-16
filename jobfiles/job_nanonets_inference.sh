#!/bin/bash

# ===============================
# SLURM Directives
# ===============================
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=03:00:00
#SBATCH --job-name=new_few_inven
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/NANONETS/%j/new_few_inven.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/NANONETS/%j/new_few_inven.err

# ===============================
# Environment Setup
# ===============================

# Create log directory for this job
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/NANONETS/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# Load necessary modules
module load python/3.12-conda
module load cuda/11.8.0

# Activate conda environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/nanonets

# Navigate to script directory
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/nanonets/shots/
# ===============================
# Run the inference script
# ===============================

echo "Starting Nanonets-OCR-s inference..."
echo "Job ID: $SLURM_JOB_ID"
echo "Log Directory: $LOG_DIR"
echo "Job started at: $(date)"

python3 new_few_inven.py
echo "Job completed at: $(date)"
