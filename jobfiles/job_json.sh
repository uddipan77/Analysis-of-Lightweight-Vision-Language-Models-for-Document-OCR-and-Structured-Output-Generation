#!/bin/bash -l
#
# SLURM Job Script for GOT-OCR2 Inference on inventory data
#
# ===============================
# SLURM Directives
# ===============================
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1               # Request 1 NVIDIA V100 GPU
#SBATCH --partition=v100                # Specify the GPU partition
#SBATCH --time=01:00:00                 # Maximum runtime of 24 hours
#SBATCH --export=NONE                   # Do not export current environment variables
#SBATCH --job-name=staircase                  # Job name
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/JSON/staircase.out  # Standard output log file
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/JSON/staircase.err   # Standard error log file

# ===============================
# Environment Configuration
# ===============================

# Set the CUDA memory allocation configuration to help avoid fragmentation issues.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Enable CUDA blocking for better error reporting
export CUDA_LAUNCH_BLOCKING=1

# Set HTTP and HTTPS proxies if required
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# Unset SLURM_EXPORT_ENV to prevent SLURM from exporting environment variables
unset SLURM_EXPORT_ENV

# Load the necessary modules
module load python/3.12-conda        # Load Python Anaconda module
module load cuda/11.8.0              # Load CUDA 11.8 for better V100 compatibility

# Activate the Conda environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/uddipan_thesis

# ===============================
# Fix PyTorch CUDA Version (if needed)
# ===============================

# ===============================
# Navigate to Script Directory
# ===============================
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/dataquality

# ===============================
# Execute the Python Script
# ===============================

python3 check_staircase2.py
