#!/bin/bash -l
#SBATCH --account=iwi5
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --job-name=schmuck_finetune
 
# Use SLURM_JOB_ID for log folder - %j expands to job ID
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/dots/%j/schmuck_finetune.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/dots/%j/schmuck_finetune.err

# Create log directory BEFORE job execution
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/QWEN/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
 
# Enhanced environment configuration for CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
 
# Set HTTP and HTTPS proxies if required
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
 
# Load modules
module load cuda/11.8.0
module load python/3.12-conda
 
# Activate conda environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/qwen_vision
 
# Make sure your env's lib/ is prioritized
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
 
# Ignore any ~/.local site-packages
export PYTHONNOUSERSITE=1
 
# Point Python at the DTrOCR source directory
export PYTHONPATH=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/Dtrocr/DTrOCR:$PYTHONPATH
 
# Set CUDA memory allocation configuration
export CUDA_VISIBLE_DEVICES=0
 
echo "==================================="
echo "TrOCR Training Job Configuration:"
echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Log Directory: $LOG_DIR"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "Job started at: $(date)"
echo "==================================="
 
# Navigate to script directory and run
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/dots/finetune
python3 schmuck.py
 