#!/bin/bash -l
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --job-name=zero2_docling
 
# Use SLURM_JOB_ID for log folder - %j expands to job ID
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/docling/%j/zero_inven2.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/docling/%j/zero_inven2.err
 
# Create log directory BEFORE job execution (FIXED: correct path)
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/docling/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
 
# Enhanced environment configuration for CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
 
# Set HTTP and HTTPS proxies if required
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
 
# Load modules
module load cuda/11.8.0
module load python/3.12-conda
 
# Activate conda environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/global_env1
 
# Make sure your env's lib/ is prioritized
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
 
# Ignore any ~/.local site-packages
export PYTHONNOUSERSITE=1
 
# UPDATED: Remove TrOCR-specific PYTHONPATH (not needed for Docling)
# export PYTHONPATH=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/Dtrocr/DTrOCR:$PYTHONPATH
 
# Set CUDA memory allocation configuration
export CUDA_VISIBLE_DEVICES=0
 
# ADDED: Docling-specific environment variables
export HF_HOME=/home/vault/iwi5/iwi5298h/models/hf_cache  # Optional: control HF cache location
export TOKENIZERS_PARALLELISM=false  # Prevent tokenizer warnings
 
echo "=========================================="
echo "Granite Docling Training Job Configuration:"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Log Directory: $LOG_DIR"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "Script: /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/docling/few_zero/zero_inven2.py"
echo "Job started at: $(date)"
echo "=========================================="
 
# Navigate to script directory and run
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/docling/few_zero
python3 zero_inven2.py

# ADDED: Job completion info
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $?"
echo "=========================================="
