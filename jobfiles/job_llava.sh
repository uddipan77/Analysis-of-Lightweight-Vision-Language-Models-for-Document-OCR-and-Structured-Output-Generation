#!/bin/bash -l
#
# SLURM Job Script for LLAVA Fine-tuning
#
# ===============================
# SLURM Directives
# ===============================
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --job-name=llava_finetune
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/LLAVA/%j/schmuck.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/LLAVA/%j/schmuck.err

# ===============================
# Environment Setup
# ===============================
# Create log directory for this job
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/LLAVA/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# Set proxies
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# Load modules
module load python/3.12-conda
module load cuda/11.8.0

# Activate conda environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/global_env1

# Navigate to script directory
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/LLAVA/fine_tune

# ===============================
# Configuration
# ===============================

echo "==================================="
echo "Starting LLaVA Fine-tuning"
echo "Timestamp: $(date)"
echo "==================================="

# Print system info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""


# ===============================
# Run Training
# ===============================

echo "Starting training..."
export RESUME_DIR="/home/vault/iwi5/iwi5298h/models_image_text/llava/schmuck/llava_schmuck_hf_20250924_203922"
python3 schmuck.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "==================================="
    echo "Training completed successfully!"
    echo "Timestamp: $(date)"
    echo "==================================="
else
    echo "==================================="
    echo "Training failed with exit code: $?"
    echo "Timestamp: $(date)"
    echo "==================================="
    exit 1
fi

# ===============================
# Cleanup and Final Info
# ===============================

echo "Final GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "Job completed at: $(date)"
