#!/bin/bash -l
#SBATCH --account=iwi5
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=inventory_updated_hpo
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/gemma/%j/inventory_updated_hpo.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/gemma/%j/inventory_updated_hpo.err

# Create log directory

LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/gemma/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# Load modules (same as your working Qwen setup)
module load cuda/11.8.0
module load python/3.12-conda


# Memory settings (same as your working Qwen)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# Activate environment (your Qwen environment that works!)
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/qwen_vision


# Run training
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/gemma3/finetune/
python3 inventory_updated_hpo.py
echo "Gemma-3 training completed at: $(date)"
