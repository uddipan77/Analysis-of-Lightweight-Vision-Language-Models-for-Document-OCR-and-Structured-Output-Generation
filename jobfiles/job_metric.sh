#!/bin/bash -l
#SBATCH --account=iwi5
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=00:10:00
#SBATCH --job-name=gemma_inven
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/metric/%j/one_donut_base.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/metric/%j/one_donut_base.err

# Create log directory

LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/metric/${SLURM_JOB_ID}"
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
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/global_env1


# Run training
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/metric
python3 one.py
echo "completed at: $(date)"
