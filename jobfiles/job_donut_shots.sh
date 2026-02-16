#!/bin/bash -l
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=02:30:00
#SBATCH --job-name=donutbase_zero_schmuck
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/%j/zero_shot_base_schmuck.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/%j/zero_shot_base_schmuck.err
# Create log directory

LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/${SLURM_JOB_ID}"
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
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/DONUT/inference/
python3 zero_shot_base_schmuck.py
echo "DONUT zero-shot inference completed at: $(date)"
