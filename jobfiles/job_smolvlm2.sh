#!/bin/bash -l
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=10:00:00
#SBATCH --job-name=smolvlm_few_inven
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/SmolVLM2/%j/few_inven.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/SmolVLM2/%j/few_inven.err
# Create log directory

LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/SmolVLM2/${SLURM_JOB_ID}"
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
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/smolvlm2/few_zero
python3 few_inven.py
echo "SmolVLM2 training completed at: $(date)"
