#!/bin/bash -l
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=20:00:00
#SBATCH --export=NONE
#SBATCH --job-name=smolvlm2_inven
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/SmolVLM2/%j/inven.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/SmolVLM2/%j/inven.err

# Environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
unset SLURM_EXPORT_ENV

module load python/3.12-conda
module load cuda/11.8.0
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/global_env1

# Create log directory
mkdir -p "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/SmolVLM2/${SLURM_JOB_ID}"

# Run training
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/smolvlm2/finetune_with_no_post_process/

echo "Starting training at: $(date)"
python3 inven.py
STATUS=$?

echo "Finished at: $(date)"
echo "Exit status: $STATUS"
