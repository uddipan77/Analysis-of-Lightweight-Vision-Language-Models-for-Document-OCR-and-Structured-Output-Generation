#!/bin/bash -l

#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --job-name=staircase_new_base


#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/%j/staircase_new_base.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/%j/staircase_new_base.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
unset SLURM_EXPORT_ENV

module load python/3.12-conda
module load cuda/11.8.0
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/qwen_vision

LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/DONUT/finetune_with_image_text/

echo "Job started at: $(date)"
python3 staircase_new.py
STATUS=$?
echo "Job completed at: $(date)"
exit $STATUS
