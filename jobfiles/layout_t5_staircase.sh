#!/bin/bash -l
#
# SLURM Job Script for LayoutLMv3+T5 Staircase Inference/Training
#
# ===============================
# SLURM Directives
# ===============================
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1               # Request 1 NVIDIA V100 GPU
#SBATCH --partition=v100                # Specify the GPU partition
#SBATCH --time=15:00:00                 # Set max runtime
#SBATCH --export=NONE                   
#SBATCH --job-name=T5_Lay_Staircase     
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/T5_Layoutlmv3/%j/staircase2.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/T5_Layoutlmv3/%j/staircase2.err

# ===============================
# CUDA Environment Configuration
# ===============================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# Set proxies only if needed
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

unset SLURM_EXPORT_ENV

# ===============================
# Load Modules and Activate Conda
# ===============================
module load python/3.12-conda
module load cuda/11.8.0
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/uddipan_thesis

# ===============================
# Create log directory BEFORE job execution
# ===============================
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/T5_Layoutlmv3/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# ===============================
# Move to Script Directory
# ===============================
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/T5_LayoutLMV3/finetune_with_image_text/

# ===============================
# Run the Updated Python Script (training + test prediction)
# ===============================
python3 staircase2.py \
    --train_jsonl /home/woody/iwi5/iwi5298h/json_staircase/train.jsonl \
    --val_jsonl   /home/woody/iwi5/iwi5298h/json_staircase/val.jsonl \
    --test_jsonl  /home/woody/iwi5/iwi5298h/json_staircase/test.jsonl \
    --image_dir   /home/woody/iwi5/iwi5298h/staircase_images \
    --output_root_dir /home/vault/iwi5/iwi5298h/models_image_text/layoutlmv3_t5/staircase \
    --epochs 50 \
    --batch_size 2 \
    --lr 1e-5 \
    --max_length 512 \
    --augmentation_factor 2

# ===============================
# End of script
# ===============================
