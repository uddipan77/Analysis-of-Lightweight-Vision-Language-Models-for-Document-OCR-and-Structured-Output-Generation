#!/bin/bash -l
#SBATCH --account=iwi5
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=12:00:00
#SBATCH --job-name=got_inventory_train
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/GOT/%j/train.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/GOT/%j/train.err

# Create log directory
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/GOT/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# Load modules
module load cuda/12.8.0
module load python/3.12-conda

# Memory settings
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# Activate GOT-OCR environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/got_ocr

# Output directory
OUTPUT_DIR="/home/vault/iwi5/iwi5298h/models_image_text/got/inven"
mkdir -p "$OUTPUT_DIR"

echo "Starting training at: $(date)"
echo "Using CUDA version: $(nvcc --version | grep release)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Run fine-tuning
cd /home/woody/iwi5/iwi5298h/ms-swift

CUDA_VISIBLE_DEVICES=0 swift sft \
--model_type got_ocr2 \
--model stepfun-ai/GOT-OCR2_0 \
--dataset /home/woody/iwi5/iwi5298h/json_inven_got/train_converted.jsonl \
--val_dataset /home/woody/iwi5/iwi5298h/json_inven_got/val_converted.jsonl \
--output_dir "$OUTPUT_DIR" \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--learning_rate 2e-5 \
--save_steps 200 \
--eval_steps 200 \
--logging_steps 10 \
--save_total_limit 2 \
--max_length 4096 \
--warmup_ratio 0.05 \
--weight_decay 0.01 \
--dataloader_num_workers 4

echo "Training completed at: $(date)"
echo "Model saved to: $OUTPUT_DIR"
