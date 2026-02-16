#!/bin/bash -l
#SBATCH --account=iwi5
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=24:00:00
#SBATCH --job-name=got_inventory_train_v3
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/GOT/%j/trainV3.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/GOT/%j/trainV3.err

LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/GOT/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

module load cuda/12.8.0
module load python/3.12-conda

export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/got_ocr

OUTPUT_DIR="/home/vault/iwi5/iwi5298h/models_image_text/got/inven_v3"
mkdir -p "$OUTPUT_DIR"

echo "Starting training V3 at: $(date)"

cd /home/woody/iwi5/iwi5298h/ms-swift

CUDA_VISIBLE_DEVICES=0 swift sft \
--model_type got_ocr2 \
--model stepfun-ai/GOT-OCR2_0 \
--dataset /home/woody/iwi5/iwi5298h/json_inven_got/train_messages.jsonl \
--val_dataset /home/woody/iwi5/iwi5298h/json_inven_got/val_messages.jsonl \
--output_dir "$OUTPUT_DIR" \
--num_train_epochs 30 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--learning_rate 2e-5 \
--lora_rank 128 \
--lora_alpha 256 \
--save_steps 100 \
--eval_steps 100 \
--logging_steps 10 \
--save_total_limit 3 \
--max_length 4096 \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--dataloader_num_workers 4

echo "Training completed at: $(date)"
