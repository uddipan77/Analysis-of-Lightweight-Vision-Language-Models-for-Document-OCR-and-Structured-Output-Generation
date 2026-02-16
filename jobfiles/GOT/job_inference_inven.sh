#!/bin/bash -l
#SBATCH --account=iwi5
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=02:00:00
#SBATCH --job-name=got_inventory_test_v2
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/GOT/%j/test.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/GOT/%j/test.err

LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/GOT/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

module load cuda/12.8.0
module load python/3.12-conda

export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/got_ocr

CHECKPOINT_DIR="/home/vault/iwi5/iwi5298h/models_image_text/got/inven_v2/v0-20251016-224636/checkpoint-130"

echo "Starting inference V2 at: $(date)"
echo "Using checkpoint: $CHECKPOINT_DIR"

cd /home/woody/iwi5/iwi5298h/ms-swift

CUDA_VISIBLE_DEVICES=0 swift infer \
--ckpt_dir "$CHECKPOINT_DIR" \
--val_dataset /home/woody/iwi5/iwi5298h/json_inven_got/test_messages.jsonl \
--max_new_tokens 2048 \
--temperature 0.1 \
--top_p 0.9

echo "Inference completed at: $(date)"
echo "Results saved to: ${CHECKPOINT_DIR}/infer_result/"
