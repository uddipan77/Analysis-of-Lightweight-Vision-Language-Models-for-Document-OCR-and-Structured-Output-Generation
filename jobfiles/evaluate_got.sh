#!/bin/bash -l
#SBATCH --account=iwi5
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=4:00:00
#SBATCH --export=NONE
#SBATCH --job-name=GOT_eval
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/finetune_GOT/logs/eval_swift.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/finetune_GOT/logs/eval_swift.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
unset SLURM_EXPORT_ENV

module load python/3.12-conda
module load cuda/11.8.0
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/uddipan_thesis

# Go to ms-swift directory
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/swift_got_finetune/ms-swift

swift infer \
  --model_type got_ocr2 \
  --model stepfun-ai/GOT-OCR2_0 \
  --ckpt_dir /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/swift_got_finetune/ms-swift/output/GOT-OCR2_0/v4-20250531-224249/checkpoint-42 \
  --dataset /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/finetune_GOT/data/test.jsonl \
  --result_path /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/finetune_GOT/eval_results/predictions.jsonl \
  --max_new_tokens 512
