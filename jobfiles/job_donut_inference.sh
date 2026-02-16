#!/bin/bash -l
#
# SLURM Job Script for DONUT OCR Inference
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:rtx3080:1               # Request 1 NVIDIA V100 GPU
#SBATCH --partition=rtx3080                # Specify the GPU partition
#SBATCH --time=02:00:00                 # Max runtime
#SBATCH --export=NONE
#SBATCH --job-name=donut_infer          # Job name
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/infer_code1.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/infer_code1.err

# -----------------------
# Environment
# -----------------------

# Set HTTP and HTTPS proxies if required
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
unset SLURM_EXPORT_ENV

module load python/3.12-conda
module load cuda/11.8.0

conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/uddipan_thesis

# -----------------------
# Navigate to inference code
# -----------------------
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/DONUT/inference

# -----------------------
# Run inference
# -----------------------
python3 code1.py \
  --image_dir "/home/woody/iwi5/iwi5298h/inventory_images" \
  --model_id "naver-clova-ix/donut-base" \
  --output_path "/home/vault/iwi5/iwi5298h/donut/inferencedonut/predictions.jsonl" \
  --max_images 10

echo "Inference complete. Predictions saved to /home/vault/iwi5/iwi5298h/donut/inferencedonut/predictions.jsonl"
