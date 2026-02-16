#!/bin/bash -l
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --job-name=multidataset_finetune
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/phi/%j/multidataset_finetune.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/phi/%j/multidataset_finetune.err

# Environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
unset SLURM_EXPORT_ENV

module load python/3.12-conda
module load cuda/11.8.0
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/phi_ocr

# Create log directory
mkdir -p "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/phi/${SLURM_JOB_ID}"

# Run training
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/phivision/multistage/
echo "Starting training at: $(date)"
python3 multidataset_finetune.py \
  --run_dir /home/vault/iwi5/iwi5298h/models_image_text/phi/general/run_20260213_215613_multidataset_bestCER
STATUS=$?

echo "Finished at: $(date)"
echo "Exit status: $STATUS"
