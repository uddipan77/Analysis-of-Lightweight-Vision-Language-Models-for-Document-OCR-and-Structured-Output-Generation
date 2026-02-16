#!/bin/bash -l
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --job-name=inven_updated_hpo
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/phi/%j/inven_updated_hpo.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/phi/%j/inven_updated_hpo.err

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
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/phivision/finetune/
echo "Starting training at: $(date)"
python3 inven_updated_hpo.py
STATUS=$?

echo "Finished at: $(date)"
echo "Exit status: $STATUS"
