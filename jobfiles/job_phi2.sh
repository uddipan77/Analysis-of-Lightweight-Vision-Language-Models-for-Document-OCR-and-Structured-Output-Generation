#!/bin/bash -l
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --job-name=phi_multidataset
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/phi/%j/multidataset.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/phi/%j/multidataset.err

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

# Existing run directory to resume from
RUN_DIR="/home/vault/iwi5/iwi5298h/models_image_text/phi/general/run_20251212_133535_multi_dataset"

# Optional: clean problematic RNG state files so torch.load doesn't fail on resume
if [ -d "${RUN_DIR}/stage2" ]; then
    echo "Found stage2 directory under RUN_DIR: ${RUN_DIR}/stage2"
    echo "Removing any rng_state.pth files before resuming..."
    find "${RUN_DIR}/stage2" -name "rng_state.pth" -print -delete || true
else
    echo "No stage2 directory found in RUN_DIR (${RUN_DIR}), nothing to clean."
fi

# Run training
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/phivision/multistage
echo "Starting training at: $(date)"
python3 multidataset.py --run_dir "$RUN_DIR"
STATUS=$?

echo "Finished at: $(date)"
echo "Exit status: $STATUS"
