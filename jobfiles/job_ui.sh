#!/bin/bash -l
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --job-name=phi_stair_ui2
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/phi/%j/UI2.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/phi/%j/UI2.err

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# If your cluster sets HTTP(S)_PROXY globally, clear them so Gradio
# can talk to localhost without going through the proxy.
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

# Just in case anything still respects NO_PROXY:
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=localhost,127.0.0.1,::1

unset SLURM_EXPORT_ENV

module load python/3.12-conda
module load cuda/11.8.0
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/phi_ocr

# -----------------------------------------------------------------------------
# Logs
# -----------------------------------------------------------------------------
mkdir -p "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/phi/${SLURM_JOB_ID}"

# -----------------------------------------------------------------------------
# Run UI
# -----------------------------------------------------------------------------
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/UI/

echo "Starting UI at: $(date)"
python3 stair_ui_phi2.py
STATUS=$?

echo "Finished at: $(date)"
echo "Exit status: $STATUS"
