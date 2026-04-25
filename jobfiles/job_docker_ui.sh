#!/bin/bash -l
#SBATCH --account=iwi5
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=a100
#SBATCH --time=04:00:00
#SBATCH --job-name=gemma_ocr_ui
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/gemma/%j/gemma_ocr_ui.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/gemma/%j/gemma_ocr_ui.err

# ── Create log directory ──
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/gemma/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# ── Load modules ──
module load cuda/11.8.0
module load python/3.12-conda

# ── Environment ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# ── Activate conda env (reuse your existing one) ──
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/qwen_vision

# ── Print access info ──
NODE=$(hostname)
PORT=8000
echo "════════════════════════════════════════════════════════════"
echo "  Gemma-3 OCR UI  (no container, direct Python)"
echo "  Node:  $NODE"
echo "  Port:  $PORT"
echo ""
echo "  To access from your local machine, run in a NEW terminal:"
echo ""
echo "    ssh -L ${PORT}:${NODE}:${PORT} iwi5298h@hpc-login.fau.de"
echo ""
echo "  Then open: http://localhost:${PORT}/ui"
echo "════════════════════════════════════════════════════════════"

# ── Run the UI directly (uses HPC model paths baked into gemma_ui.py) ──
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/UI_main/
python3 gemma_ui.py

echo "$(date) | UI server exited."
