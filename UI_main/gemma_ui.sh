#!/bin/bash -l
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --job-name=gemma_multi_ui
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/UI/%j/gemma_multi_ui.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/UI/%j/gemma_multi_ui.err

# ─────────────────────────────────────────────────────────────────────
#  Gemma-3 4B Multi-Dataset OCR UI  (single model, Gradio + FastAPI)
#  Port 8000
# ─────────────────────────────────────────────────────────────────────

LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/UI/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# ── Environment ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=localhost,127.0.0.1,::1

unset SLURM_EXPORT_ENV

module load python/3.12-conda
module load cuda/11.8.0
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/gemma_qwen_ui

# Ensure dependencies

# ── Run ──
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/UI_main/

NODE=$(hostname)

echo "=============================================="
echo "  Gemma-3 Multi-Dataset OCR UI"
echo "  Node:     ${NODE}"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "  GPU:      ${CUDA_VISIBLE_DEVICES:-unknown}"
echo "  Started:  $(date)"
echo "=============================================="
echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║  To access the UI, create an SSH tunnel:        ║"
echo "  ║                                                  ║"
echo "  ║  ssh -L 8000:${NODE}:8000 <user>@<login-node>   ║"
echo "  ║                                                  ║"
echo "  ║  Then open http://localhost:8000/ui               ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""

python3 gemma_ui.py
STATUS=$?

echo "Finished at: $(date)"
echo "Exit status: $STATUS"
