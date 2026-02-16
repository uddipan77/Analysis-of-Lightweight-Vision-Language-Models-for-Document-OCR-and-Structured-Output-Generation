#!/bin/bash -l
#
# SLURM Job Script for Qwen Few-Shot Inference on Staircase Dataset
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --job-name=qwen_staircase_inference


# Use SLURM_JOB_ID for log folder - %j expands to job ID
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/QWEN/%j/staircase_inference.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/QWEN/%j/staircase_inference.err


# ===============================
# Environment Configuration
# ===============================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
unset SLURM_EXPORT_ENV


module load python/3.12-conda
module load cuda/11.8.0


conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/uddipan_thesis


# ===============================
# Create log directory BEFORE job execution
# ===============================
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/QWEN/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"


# ===============================
# Navigate to Script Directory
# ===============================
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/qwen/inference


# ===============================
# Parameter Setup
# ===============================
PYTHON_SCRIPT="qwen7b_fewshot_inven.py"
DATA_DIR="/home/woody/iwi5/iwi5298h/json_staircase"
IMAGE_DIR="/home/woody/iwi5/iwi5298h/staircase_images"
MODEL_DIR="/home/vault/iwi5/iwi5298h/models/qwen7b"
OUTPUT_DIR="/home/vault/iwi5/iwi5298h/models_image_text/qwen/staircase_inference"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)


echo "==================================="
echo "Qwen Staircase Inference Configuration:"
echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Log Directory: $LOG_DIR"
echo "Script: $PYTHON_SCRIPT"
echo "Data Directory: $DATA_DIR"
echo "Image Directory: $IMAGE_DIR"
echo "Model Directory: $MODEL_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""
echo "Few-shot examples:"
echo "- FMIS_FormblätterMielke_gefüllt (14).jpg"
echo "- FMIS_FormblätterMielke_gefüllt (16).jpg"
echo "- FMIS_FormblätterMielke_gefüllt (38).jpg"
echo "==================================="


# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"


# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found in current directory"
    echo "Current directory: $(pwd)"
    echo "Expected path: /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/qwen/inference/$PYTHON_SCRIPT"
    exit 1
fi


# Check if required directories exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' not found"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory '$IMAGE_DIR' not found"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory '$MODEL_DIR' not found"
    exit 1
fi


echo "All directories verified. Starting inference..."
echo "==================================="


# ===============================
# Run Python Script
# ===============================
python3 "$PYTHON_SCRIPT"


STATUS=$?


echo "==================================="
if [ $STATUS -eq 0 ]; then
    echo "Qwen inference completed successfully!"
    echo ""
    echo "Inference Summary:"
    echo "- Dataset: Staircase (9 test images)"
    echo "- Model: Qwen2.5-VL-7B-Instruct"
    echo "- Few-shot examples: 3 training images"
    echo "- Method: Few-shot learning with image-text pairs"
    echo ""
    echo "Generated files:"
    echo "- Predictions: $OUTPUT_DIR/test_predictions_qwen.jsonl"
    echo "- Summary: $OUTPUT_DIR/cer_evaluation_summary.json"
    echo ""
    echo "Key metrics saved:"
    echo "- Character Error Rate (CER) scores"
    echo "- Perfect match count"
    echo "- Weighted CER statistics"
else
    echo "Qwen inference failed with exit code: $STATUS"
    echo "Check logs for detailed error information."
fi

echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo ""
echo "To view detailed results:"
echo "cat $OUTPUT_DIR/cer_evaluation_summary.json"
echo ""
echo "To check individual predictions:"
echo "head -n 3 $OUTPUT_DIR/test_predictions_qwen.jsonl"
echo ""
echo "Job completed at: $(date)"
echo "==================================="
