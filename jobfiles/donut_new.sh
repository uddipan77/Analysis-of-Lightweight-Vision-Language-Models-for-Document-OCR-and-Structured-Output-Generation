#!/bin/bash -l
#
# SLURM Job Script for DONUT Fine-tuning with Multi-Stage Training
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --job-name=donut_inventory

# Use SLURM_JOB_ID for log folder - %j expands to job ID
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/%j/inven_cord_new.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/%j/inven_cord_new.err

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
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# ===============================
# Navigate to Script Directory
# ===============================
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/DONUT/finetune_with_image_text

# ===============================
# Parameter Setup
# ===============================
DATA_DIR="/home/woody/iwi5/iwi5298h/json_inven"
IMAGE_DIR="/home/woody/iwi5/iwi5298h/inventory_images"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/home/vault/iwi5/iwi5298h/models_image_text/donut/inventory/inven_data_cord_${TIMESTAMP}"
NUM_EPOCHS=50
BATCH_SIZE=2
LEARNING_RATE=1e-5
MAX_LENGTH=512
PREDICT_ON="test"
PREDICTION_BATCH_SIZE=2
EARLY_STOPPING_PATIENCE=5
EARLY_STOPPING_THRESHOLD=0.001
USE_MODEL_SOUP=""  # Add --use_model_soup to enable model soup

echo "==================================="
echo "DONUT Multi-Stage Training Configuration:"
echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Log Directory: $LOG_DIR"
echo "Data Directory: $DATA_DIR"
echo "Image Directory: $IMAGE_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Num Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Length: $MAX_LENGTH"
echo "Prediction On: $PREDICT_ON"
echo "Prediction Batch Size: $PREDICTION_BATCH_SIZE"
echo "Early Stopping Patience: $EARLY_STOPPING_PATIENCE"
echo "Early Stopping Threshold: $EARLY_STOPPING_THRESHOLD"
echo "Use Model Soup: $USE_MODEL_SOUP"
echo "==================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ===============================
# Run Python Script
# ===============================
python3 cord_new_inven.py \
    --data_dir "$DATA_DIR" \
    --image_dir "$IMAGE_DIR" \
    --epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --predict_on "$PREDICT_ON" \
    --prediction_batch_size $PREDICTION_BATCH_SIZE \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --early_stopping_threshold $EARLY_STOPPING_THRESHOLD \
    $USE_MODEL_SOUP

STATUS=$?

echo "==================================="
if [ $STATUS -eq 0 ]; then
    echo "Multi-stage training completed successfully!"
    echo "Strategy: Multi-Stage Training with Early Stopping"
else
    echo "Training failed with exit code: $STATUS"
    echo "Check logs for errors."
fi
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo "Job completed at: $(date)"
