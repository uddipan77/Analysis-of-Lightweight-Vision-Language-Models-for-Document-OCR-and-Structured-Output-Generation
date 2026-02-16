#!/bin/bash -l
#
# SLURM Job Script for DONUT Fine-tuning on Staircase Dataset with Data Augmentation
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=20:00:00
#SBATCH --export=NONE
#SBATCH --job-name=donut_base_stair


# Use SLURM_JOB_ID for log folder - %j expands to job ID
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/%j/staircase_base.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/%j/staircase_base.err


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


conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/qwen_vision


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
DATA_DIR="/home/woody/iwi5/iwi5298h/json_staircase"
IMAGE_DIR="/home/woody/iwi5/iwi5298h/staircase_images"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/home/vault/iwi5/iwi5298h/models_image_text/donut/staircase/donut_base/donut_base_${TIMESTAMP}"
NUM_EPOCHS=30
BATCH_SIZE=2
LEARNING_RATE=2e-5
MAX_LENGTH=768
PREDICT_ON="test"
PREDICTION_BATCH_SIZE=1
EARLY_STOPPING_PATIENCE=10
EARLY_STOPPING_THRESHOLD=0.005
AUGMENTATION_FACTOR=3
# Set to "true" if you want to disable augmentation
DISABLE_AUGMENTATION="false"


echo "==================================="
echo "DONUT Staircase Dataset Fine-tuning Configuration:"
echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Log Directory: $LOG_DIR"
echo "Data Directory: $DATA_DIR"
echo "Image Directory: $IMAGE_DIR"
echo "Model Cache Directory: /home/vault/iwi5/iwi5298h/models/donut_base"
echo "Output Directory: $OUTPUT_DIR"
echo "Num Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Length: $MAX_LENGTH"
echo "Prediction On: $PREDICT_ON"
echo "Prediction Batch Size: $PREDICTION_BATCH_SIZE"
echo "Early Stopping Patience: $EARLY_STOPPING_PATIENCE"
echo "Early Stopping Threshold: $EARLY_STOPPING_THRESHOLD"
echo "Augmentation Factor: $AUGMENTATION_FACTOR"
echo "Disable Augmentation: $DISABLE_AUGMENTATION"
echo "==================================="


# Create output directory
mkdir -p "$OUTPUT_DIR"


# ===============================
# Build Python Command with Conditional Augmentation
# ===============================
PYTHON_CMD="python3 staircase_base.py \
    --data_dir \"$DATA_DIR\" \
    --image_dir \"$IMAGE_DIR\" \
    --epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --predict_on \"$PREDICT_ON\" \
    --prediction_batch_size $PREDICTION_BATCH_SIZE \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --early_stopping_threshold $EARLY_STOPPING_THRESHOLD \
    --augmentation_factor $AUGMENTATION_FACTOR"

# Add augmentation flag if disabled
if [ "$DISABLE_AUGMENTATION" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --disable_augmentation"
fi

echo "Running command: $PYTHON_CMD"
echo "==================================="


# ===============================
# Run Python Script
# ===============================
eval $PYTHON_CMD


STATUS=$?


echo "==================================="
if [ $STATUS -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""
    echo "Training Summary:"
    echo "- Dataset: Staircase (26 train, 9 val, 9 test images)"
    if [ "$DISABLE_AUGMENTATION" = "true" ]; then
        echo "- Augmentation: DISABLED"
        echo "- Total training samples: 26"
    else
        TOTAL_SAMPLES=$((26 * ($AUGMENTATION_FACTOR + 1)))
        echo "- Augmentation: ENABLED (${AUGMENTATION_FACTOR}x factor)"
        echo "- Total training samples: $TOTAL_SAMPLES"
    fi
    echo "- Epochs completed: Check final_CER_scores.txt for details"
    echo "- Model saved in: $OUTPUT_DIR/final_model/"
else
    echo "Training failed with exit code: $STATUS"
    echo "Check logs for errors."
fi
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo ""
echo "Key files generated:"
echo "- Model: $OUTPUT_DIR/final_model/"
echo "- CER scores: $OUTPUT_DIR/final_CER_scores.txt"
echo "- Training summary: $OUTPUT_DIR/training_summary.txt"
echo "- Predictions: $OUTPUT_DIR/*_predictions.jsonl"
echo "- TensorBoard logs: $OUTPUT_DIR/tensorboard_logs/"
echo ""
echo "To view training curves:"
echo "tensorboard --logdir $OUTPUT_DIR/tensorboard_logs/"
echo "Job completed at: $(date)"
