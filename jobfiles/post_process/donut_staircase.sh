#!/bin/bash -l
#
# SLURM Job Script for DONUT Staircase Fine-tuning (All outputs in OUTPUT_DIR)
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=10:00:00
#SBATCH --export=NONE
#SBATCH --job-name=donut_staircase
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/%j/donut_staircase_training.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/%j/donut_staircase_training.err

# ===============================
# Environment Configuration
# ===============================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=1
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
unset SLURM_EXPORT_ENV

module load python/3.12-conda
module load cuda/12.6

conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/uddipan_thesis

# ===============================
# Create log directory for this job
# ===============================
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# ===============================
# Script & Data Locations
# ===============================
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/DONUT/finetune_with_post_process

DATA_DIR="/home/woody/iwi5/iwi5298h/json_staircase"
IMAGE_DIR="/home/woody/iwi5/iwi5298h/staircase_images"
DONUT_MODEL_DIR="/home/vault/iwi5/iwi5298h/models/donut/models--naver-clova-ix--donut-base-finetuned-cord-v2/snapshots/8003d433113256b4ce3a0f5bf604b29ff78a7451"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/home/vault/iwi5/iwi5298h/models_with_post_process/donut/staircase_data/donut_staircase_data_${SLURM_JOB_ID}_${TIMESTAMP}"

NUM_EPOCHS=25
BATCH_SIZE=1
LEARNING_RATE=2e-5
MAX_LENGTH=1024
PREDICT_ON="test"
PREDICTION_BATCH_SIZE=1
EARLY_STOPPING_PATIENCE=3
EARLY_STOPPING_THRESHOLD=0.0

# ===============================
# Print Configuration
# ===============================
echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Log Directory: $LOG_DIR"
echo "Data Directory: $DATA_DIR"
echo "Image Directory: $IMAGE_DIR"
echo "Donut Model Directory: $DONUT_MODEL_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Num Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Length: $MAX_LENGTH"
echo "Prediction On: $PREDICT_ON"
echo "Prediction Batch Size: $PREDICTION_BATCH_SIZE"
echo "Early Stopping Patience: $EARLY_STOPPING_PATIENCE"
echo "Early Stopping Threshold: $EARLY_STOPPING_THRESHOLD"
echo "==================================="

mkdir -p "$OUTPUT_DIR"

# ===============================
# Validate inputs
# ===============================
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "ERROR: Image directory does not exist: $IMAGE_DIR"
    exit 1
fi

if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "ERROR: Training data file not found: $DATA_DIR/train.jsonl"
    exit 1
fi

if [ ! -f "$DATA_DIR/val.jsonl" ]; then
    echo "ERROR: Validation data file not found: $DATA_DIR/val.jsonl"
    exit 1
fi

# ===============================
# System Info
# ===============================
echo "==================================="
echo "GPU Info:"
nvidia-smi
echo "Python:"
python3 --version
echo "PyTorch Version:"
python3 -c "import torch; print(torch.__version__); print(f'CUDA: {torch.version.cuda}')"
echo "Transformers Version:"
python3 -c "import transformers; print(transformers.__version__)"
echo "==================================="

# ===============================
# Start Training
# ===============================
echo "Launching DONUT training script..."
python3 staircase2.py \
    --data_dir "$DATA_DIR" \
    --image_dir "$IMAGE_DIR" \
    --epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --predict_on "$PREDICT_ON" \
    --prediction_batch_size $PREDICTION_BATCH_SIZE \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --early_stopping_threshold $EARLY_STOPPING_THRESHOLD

STATUS=$?

echo "==================================="
if [ $STATUS -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Output directory: $OUTPUT_DIR"
    echo "Log directory: $LOG_DIR"
    echo "Job completed at: $(date)"
else
    echo "Training failed with exit code: $STATUS"
    echo "Check logs for errors. Error log: $LOG_DIR/donut_staircase_training.err"
    exit 1
fi
