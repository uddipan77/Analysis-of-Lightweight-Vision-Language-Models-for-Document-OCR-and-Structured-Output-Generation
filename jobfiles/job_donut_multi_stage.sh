#!/bin/bash -l
#
# SLURM Job: DONUT Multi-Stage + Contrastive (Staircase)
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=20:00:00
#SBATCH --export=NONE
#SBATCH --job-name=donut_stair_multistage

# Per-job log dir using %j (job id)
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/%j/staircase_multistage_cord.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/%j/staircase_multistage_cord.err

# ===============================
# Environment
# ===============================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
unset SLURM_EXPORT_ENV

module load python/3.12-conda
module load cuda/11.8.0

conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/global_env1

# ===============================
# Log directory (create before run)
# ===============================
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# ===============================
# Go to script folder
# ===============================
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/DONUT/finetune_with_image_text

# ===============================
# Params (tweak as needed)
# ===============================
DATA_DIR="/home/woody/iwi5/iwi5298h/json_staircase"
IMAGE_DIR="/home/woody/iwi5/iwi5298h/staircase_images"

NUM_EPOCHS=15
BATCH_SIZE=2
LEARNING_RATE=2e-5
MAX_LENGTH=768

PREDICT_ON="test"            # train | val | test | all
PREDICTION_BATCH_SIZE=1

AUGMENTATION_FACTOR=2       # default per your request
DISABLE_AUGMENTATION="false"

# Contrastive loss settings
CONTRASTIVE_WEIGHT=0.10
TEMPERATURE=0.07

# Model soup toggle (train 4 variants then average)
USE_MODEL_SOUP="false"      # set to "true" to enable

echo "==================================="
echo "DONUT Staircase: Multistage + Contrastive (+Soup optional)"
echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Log Directory: $LOG_DIR"
echo "Data Directory: $DATA_DIR"
echo "Image Directory: $IMAGE_DIR"
echo "Num Epochs (total): $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Length: $MAX_LENGTH"
echo "Predict On: $PREDICT_ON"
echo "Prediction Batch Size: $PREDICTION_BATCH_SIZE"
echo "Augmentation Factor: $AUGMENTATION_FACTOR"
echo "Disable Augmentation: $DISABLE_AUGMENTATION"
echo "Contrastive Weight: $CONTRASTIVE_WEIGHT"
echo "Temperature: $TEMPERATURE"
echo "Use Model Soup: $USE_MODEL_SOUP"
echo "==================================="

# ===============================
# Build python command
# NOTE: The Python script creates its own timestamped OUTPUT_DIR,
# so we don't pass an output_dir here.
# ===============================
PYTHON_CMD="python3 staircase_multistage_cord.py \
  --data_dir \"$DATA_DIR\" \
  --image_dir \"$IMAGE_DIR\" \
  --epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LEARNING_RATE \
  --max_length $MAX_LENGTH \
  --predict_on \"$PREDICT_ON\" \
  --prediction_batch_size $PREDICTION_BATCH_SIZE \
  --augmentation_factor $AUGMENTATION_FACTOR \
  --contrastive_weight $CONTRASTIVE_WEIGHT \
  --temperature $TEMPERATURE"

if [ "$DISABLE_AUGMENTATION" = "true" ]; then
  PYTHON_CMD="$PYTHON_CMD --disable_augmentation"
fi

if [ "$USE_MODEL_SOUP" = "true" ]; then
  PYTHON_CMD="$PYTHON_CMD --use_model_soup"
fi

echo "Running command:"
echo "$PYTHON_CMD"
echo "==================================="

# ===============================
# Execute
# ===============================
eval $PYTHON_CMD
STATUS=$?

echo "==================================="
if [ $STATUS -eq 0 ]; then
  echo "Training completed successfully!"
  echo ""
  echo "Notes:"
  echo "- The Python script writes a timestamped output directory under:"
  echo "  /home/vault/iwi5/iwi5298h/models_image_text/donut/staircase/"
  echo "- Check the job output log for the final 'All outputs saved to:' line"
  echo "  to see the exact output path (includes tensorboard logs, predictions, CER, etc.)"
else
  echo "Training failed with exit code: $STATUS"
  echo "Check logs for errors."
fi

echo "Log directory: $LOG_DIR"
echo "Job completed at: $(date)"
