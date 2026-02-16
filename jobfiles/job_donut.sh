#!/bin/bash -l
#
# SLURM Job Script for DONUT Fine-tuning (Fixed Version)
#
# ===============================
# SLURM Directives
# ===============================
#SBATCH --account=iwi5
#SBATCH --gres=gpu:rtx3080:1               # Request 1 NVIDIA V100 GPU
#SBATCH --partition=rtx3080                # Specify the GPU partition
#SBATCH --time=12:00:00                 # Maximum runtime of 12 hours
#SBATCH --export=NONE                   # Do not export current environment variables
#SBATCH --job-name=donut_fixed          # Job name (updated)
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/simple_strcuture3.out  # Standard output log file
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DONUT/simple_strcuture3.err   # Standard error log file

# ===============================
# Environment Configuration
# ===============================

# Set the CUDA memory allocation configuration for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Enable CUDA blocking for better error reporting (helpful for debugging)
export CUDA_LAUNCH_BLOCKING=1

# Enable CUDA device-side assertions for better error messages
export TORCH_USE_CUDA_DSA=1

# Set HTTP and HTTPS proxies if required
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# Unset SLURM_EXPORT_ENV to prevent SLURM from exporting environment variables
unset SLURM_EXPORT_ENV

# Load the necessary modules
module load python/3.12-conda        # Load Python Anaconda module
module load cuda/11.8.0              # Load CUDA 11.8 for better V100 compatibility

# Activate the Conda environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/uddipan_thesis

# ===============================
# Navigate to Script Directory
# ===============================
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/DONUT

# ===============================
# Training Configuration (Updated for Fixed Script)
# ===============================

# Set your training parameters here
DATA_DIR="/home/woody/iwi5/iwi5298h/json_inventory_images"
IMAGE_DIR="/home/woody/iwi5/iwi5298h/inventory_images"
OUTPUT_DIR="/home/vault/iwi5/iwi5298h/donut/donut_code_simple_structure3"
NUM_EPOCHS=10          # Start with fewer epochs for testing
BATCH_SIZE=1          # Keep at 1 for V100 memory constraints
LEARNING_RATE=1e-5    # Conservative learning rate
MAX_LENGTH=512        # Reduced to match the fixed code (was 1024)

# Prediction settings
PREDICTION_BATCH_SIZE=2  # Can be slightly higher for inference
PREDICT_ON="test"        # Start with validation set only

# Print the configuration for logging
echo "==================================="
echo "DONUT Fixed Fine-tuning Configuration:"
echo "==================================="
echo "Data Directory: $DATA_DIR"
echo "Image Directory: $IMAGE_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Length: $MAX_LENGTH"
echo "Prediction Batch Size: $PREDICTION_BATCH_SIZE"
echo "Predict On: $PREDICT_ON"
echo "==================================="

# Check if data directories exist
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "ERROR: Image directory does not exist: $IMAGE_DIR"
    exit 1
fi

# Check if required JSONL files exist
for file in train.jsonl val.jsonl; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "ERROR: Required file does not exist: $DATA_DIR/$file"
        exit 1
    fi
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print GPU information
echo "==================================="
echo "GPU Information:"
echo "==================================="
nvidia-smi
echo "==================================="

# Print Python environment information
echo "Python Environment:"
echo "==================================="
python3 --version
pip list | grep -E "(torch|transformers|pillow)"
echo "==================================="

# Print CUDA information
echo "CUDA Information:"
echo "==================================="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current GPU: {torch.cuda.current_device() if torch.cuda.is_available() else \"None\"}')"
echo "==================================="

# Test a small sample first (optional - uncomment to test)
# echo "Testing with a small sample first..."
# python3 -c "
# import json
# import os
# data_dir = '$DATA_DIR'
# for split in ['train', 'val']:
#     with open(os.path.join(data_dir, f'{split}.jsonl'), 'r') as f:
#         lines = f.readlines()
#         print(f'{split}.jsonl has {len(lines)} samples')
#         if len(lines) > 0:
#             sample = json.loads(lines[0])
#             print(f'Sample keys: {list(sample.keys())}')
# "

# Execute the Python Script with updated parameters
echo "Running fixed DONUT training..."
python3 simple_structure3.py \
    --data_dir "$DATA_DIR" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --predict_on "$PREDICT_ON" \
    --prediction_batch_size $PREDICTION_BATCH_SIZE

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "==================================="
    echo "Training completed successfully!"
    echo "Output directory: $OUTPUT_DIR"
    echo "==================================="
    
    # List the generated files
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"
    
    # Show final model directory
    if [ -d "$OUTPUT_DIR/final_model" ]; then
        echo "==================================="
        echo "Final model saved in: $OUTPUT_DIR/final_model"
        ls -la "$OUTPUT_DIR/final_model"
        echo "==================================="
    fi
    
    # Show prediction files
    for pred_file in "$OUTPUT_DIR"/*_predictions.jsonl; do
        if [ -f "$pred_file" ]; then
            echo "Prediction file: $pred_file"
            echo "Number of predictions: $(wc -l < "$pred_file")"
        fi
    done
    
    # Show disk usage
    echo "==================================="
    echo "Disk usage of output directory:"
    du -sh "$OUTPUT_DIR"/*
    echo "Total: $(du -sh "$OUTPUT_DIR")"
    echo "==================================="
    
    # Show GPU memory usage at the end
    echo "Final GPU memory usage:"
    nvidia-smi
    
else
    echo "==================================="
    echo "Training failed with exit code: $?"
    echo "Check the error log for details."
    echo "==================================="
    
    # Show GPU memory usage even on failure
    echo "GPU memory usage at failure:"
    nvidia-smi
    exit 1
fi

echo "Job completed at: $(date)"