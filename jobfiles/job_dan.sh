#!/bin/bash -l
#
# SLURM Job Script for DAN OCR Evaluation
#
# ===============================
# SLURM Directives
# ===============================
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:1               # Request 1 RTX3080 GPU
#SBATCH --partition=v100                # Specify the GPU partition
#SBATCH --time=03:00:00                    # Maximum runtime of 2 hours (adjust as needed)
#SBATCH --export=NONE                      # Do not export current environment variables
#SBATCH --job-name=dan_ocr_eval            # Job name
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DAN/%j/dan_ocr_eval.out  # Standard output log file
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DAN/%j/dan_ocr_eval.err   # Standard error log file

# ===============================
# Environment Configuration
# ===============================
mkdir -p "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DAN/${SLURM_JOB_ID}"
# Set HTTP and HTTPS proxies if required
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# Unset SLURM_EXPORT_ENV to prevent SLURM from exporting environment variables
unset SLURM_EXPORT_ENV

# Load the necessary modules
module load python/3.12-conda        # Load Python Anaconda module
module load cuda/11.8.0              # Load CUDA 11.8

# Activate the Conda environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/dan_env

# ===============================
# Navigate to Script Directory
# ===============================
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/DAN/DAN/OCR/document_OCR/dan/

# ===============================
# Create logs directory if it doesn't exist
# ===============================
mkdir -p /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/DAN

# Print configuration for logging
echo "==================================="
echo "DAN OCR Evaluation Configuration:"
echo "==================================="
echo "Script Location: $(pwd)/predict_example2.py"
echo "Model Path: outputs/dan_rimes_page/checkpoints/dan_rimes_page.pt"
echo "Images Directory: /home/woody/iwi5/iwi5298h/inventory_images"
echo "Test Data: /home/woody/iwi5/iwi5298h/json_inventory_images/test.jsonl"
echo "Output Directory: /home/vault/iwi5/iwi5298h/models_image_text/DAN/inventory"
echo "==================================="

# Check if required files and directories exist
#echo "Checking required files and directories..."

# Check if Python script exists
#if [ ! -f "predict_example2.py" ]; then
#    echo "ERROR: Python script does not exist: $(pwd)/predict_example2.py"
#    exit 1
#fi

# Check if model file exists
#if [ ! -f "outputs/dan_rimes_page/checkpoints/dan_rimes_page.pt" ]; then
#    echo "ERROR: Model file does not exist: outputs/dan_rimes_page/checkpoints/dan_rimes_page.pt"
#    exit 1
#fi

# Check if images directory exists
#if [ ! -d "/home/woody/iwi5/iwi5298h/inventory_images" ]; then
#    echo "ERROR: Images directory does not exist: /home/woody/iwi5/iwi5298h/inventory_images"
#    exit 1
#fi

# Check if test data exists
#if [ ! -f "/home/woody/iwi5/iwi5298h/json_inventory_images/test.jsonl" ]; then
#    echo "ERROR: Test data file does not exist: /home/woody/iwi5/iwi5298h/json_inventory_images/test.jsonl"
#    exit 1
#fi

# Create output directory if it doesn't exist
mkdir -p /home/vault/iwi5/iwi5298h/models_image_text/DAN/inventory

echo "All required files and directories found!"

# Print environment information
echo "==================================="
echo "Environment Information:"
echo "==================================="
python3 --version
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "==================================="

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo "==================================="

# Count test images
echo "Test Data Information:"
echo "Number of test samples: $(wc -l < /home/woody/iwi5/iwi5298h/json_inventory_images/test.jsonl)"
echo "==================================="

# Execute the Python Script
echo "Starting DAN OCR evaluation..."
python3 predict_example2.py

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo "==================================="
    echo "DAN OCR evaluation completed successfully!"
    echo "==================================="
    
    # Show the latest output directory
    LATEST_OUTPUT=$(ls -t /home/vault/iwi5/iwi5298h/models_image_text/DAN/inventory/ | head -1)
    if [ -n "$LATEST_OUTPUT" ]; then
        OUTPUT_PATH="/home/vault/iwi5/iwi5298h/models_image_text/DAN/inventory/$LATEST_OUTPUT"
        echo "Results saved in: $OUTPUT_PATH"
        
        # List generated files
        echo "Generated files:"
        ls -la "$OUTPUT_PATH"
        
        # Show summary if it exists
        if [ -f "$OUTPUT_PATH/summary.txt" ]; then
            echo "==================================="
            echo "Evaluation Summary:"
            cat "$OUTPUT_PATH/summary.txt"
            echo "==================================="
        fi
        
        # Show disk usage
        echo "Output directory size: $(du -sh "$OUTPUT_PATH" | cut -f1)"
    fi
    
else
    echo "==================================="
    echo "DAN OCR evaluation failed with exit code: $?"
    echo "Check the error log for details."
    echo "==================================="
    exit 1
fi

echo "Job completed at: $(date)"
