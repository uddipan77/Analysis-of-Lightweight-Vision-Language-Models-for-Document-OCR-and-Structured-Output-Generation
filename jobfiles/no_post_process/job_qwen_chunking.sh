#!/bin/bash -l
#
# SLURM Job Script for Qwen2.5-VL-7B Fine-tuning with DeepSpeed and Chunking
#
#SBATCH --account=iwi5
#SBATCH --gres=gpu:v100:4                 # Request 4 NVIDIA V100 GPUs
#SBATCH --partition=v100                  # Specify the GPU partition
#SBATCH --nodes=1                         # Use 1 node
#SBATCH --ntasks-per-node=4               # 4 tasks per node (one per GPU)
#SBATCH --cpus-per-task=8                 # 8 CPUs per task
#SBATCH --time=10:00:00                   # Maximum runtime of 10 hours
#SBATCH --export=NONE
#SBATCH --job-name=qwen7b_finetune_chunked
#SBATCH --output=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/QWEN/%j/qwen7b_finetune.out
#SBATCH --error=/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/QWEN/%j/qwen7b_finetune.err

# ===============================
# Environment Configuration
# ===============================

# Create log directory BEFORE job execution
LOG_DIR="/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/logs/QWEN/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# Create timestamped output directory
BASE_OUTPUT_DIR="/home/vault/iwi5/iwi5298h/models_with_no_post_process/qwen7b/inventory_data"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/run_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 20001-29999 -n 1)
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# CUDA and memory optimization (enhanced for chunking)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0
export CUDA_MEMORY_FRACTION=0.95
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Proxy settings
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80
unset SLURM_EXPORT_ENV

# Load modules
module load python/3.12-conda
module load cuda/12.6

# Activate conda environment
conda activate /home/woody/iwi5/iwi5298h/software/private/conda/envs/uddipan_thesis

# ===============================
# Navigate to Script Directory
# ===============================
cd /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/qwen/finetune

# ===============================
# Verify Required Files
# ===============================
echo "Verifying required files..."
# Corrected:
if [ ! -f "inventory_chunking.py" ]; then
    echo "Error: inventory_chunking.py not found!"
    exit 1
fi

# Copy configuration files for reproducibility
cp ds_config_stage3.json "$OUTPUT_DIR/"
cp inventory_chunking.py "$OUTPUT_DIR/"


# ===============================
# Run the Training Script with DeepSpeed
# ===============================

echo "Starting Qwen2.5-VL-7B fine-tuning with DeepSpeed and Chunking..."
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "Using chunked dataset processing for memory efficiency"

# Run with DeepSpeed launcher
deepspeed --num_gpus=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    inventory_chunking.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --train_jsonl_path "/home/woody/iwi5/iwi5298h/json_inventory_images/train.jsonl" \
    --val_jsonl_path "/home/woody/iwi5/iwi5298h/json_inventory_images/val.jsonl" \
    --test_jsonl_path "/home/woody/iwi5/iwi5298h/json_inventory_images/test.jsonl" \
    --images_dir "/home/woody/iwi5/iwi5298h/inventory_images" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --metric_for_best_model "eval_cer" \
    --greater_is_better False \
    --bf16 \
    --gradient_checkpointing \
    --dataloader_num_workers 1 \
    --dataloader_pin_memory False \
    --dataloader_prefetch_factor 1 \
    --remove_unused_columns False \
    --deepspeed "ds_config_stage3.json" \
    --max_seq_length 1024 \
    --chunk_size 4

echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Check the following files:"
echo "  - Model: $OUTPUT_DIR/pytorch_model.bin"
echo "  - Predictions: $OUTPUT_DIR/test_predictions.jsonl"
echo "  - CER Results: $OUTPUT_DIR/cer_evaluation_results.txt"
echo "  - Configuration backup: $OUTPUT_DIR/ds_config_stage3.json"
echo "  - Script backup: $OUTPUT_DIR/inventory_chunking.py"
