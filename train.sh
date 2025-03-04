#!/bin/bash

# ====================================================================================
# XTB Dataset Training Script
# This script runs the training pipeline for the XTB reaction dataset
# ====================================================================================

# Exit on any error
set -e

# Set default environment and paths
PYTHON_ENV="python"  # Change to your conda environment if needed, e.g., "conda activate myenv && python"
SCRIPT_DIR=$(dirname "$(realpath "$0")")
TRAIN_SCRIPT="${SCRIPT_DIR}/deep/cli/train.py"
OUTPUT_DIR="${SCRIPT_DIR}/results/xtb"

# Default parameters for XTB dataset
DATASET="XTB"
TARGET_ID=0
READOUT="mean"
MODEL_TYPE="dimenet++"
BATCH_SIZE=16
NODE_DIM=128
RANDOM_SEED=42
EPOCHS=1
MIN_EPOCHS=1
EARLY_STOPPING=30
LR=0.0005
OPTIMIZER="adamw"
SCHEDULER="warmup_cosine"
WARMUP_EPOCHS=10
MIN_LR=0.0000001
WEIGHT_DECAY=0.0001
DROPOUT=0.1

# XTB dataset specific parameters
REACTION_ROOT="${SCRIPT_DIR}/dataset/DATASET_DA"
REACTION_CSV="${SCRIPT_DIR}/dataset/DATASET_DA/DA_dataset_cleaned.csv"

# Parse command line arguments
function print_usage {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                  Show this help message and exit"
    echo "  -t, --target INT            Target property index to predict (default: ${TARGET_ID})"
    echo "  -r, --readout TYPE          Readout function (default: ${READOUT})"
    echo "                              Options: set_transformer, mean, sum, max, attention, multihead_attention"
    echo "  -b, --batch INT             Batch size (default: ${BATCH_SIZE})"
    echo "  -n, --node-dim INT          Node latent dimension (default: ${NODE_DIM})"
    echo "  -s, --seed INT              Random seed (default: ${RANDOM_SEED})"
    echo "  -e, --epochs INT            Maximum number of epochs (default: ${EPOCHS})"
    echo "  --min-epochs INT            Minimum number of epochs (default: ${MIN_EPOCHS})"
    echo "  --early-stopping INT        Early stopping patience (default: ${EARLY_STOPPING})"
    echo "  --lr FLOAT                  Learning rate (default: ${LR})"
    echo "  --warmup INT                Warmup epochs (default: ${WARMUP_EPOCHS})"
    echo "  --weight-decay FLOAT        Weight decay factor (default: ${WEIGHT_DECAY})"
    echo "  --dropout FLOAT             Dropout probability (default: ${DROPOUT})"
    echo "  -o, --output DIR            Output directory (default: ${OUTPUT_DIR})"
    echo "  --reaction-root DIR         Root directory for reaction dataset"
    echo "                              (default: ${REACTION_ROOT})"
    echo "  --reaction-csv FILE         CSV file path for reaction dataset"
    echo "                              (default: ${REACTION_CSV})"
    echo "  --ckpt PATH                 Path to checkpoint to resume training"
    echo "  --cuda / --no-cuda          Enable/disable CUDA (default: enabled if available)"
    echo ""
    echo "Example:"
    echo "  $0 --batch 32 --epochs 200 --lr 0.0003 --output ./results/xtb_custom"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            print_usage
            ;;
        -t|--target)
            TARGET_ID="$2"
            shift 2
            ;;
        -r|--readout)
            READOUT="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -n|--node-dim)
            NODE_DIM="$2"
            shift 2
            ;;
        -s|--seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --min-epochs)
            MIN_EPOCHS="$2"
            shift 2
            ;;
        --early-stopping)
            EARLY_STOPPING="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --warmup)
            WARMUP_EPOCHS="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --reaction-root)
            REACTION_ROOT="$2"
            shift 2
            ;;
        --reaction-csv)
            REACTION_CSV="$2"
            shift 2
            ;;
        --ckpt)
            CKPT_PATH="$2"
            shift 2
            ;;
        --cuda)
            CUDA="--cuda"
            shift
            ;;
        --no-cuda)
            CUDA="--no-cuda"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Create command with all parameters
CMD="${PYTHON_ENV} ${TRAIN_SCRIPT}"
CMD="${CMD} --dataset ${DATASET}"
CMD="${CMD} --target_id ${TARGET_ID}"
CMD="${CMD} --readout ${READOUT}"
CMD="${CMD} --model_type ${MODEL_TYPE}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --node_latent_dim ${NODE_DIM}"
CMD="${CMD} --dropout ${DROPOUT}"
CMD="${CMD} --random_seed ${RANDOM_SEED}"
CMD="${CMD} --max_epochs ${EPOCHS}"
CMD="${CMD} --min_epochs ${MIN_EPOCHS}"
CMD="${CMD} --early_stopping_patience ${EARLY_STOPPING}"
CMD="${CMD} --lr ${LR}"
CMD="${CMD} --optimizer ${OPTIMIZER}"
CMD="${CMD} --scheduler ${SCHEDULER}"
CMD="${CMD} --warmup_epochs ${WARMUP_EPOCHS}"
CMD="${CMD} --min_lr ${MIN_LR}"
CMD="${CMD} --weight_decay ${WEIGHT_DECAY}"
CMD="${CMD} --out_dir ${OUTPUT_DIR}"
CMD="${CMD} --reaction_dataset_root ${REACTION_ROOT}"
CMD="${CMD} --reaction_dataset_csv ${REACTION_CSV}"
CMD="${CMD} --save_best_model --save_predictions --save_visualizations"
CMD="${CMD} --log_to_file --logger_type tensorboard"

# Add CUDA flag if specified
if [ -n "$CUDA" ]; then
    CMD="${CMD} ${CUDA}"
else
    # Default to using CUDA if available
    if [ -x "$(command -v nvidia-smi)" ]; then
        CMD="${CMD} --cuda"
    else
        CMD="${CMD} --no-cuda"
    fi
fi

# Add checkpoint path if specified
if [ -n "$CKPT_PATH" ]; then
    CMD="${CMD} --ckpt_path ${CKPT_PATH}"
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Print command
echo "========================================================================================="
echo "Running XTB dataset training with the following configuration:"
echo "- Dataset:       XTB"
echo "- Target ID:     ${TARGET_ID}"
echo "- Readout:       ${READOUT}"
echo "- Model:         ${MODEL_TYPE}"
echo "- Batch size:    ${BATCH_SIZE}"
echo "- Node dim:      ${NODE_DIM}"
echo "- Epochs:        ${EPOCHS} (min: ${MIN_EPOCHS})"
echo "- Learning rate: ${LR}"
echo "- Optimizer:     ${OPTIMIZER}"
echo "- Scheduler:     ${SCHEDULER} (warmup: ${WARMUP_EPOCHS})"
echo "- Dataset root:  ${REACTION_ROOT}"
echo "- Dataset CSV:   ${REACTION_CSV}"
echo "- Output dir:    ${OUTPUT_DIR}"
echo ""
echo "Command:"
echo "$CMD"
echo "========================================================================================="

# Execute command
eval "$CMD"

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================================================================="
    echo "XTB dataset training completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "========================================================================================="
else
    echo "========================================================================================="
    echo "Training failed with exit code $?"
    echo "========================================================================================="
    exit 1
fi