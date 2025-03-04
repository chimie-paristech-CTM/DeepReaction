#!/bin/bash
# run_inference.sh
# This script runs inference using a trained molecular property prediction model

# Exit on any error
set -e

# Set default environment and paths
PYTHON_ENV="python"  # Change to your conda environment if needed
SCRIPT_DIR=$(dirname "$(realpath "$0")")
INFERENCE_SCRIPT="${SCRIPT_DIR}/deep/cli/inference.py"
MODEL_PATH="${SCRIPT_DIR}/results/xtb/dimenet++/XTB/42/0/mean/XTB_target0_dimenet++_mean_seed42_20250304_114228/checkpoints/best-epoch=0000-val_total_loss=0.1673.ckpt"  # Change to your trained model path
OUTPUT_DIR="${SCRIPT_DIR}/results/inference"

# Default parameters
DATASET="XTB"
TARGET_ID=0
BATCH_SIZE=32
SPLIT="test"
CUDA=true
UNCERTAINTY=false
MC_SAMPLES=10
MAX_NUM_ATOMS=100

# XTB dataset specific parameters
REACTION_ROOT="/root/attention-based-pooling-for-quantum-properties-main/data_loading/DATASET_DA"
REACTION_CSV="/root/attention-based-pooling-for-quantum-properties-main/data_loading/DATASET_DA/DA_dataset_cleaned.csv"

# Parse command line arguments
function print_usage {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                  Show this help message and exit"
    echo "  -m, --model PATH            Path to trained model checkpoint (required)"
    echo "  -d, --dataset TYPE          Dataset to use (default: ${DATASET})"
    echo "                              Options: QM7, QM8, QM9, QMugs, XTB, benzene, aspirin, malonaldehyde, ethanol, toluene"
    echo "  -t, --target INT            Target property index (default: ${TARGET_ID})"
    echo "  -b, --batch INT             Batch size (default: ${BATCH_SIZE})"
    echo "  -s, --split TYPE            Dataset split to use (default: ${SPLIT})"
    echo "                              Options: train, val, test, all"
    echo "  --uncertainty               Enable uncertainty estimation using Monte Carlo dropout"
    echo "  --mc-samples INT            Number of Monte Carlo samples (default: ${MC_SAMPLES})"
    echo "  --ensemble DIR              Run ensemble inference using models in directory"
    echo "  --no-cuda                   Disable CUDA (default: enabled if available)"
    echo "  --reaction-root DIR         Root directory for reaction dataset"
    echo "                              (default: ${REACTION_ROOT})"
    echo "  --reaction-csv FILE         CSV file path for reaction dataset"
    echo "                              (default: ${REACTION_CSV})"
    echo "  -o, --output DIR            Output directory (default: ${OUTPUT_DIR})"
    echo "  --format FORMAT             Export format (default: csv)"
    echo "                              Options: csv, json, npy"
    echo "  --save-embeddings           Save molecular embeddings"
    echo "  --no-visualizations         Disable saving visualizations"
    echo ""
    echo "Example:"
    echo "  $0 --model ./outputs/checkpoints/best.ckpt --dataset XTB --split test --uncertainty"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            print_usage
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -t|--target)
            TARGET_ID="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        --uncertainty)
            UNCERTAINTY=true
            shift
            ;;
        --mc-samples)
            MC_SAMPLES="$2"
            shift 2
            ;;
        --ensemble)
            ENSEMBLE_DIR="$2"
            shift 2
            ;;
        --no-cuda)
            CUDA=false
            shift
            ;;
        --reaction-root)
            REACTION_ROOT="$2"
            shift 2
            ;;
        --reaction-csv)
            REACTION_CSV="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --format)
            EXPORT_FORMAT="$2"
            shift 2
            ;;
        --save-embeddings)
            SAVE_EMBEDDINGS=true
            shift
            ;;
        --no-visualizations)
            SAVE_VISUALIZATIONS=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required"
    print_usage
fi

# Create command with all parameters
CMD="${PYTHON_ENV} ${INFERENCE_SCRIPT}"
CMD="${CMD} --model_path ${MODEL_PATH}"
CMD="${CMD} --dataset ${DATASET}"
CMD="${CMD} --target_id ${TARGET_ID}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --split ${SPLIT}"
CMD="${CMD} --reaction_dataset_root ${REACTION_ROOT}"
CMD="${CMD} --reaction_dataset_csv ${REACTION_CSV}"
CMD="${CMD} --max_num_atoms ${MAX_NUM_ATOMS}"
CMD="${CMD} --output_dir ${OUTPUT_DIR}"

# Add optional parameters
if [ "$UNCERTAINTY" = true ]; then
    CMD="${CMD} --uncertainty --monte_carlo_dropout ${MC_SAMPLES}"
fi

if [ -n "$ENSEMBLE_DIR" ]; then
    CMD="${CMD} --ensemble --ensemble_dir ${ENSEMBLE_DIR}"
fi

if [ "$CUDA" = true ]; then
    CMD="${CMD} --cuda"
fi

if [ -n "$EXPORT_FORMAT" ]; then
    CMD="${CMD} --export_format ${EXPORT_FORMAT}"
fi

if [ "$SAVE_EMBEDDINGS" = true ]; then
    CMD="${CMD} --save_embeddings"
fi

# By default, save visualizations unless disabled
if [ "$SAVE_VISUALIZATIONS" != false ]; then
    CMD="${CMD} --save_visualizations"
fi

# Always save predictions
CMD="${CMD} --save_predictions"

# Add precision (default to mixed precision for faster inference)
CMD="${CMD} --precision 16"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Print command
echo "========================================================================================="
echo "Running inference with the following configuration:"
echo "- Model:          ${MODEL_PATH}"
echo "- Dataset:        ${DATASET}"
echo "- Split:          ${SPLIT}"
echo "- Batch size:     ${BATCH_SIZE}"
if [ "$UNCERTAINTY" = true ]; then
    echo "- Uncertainty:    Enabled (MC samples: ${MC_SAMPLES})"
else
    echo "- Uncertainty:    Disabled"
fi
if [ -n "$ENSEMBLE_DIR" ]; then
    echo "- Ensemble:       Enabled (directory: ${ENSEMBLE_DIR})"
else
    echo "- Ensemble:       Disabled"
fi
echo "- Dataset root:   ${REACTION_ROOT}"
echo "- Dataset CSV:    ${REACTION_CSV}"
echo "- Output dir:     ${OUTPUT_DIR}"
echo ""
echo "Command:"
echo "$CMD"
echo "========================================================================================="

# Execute command
eval "$CMD"

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================================================================="
    echo "Inference completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "========================================================================================="
else
    echo "========================================================================================="
    echo "Inference failed with exit code $?"
    echo "========================================================================================="
    exit 1
fi