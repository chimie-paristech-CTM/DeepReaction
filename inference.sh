#!/bin/bash
set -e

PYTHON_ENV="CUDA_VISIBLE_DEVICES=0 python"
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PREDICT_SCRIPT="${SCRIPT_DIR}/deep/cli/inference.py"

# Default parameters
CKPT_PATH="${SCRIPT_DIR}/results/xtb_multi/XTB_dimenet++_mean_seed42234_20250402_160400/checkpoints/best-epoch=0116-val_total_loss=0.0401.ckpt"
CSV_FILE="${SCRIPT_DIR}/dataset/DATASET_DA_F/dataset_xtb_final.csv"
DATA_ROOT="${SCRIPT_DIR}/dataset/DATASET_DA_F"
OUTPUT="predictions.csv"
BATCH_SIZE=32
CUDA=1
NUM_WORKERS=4
REACTANT_SUFFIX="_reactant.xyz"
TS_SUFFIX="_ts.xyz"
PRODUCT_SUFFIX="_product.xyz"
INPUT_FEATURES=("G(TS)_xtb" "DrG_xtb")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--checkpoint) CHECKPOINT="$2"; shift 2 ;;
        -f|--csv-file) CSV_FILE="$2"; shift 2 ;;
        -d|--data-root) DATA_ROOT="$2"; shift 2 ;;
        -o|--output) OUTPUT="$2"; shift 2 ;;
        -b|--batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --no-cuda) CUDA=0; shift ;;
        --workers) NUM_WORKERS="$2"; shift 2 ;;
        --reactant-suffix) REACTANT_SUFFIX="$2"; shift 2 ;;
        --ts-suffix) TS_SUFFIX="$2"; shift 2 ;;
        --product-suffix) PRODUCT_SUFFIX="$2"; shift 2 ;;
        --input-features)
            INPUT_FEATURES=()
            IFS=' ' read -ra TEMP_FEATURES <<< "$2"
            for feature in "${TEMP_FEATURES[@]}"; do
                INPUT_FEATURES+=("$feature")
            done
            shift 2
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check required arguments
if [ -z "$CHECKPOINT" ]; then
    echo "Error: Checkpoint path (--checkpoint) is required"
    exit 1
fi

if [ -z "$CSV_FILE" ]; then
    echo "Error: CSV file path (--csv-file) is required"
    exit 1
fi

if [ -z "$DATA_ROOT" ]; then
    echo "Error: Data root directory (--data-root) is required"
    exit 1
fi

# Build the command
CMD="${PYTHON_ENV} ${PREDICT_SCRIPT}"
CMD="${CMD} --checkpoint \"${CHECKPOINT}\""
CMD="${CMD} --csv_file \"${CSV_FILE}\""
CMD="${CMD} --data_root \"${DATA_ROOT}\""
CMD="${CMD} --output \"${OUTPUT}\""
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --num_workers ${NUM_WORKERS}"

# Add CUDA flag
if [ ${CUDA} -eq 1 ]; then
    CMD="${CMD} --cuda"
else
    CMD="${CMD} --no_cuda"
fi

# Add file suffixes
CMD="${CMD} --reactant_suffix \"${REACTANT_SUFFIX}\""
CMD="${CMD} --ts_suffix \"${TS_SUFFIX}\""
CMD="${CMD} --product_suffix \"${PRODUCT_SUFFIX}\""

# Add input features
if [ ${#INPUT_FEATURES[@]} -gt 0 ]; then
    CMD="${CMD} --input_features"
    for feature in "${INPUT_FEATURES[@]}"; do
        CMD="${CMD} \"${feature}\""
    done
fi

# Print the command for debugging
echo "Executing command: ${CMD}"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "${OUTPUT}")"

# Execute the command
bash -c "${CMD}"