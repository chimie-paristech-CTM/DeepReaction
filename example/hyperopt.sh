#!/bin/bash
# run_hyperopt.sh
# This script runs hyperparameter optimization for molecular property prediction models

# Exit on any error
set -e

# Set default environment and paths
PYTHON_ENV="python"  # Change to your conda environment if needed
SCRIPT_DIR=$(dirname "$(realpath "$0")")
HYPEROPT_SCRIPT="${SCRIPT_DIR}/deep/cli/hyperopt.py"
OUTPUT_DIR="${SCRIPT_DIR}/results/hyperopt"

# Default parameters
DATASET="XTB"
TARGET_ID=0
NUM_TRIALS=50
MAX_EPOCHS=50
MIN_EPOCHS=10
EARLY_STOPPING=15
N_JOBS=1
METRIC="val_total_loss"
DIRECTION="minimize"
BATCH_SIZE=16
MAX_NUM_ATOMS=100
CV_FOLDS=0

# XTB dataset specific parameters
REACTION_ROOT="/root/attention-based-pooling-for-quantum-properties-main/data_loading/DATASET_DA"
REACTION_CSV="/root/attention-based-pooling-for-quantum-properties-main/data_loading/DATASET_DA/DA_dataset_cleaned.csv"

# Parse command line arguments
function print_usage {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                  Show this help message and exit"
    echo "  -d, --dataset TYPE          Dataset to use (default: ${DATASET})"
    echo "                              Options: QM7, QM8, QM9, QMugs, XTB, benzene, aspirin, malonaldehyde, ethanol, toluene"
    echo "  -t, --target INT            Target property index (default: ${TARGET_ID})"
    echo "  -n, --trials INT            Number of optimization trials (default: ${NUM_TRIALS})"
    echo "  -e, --epochs INT            Maximum epochs per trial (default: ${MAX_EPOCHS})"
    echo "  --min-epochs INT            Minimum epochs per trial (default: ${MIN_EPOCHS})"
    echo "  --early-stopping INT        Early stopping patience (default: ${EARLY_STOPPING})"
    echo "  -j, --jobs INT              Number of parallel jobs (default: ${N_JOBS})"
    echo "  -m, --metric METRIC         Optimization metric (default: ${METRIC})"
    echo "                              Options: val_total_loss, val_mae, val_rmse, val_r2"
    echo "  -dir, --direction DIR       Optimization direction (default: ${DIRECTION})"
    echo "                              Options: minimize, maximize"
    echo "  -b, --batch INT             Batch size (default: ${BATCH_SIZE})"
    echo "  --cv-folds INT              Number of cross-validation folds (default: ${CV_FOLDS})"
    echo "  --reaction-root DIR         Root directory for reaction dataset"
    echo "                              (default: ${REACTION_ROOT})"
    echo "  --reaction-csv FILE         CSV file path for reaction dataset"
    echo "                              (default: ${REACTION_CSV})"
    echo "  -o, --output DIR            Output directory (default: ${OUTPUT_DIR})"
    echo "  --timeout INT               Timeout in seconds (optional)"
    echo "  --cuda / --no-cuda          Enable/disable CUDA (default: enabled if available)"
    echo "  --pruning                   Enable pruning of unpromising trials"
    echo ""
    echo "Example:"
    echo "  $0 --dataset XTB --trials 100 --jobs 4 --cv-folds 5 --pruning"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            print_usage
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -t|--target)
            TARGET_ID="$2"
            shift 2
            ;;
        -n|--trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        -e|--epochs)
            MAX_EPOCHS="$2"
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
        -j|--jobs)
            N_JOBS="$2"
            shift 2
            ;;
        -m|--metric)
            METRIC="$2"
            shift 2
            ;;
        -dir|--direction)
            DIRECTION="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --cv-folds)
            CV_FOLDS="$2"
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
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
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
        --pruning)
            PRUNING="--pruning"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Create command with all parameters
CMD="${PYTHON_ENV} ${HYPEROPT_SCRIPT}"
CMD="${CMD} --dataset ${DATASET}"
CMD="${CMD} --target_id ${TARGET_ID}"
CMD="${CMD} --n_trials ${NUM_TRIALS}"
CMD="${CMD} --max_epochs ${MAX_EPOCHS}"
CMD="${CMD} --min_epochs ${MIN_EPOCHS}"
CMD="${CMD} --early_stopping_patience ${EARLY_STOPPING}"
CMD="${CMD} --n_jobs ${N_JOBS}"
CMD="${CMD} --optimization_metric ${METRIC}"
CMD="${CMD} --optimization_direction ${DIRECTION}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --max_num_atoms ${MAX_NUM_ATOMS}"
CMD="${CMD} --cross_validation_folds ${CV_FOLDS}"
CMD="${CMD} --reaction_dataset_root ${REACTION_ROOT}"
CMD="${CMD} --reaction_dataset_csv ${REACTION_CSV}"
CMD="${CMD} --output_dir ${OUTPUT_DIR}"

# Add optional parameters
if [ -n "$TIMEOUT" ]; then
    CMD="${CMD} --timeout ${TIMEOUT}"
fi

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

if [ -n "$PRUNING" ]; then
    CMD="${CMD} ${PRUNING}"
fi

# Save best model and visualizations
CMD="${CMD} --save_best_model --save_visualizations --progress_bar"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Print command
echo "========================================================================================="
echo "Running hyperparameter optimization with the following configuration:"
echo "- Dataset:        ${DATASET}"
echo "- Target ID:      ${TARGET_ID}"
echo "- Num trials:     ${NUM_TRIALS}"
echo "- Epochs:         ${MAX_EPOCHS} (min: ${MIN_EPOCHS})"
echo "- Parallel jobs:  ${N_JOBS}"
echo "- Metric:         ${METRIC} (${DIRECTION})"
if [ "${CV_FOLDS}" -gt 0 ]; then
    echo "- CV folds:       ${CV_FOLDS}"
else
    echo "- CV folds:       Disabled"
fi
if [ -n "$PRUNING" ]; then
    echo "- Pruning:        Enabled"
else
    echo "- Pruning:        Disabled"
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
    echo "Hyperparameter optimization completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "========================================================================================="
else
    echo "========================================================================================="
    echo "Hyperparameter optimization failed with exit code $?"
    echo "========================================================================================="
    exit 1
fi