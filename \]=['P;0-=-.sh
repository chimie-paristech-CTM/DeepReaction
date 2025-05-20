#!/bin/bash
set -e

PYTHON_ENV="CUDA_VISIBLE_DEVICES=0 python"
SCRIPT_DIR=$(dirname "$(realpath "$0")")
TRAIN_SCRIPT="${SCRIPT_DIR}/deep/cli/train.py"
OUTPUT_DIR="${SCRIPT_DIR}/results/xtb_multi"


DATASET="XTB"
READOUT="mean"
MODEL_TYPE="dimenet++"
BATCH_SIZE=32
NODE_DIM=128
RANDOM_SEED=42
EPOCHS=50
MIN_EPOCHS=0
EARLY_STOPPING=40
LR=0.0005
OPTIMIZER="adamw"
SCHEDULER="warmup_cosine"
WARMUP_EPOCHS=10
MIN_LR=0.0000001
WEIGHT_DECAY=0.0001
DROPOUT=0.1

PREDICTION_HIDDEN_LAYERS=4
PREDICTION_HIDDEN_DIM=512

TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1

REACTION_ROOT="${SCRIPT_DIR}/dataset/DATASET_DA_F"
DATASET_CSV="${SCRIPT_DIR}/dataset/DATASET_DA_F/dataset_xtb_goodvibe_updated.csv"
TARGET_FIELDS=("G(TS)")
TARGET_WEIGHTS=(1.0)
REACTANT_SUFFIX="_reactant.xyz"
TS_SUFFIX="_ts.xyz"
PRODUCT_SUFFIX="_product.xyz"
INPUT_FEATURES=("G(TS)_xtb" )

# Default values for separate dataset files (null if not provided)
VAL_CSV=""
TEST_CSV=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -r|--readout) READOUT="$2"; shift 2 ;;
        -b|--batch) BATCH_SIZE="$2"; shift 2 ;;
        -n|--node-dim) NODE_DIM="$2"; shift 2 ;;
        -s|--seed) RANDOM_SEED="$2"; shift 2 ;;
        -e|--epochs) EPOCHS="$2"; shift 2 ;;
        --min-epochs) MIN_EPOCHS="$2"; shift 2 ;;
        --early-stopping) EARLY_STOPPING="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --warmup) WARMUP_EPOCHS="$2"; shift 2 ;;
        --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --dropout) DROPOUT="$2"; shift 2 ;;
        --pred-layers) PREDICTION_HIDDEN_LAYERS="$2"; shift 2 ;;
        --pred-dim) PREDICTION_HIDDEN_DIM="$2"; shift 2 ;;
        --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
        --val-ratio) VAL_RATIO="$2"; shift 2 ;;
        --test-ratio) TEST_RATIO="$2"; shift 2 ;;
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        --reaction-root) REACTION_ROOT="$2"; shift 2 ;;
        --dataset-csv) DATASET_CSV="$2"; shift 2 ;;
        --val-csv) VAL_CSV="$2"; shift 2 ;;
        --test-csv) TEST_CSV="$2"; shift 2 ;;
        --target-fields)
            # Parse target fields as a single string and clear default array
            TARGET_FIELDS=()
            IFS=' ' read -ra TEMP_FIELDS <<< "$2"
            for field in "${TEMP_FIELDS[@]}"; do
                TARGET_FIELDS+=("$field")
            done
            shift 2
            ;;
        --target-weights)
            # Parse target weights as a single string and clear default array
            TARGET_WEIGHTS=()
            IFS=' ' read -ra TEMP_WEIGHTS <<< "$2"
            for weight in "${TEMP_WEIGHTS[@]}"; do
                TARGET_WEIGHTS+=("$weight")
            done
            shift 2
            ;;
        --reactant-suffix) REACTANT_SUFFIX="$2"; shift 2 ;;
        --ts-suffix) TS_SUFFIX="$2"; shift 2 ;;
        --product-suffix) PRODUCT_SUFFIX="$2"; shift 2 ;;
        --input-features)
            # Parse input features as a single string and clear default array
            INPUT_FEATURES=()
            IFS=' ' read -ra TEMP_FEATURES <<< "$2"
            for feature in "${TEMP_FEATURES[@]}"; do
                INPUT_FEATURES+=("$feature")
            done
            shift 2
            ;;
        --ckpt) CKPT_PATH="$2"; shift 2 ;;
        --cuda) CUDA="--cuda"; shift ;;
        --no-cuda) CUDA="--no-cuda"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate target fields and weights
if [ ${#TARGET_FIELDS[@]} -ne ${#TARGET_WEIGHTS[@]} ]; then
    echo "Error: Number of target fields (${#TARGET_FIELDS[@]}) doesn't match number of target weights (${#TARGET_WEIGHTS[@]})"
    exit 1
fi

# Build the command
CMD="${PYTHON_ENV} ${TRAIN_SCRIPT}"
CMD="${CMD} --dataset ${DATASET}"
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
CMD="${CMD} --prediction_hidden_layers ${PREDICTION_HIDDEN_LAYERS}"
CMD="${CMD} --prediction_hidden_dim ${PREDICTION_HIDDEN_DIM}"
CMD="${CMD} --out_dir ${OUTPUT_DIR}"
CMD="${CMD} --reaction_dataset_root ${REACTION_ROOT}"
CMD="${CMD} --dataset_csv \"${DATASET_CSV}\""

# Check if using separate validation and test files
if [ -n "$VAL_CSV" ] && [ -n "$TEST_CSV" ]; then
    CMD="${CMD} --val_csv \"${VAL_CSV}\""
    CMD="${CMD} --test_csv \"${TEST_CSV}\""
else
    # Using single dataset with automatic splitting
    CMD="${CMD} --train_ratio ${TRAIN_RATIO}"
    CMD="${CMD} --val_ratio ${VAL_RATIO}"
    CMD="${CMD} --test_ratio ${TEST_RATIO}"
fi

CMD="${CMD} --save_best_model --save_predictions"

# Add target fields with proper quoting
if [ ${#TARGET_FIELDS[@]} -gt 0 ]; then
    CMD="${CMD} --reaction_target_fields"
    for field in "${TARGET_FIELDS[@]}"; do
        CMD="${CMD} \"${field}\""
    done
fi

# Add target weights with proper quoting
if [ ${#TARGET_WEIGHTS[@]} -gt 0 ]; then
    CMD="${CMD} --target_weights"
    for weight in "${TARGET_WEIGHTS[@]}"; do
        CMD="${CMD} ${weight}"
    done
fi

# Handle file suffixes
CMD="${CMD} --reaction_file_suffixes \"${REACTANT_SUFFIX}\" \"${TS_SUFFIX}\" \"${PRODUCT_SUFFIX}\""

# Add input features with proper quoting
if [ ${#INPUT_FEATURES[@]} -gt 0 ]; then
    CMD="${CMD} --input_features"
    for feature in "${INPUT_FEATURES[@]}"; do
        CMD="${CMD} \"${feature}\""
    done
fi

if [ -n "$CUDA" ]; then
    CMD="${CMD} ${CUDA}"
elif [ -x "$(command -v nvidia-smi)" ]; then
    CMD="${CMD} --cuda"
else
    CMD="${CMD} --no-cuda"
fi

if [ -n "$CKPT_PATH" ]; then
    CMD="${CMD} --ckpt_path \"${CKPT_PATH}\""
fi

# Print the command for debugging
echo "Executing command: ${CMD}"

mkdir -p "${OUTPUT_DIR}"
# Use bash -c to properly handle the quoted arguments
bash -c "${CMD}"