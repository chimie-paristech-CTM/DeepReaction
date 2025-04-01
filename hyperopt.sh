#!/bin/bash
set -e

PYTHON_ENV="python"
SCRIPT_DIR=$(dirname "$(realpath "$0")")
HYPEROPT_SCRIPT="${SCRIPT_DIR}/deep/cli/hyperopt.py"
OUTPUT_DIR="${SCRIPT_DIR}/results/hyperopt"

# Disable strict metric checking for Ray Tune
export TUNE_DISABLE_STRICT_METRIC_CHECKING=1

DATASET="XTB"
READOUT="mean"
MODEL_TYPE="dimenet++"
BATCH_SIZE=32
NODE_DIM=128
RANDOM_SEED=4223
EPOCHS=150
MIN_EPOCHS=0
EARLY_STOPPING=40
LR=0.0005
OPTIMIZER="adamw"
SCHEDULER="warmup_cosine"
WARMUP_EPOCHS=10
MIN_LR=0.0000001
WEIGHT_DECAY=0.0001
DROPOUT=0.1

CUTOFF_VALUES="5.0 10.0 15.0"
NUM_BLOCKS_VALUES="4 5 6"
# CUTOFF_VALUES="5.0"
# NUM_BLOCKS_VALUES="4"
# PREDICTION_HIDDEN_LAYERS_VALUES="3 4 5"
# PREDICTION_HIDDEN_DIM_VALUES="128 256 512"
PREDICTION_HIDDEN_LAYERS_VALUES="3"
PREDICTION_HIDDEN_DIM_VALUES="128"

CV_FOLDS=5
CV_TEST_FOLD=-1
CV_STRATIFY=0
CV_GROUPED=1

REACTION_ROOT="${SCRIPT_DIR}/dataset/DATASET_DA_F"
DATASET_CSV="${SCRIPT_DIR}/dataset/DATASET_DA_F/dataset_xtb_final.csv"
TARGET_FIELDS=("G(TS)" "DrG")
TARGET_WEIGHTS=(1.0 1.0)
REACTANT_SUFFIX="_reactant.xyz"
TS_SUFFIX="_ts.xyz"
PRODUCT_SUFFIX="_product.xyz"
INPUT_FEATURES=("G(TS)_xtb" "DrG_xtb")

HIDDEN_CHANNELS=128
INT_EMB_SIZE=64
BASIS_EMB_SIZE=8
OUT_EMB_CHANNELS=256
NUM_SPHERICAL=7
NUM_RADIAL=6
ENVELOPE_EXPONENT=5
NUM_BEFORE_SKIP=1
NUM_AFTER_SKIP=2
NUM_OUTPUT_LAYERS=3
MAX_NUM_NEIGHBORS=32
# PREDICTION_HIDDEN_LAYERS=3
# PREDICTION_HIDDEN_DIM=128

# Detect GPU count for parallel jobs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
PARALLEL_JOBS=$NUM_GPUS
if [ $PARALLEL_JOBS -eq 0 ]; then
    PARALLEL_JOBS=1
fi

# Process command line arguments
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
        --cutoff-values) CUTOFF_VALUES="$2"; shift 2 ;;
        --num-blocks-values) NUM_BLOCKS_VALUES="$2"; shift 2 ;;
        --pred-layers-values) PREDICTION_HIDDEN_LAYERS_VALUES="$2"; shift 2 ;;
        --pred-dim-values) PREDICTION_HIDDEN_DIM_VALUES="$2"; shift 2 ;;
        --cv-folds) CV_FOLDS="$2"; shift 2 ;;
        --cv-test-fold) CV_TEST_FOLD="$2"; shift 2 ;;
        --cv-stratify) CV_STRATIFY=1; shift ;;
        --no-cv-grouped) CV_GROUPED=0; shift ;;
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        --reaction-root) REACTION_ROOT="$2"; shift 2 ;;
        --dataset-csv) DATASET_CSV="$2"; shift 2 ;;
        --model-type) MODEL_TYPE="$2"; shift 2 ;;
        --hidden-channels) HIDDEN_CHANNELS="$2"; shift 2 ;;
        --int-emb-size) INT_EMB_SIZE="$2"; shift 2 ;;
        --basis-emb-size) BASIS_EMB_SIZE="$2"; shift 2 ;;
        --out-emb-channels) OUT_EMB_CHANNELS="$2"; shift 2 ;;
        --num-spherical) NUM_SPHERICAL="$2"; shift 2 ;;
        --num-radial) NUM_RADIAL="$2"; shift 2 ;;
        --envelope-exponent) ENVELOPE_EXPONENT="$2"; shift 2 ;;
        --num-before-skip) NUM_BEFORE_SKIP="$2"; shift 2 ;;
        --num-after-skip) NUM_AFTER_SKIP="$2"; shift 2 ;;
        --num-output-layers) NUM_OUTPUT_LAYERS="$2"; shift 2 ;;
        --max-num-neighbors) MAX_NUM_NEIGHBORS="$2"; shift 2 ;;
        --parallel-jobs) PARALLEL_JOBS="$2"; shift 2 ;;
        --target-fields)
            TARGET_FIELDS=()
            IFS=' ' read -ra TEMP_FIELDS <<< "$2"
            for field in "${TEMP_FIELDS[@]}"; do
                TARGET_FIELDS+=("$field")
            done
            shift 2
            ;;
        --target-weights)
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
            INPUT_FEATURES=()
            IFS=' ' read -ra TEMP_FEATURES <<< "$2"
            for feature in "${TEMP_FEATURES[@]}"; do
                INPUT_FEATURES+=("$feature")
            done
            shift 2
            ;;
        --cuda) CUDA="--cuda"; shift ;;
        --no-cuda) CUDA="--no-cuda"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate target fields and weights match
if [ ${#TARGET_FIELDS[@]} -ne ${#TARGET_WEIGHTS[@]} ]; then
    echo "Error: Number of target fields (${#TARGET_FIELDS[@]}) doesn't match number of target weights (${#TARGET_WEIGHTS[@]})"
    exit 1
fi

# Build the command
CMD="${PYTHON_ENV} ${HYPEROPT_SCRIPT}"
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
CMD="${CMD} --hidden_channels ${HIDDEN_CHANNELS}"
CMD="${CMD} --int_emb_size ${INT_EMB_SIZE}"
CMD="${CMD} --basis_emb_size ${BASIS_EMB_SIZE}"
CMD="${CMD} --out_emb_channels ${OUT_EMB_CHANNELS}"
CMD="${CMD} --num_spherical ${NUM_SPHERICAL}"
CMD="${CMD} --num_radial ${NUM_RADIAL}"
CMD="${CMD} --envelope_exponent ${ENVELOPE_EXPONENT}"
CMD="${CMD} --num_before_skip ${NUM_BEFORE_SKIP}"
CMD="${CMD} --num_after_skip ${NUM_AFTER_SKIP}"
CMD="${CMD} --num_output_layers ${NUM_OUTPUT_LAYERS}"
CMD="${CMD} --max_num_neighbors ${MAX_NUM_NEIGHBORS}"
CMD="${CMD} --out_dir ${OUTPUT_DIR}"
CMD="${CMD} --reaction_dataset_root ${REACTION_ROOT}"
CMD="${CMD} --dataset_csv ${DATASET_CSV}"

# Add hyperopt-specific parameters
CMD="${CMD} --cutoff_values ${CUTOFF_VALUES}"
CMD="${CMD} --num_blocks_values ${NUM_BLOCKS_VALUES}"
CMD="${CMD} --prediction_hidden_layers_values ${PREDICTION_HIDDEN_LAYERS_VALUES}"
CMD="${CMD} --prediction_hidden_dim_values ${PREDICTION_HIDDEN_DIM_VALUES}"
CMD="${CMD} --parallel_jobs ${PARALLEL_JOBS}"

# Add cross-validation parameters
CMD="${CMD} --cv_folds ${CV_FOLDS}"
CMD="${CMD} --cv_test_fold ${CV_TEST_FOLD}"
if [ ${CV_STRATIFY} -eq 1 ]; then
    CMD="${CMD} --cv_stratify"
fi
if [ ${CV_GROUPED} -eq 0 ]; then
    CMD="${CMD} --no-cv_grouped"
fi

# Add target fields with proper quoting
if [ ${#TARGET_FIELDS[@]} -gt 0 ]; then
    CMD="${CMD} --reaction_target_fields"
    for field in "${TARGET_FIELDS[@]}"; do
        CMD="${CMD} \"${field}\""
    done
fi

# Add target weights
if [ ${#TARGET_WEIGHTS[@]} -gt 0 ]; then
    CMD="${CMD} --target_weights"
    for weight in "${TARGET_WEIGHTS[@]}"; do
        CMD="${CMD} ${weight}"
    done
fi

# Add file suffixes
CMD="${CMD} --reaction_file_suffixes \"${REACTANT_SUFFIX}\" \"${TS_SUFFIX}\" \"${PRODUCT_SUFFIX}\""

# Add input features
if [ ${#INPUT_FEATURES[@]} -gt 0 ]; then
    CMD="${CMD} --input_features"
    for feature in "${INPUT_FEATURES[@]}"; do
        CMD="${CMD} \"${feature}\""
    done
fi

# Add CUDA flag if available
if [ -n "$CUDA" ]; then
    CMD="${CMD} ${CUDA}"
elif [ -x "$(command -v nvidia-smi)" ]; then
    CMD="${CMD} --cuda"
else
    CMD="${CMD} --no-cuda"
fi

# Print and execute the command
echo "Executing command: ${CMD}"
echo "Using ${PARALLEL_JOBS} parallel jobs (GPUs)"

mkdir -p "${OUTPUT_DIR}"
eval ${CMD}