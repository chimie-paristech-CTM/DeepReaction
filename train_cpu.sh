#!/bin/bash
set -e

PYTHON_ENV="python"
SCRIPT_DIR=$(dirname "$(realpath "$0")")
TRAIN_SCRIPT="${SCRIPT_DIR}/deep/cli/train.py"
OUTPUT_DIR="${SCRIPT_DIR}/results/xtb_multi"

DATASET="XTB"
READOUT="mean"
MODEL_TYPE="dimenet++"
BATCH_SIZE=32
NODE_DIM=128
RANDOM_SEED=42234
EPOCHS=200
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
DATASET_CSV="${SCRIPT_DIR}/dataset/DATASET_DA_F/dataset_xtb_final.csv"
TARGET_FIELDS=("G(TS)" "DrG")
TARGET_WEIGHTS=(1.0 1.0)
REACTANT_SUFFIX="_reactant.xyz"
TS_SUFFIX="_ts.xyz"
PRODUCT_SUFFIX="_product.xyz"
INPUT_FEATURES=("G(TS)_xtb" "DrG_xtb")

HIDDEN_CHANNELS=128
NUM_BLOCKS=4
INT_EMB_SIZE=64
BASIS_EMB_SIZE=8
OUT_EMB_CHANNELS=256
NUM_SPHERICAL=7
NUM_RADIAL=6
CUTOFF=5.0
ENVELOPE_EXPONENT=5
NUM_BEFORE_SKIP=1
NUM_AFTER_SKIP=2
NUM_OUTPUT_LAYERS=3
MAX_NUM_NEIGHBORS=32

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
        --model-type) MODEL_TYPE="$2"; shift 2 ;;
        --hidden-channels) HIDDEN_CHANNELS="$2"; shift 2 ;;
        --num-blocks) NUM_BLOCKS="$2"; shift 2 ;;
        --int-emb-size) INT_EMB_SIZE="$2"; shift 2 ;;
        --basis-emb-size) BASIS_EMB_SIZE="$2"; shift 2 ;;
        --out-emb-channels) OUT_EMB_CHANNELS="$2"; shift 2 ;;
        --num-spherical) NUM_SPHERICAL="$2"; shift 2 ;;
        --num-radial) NUM_RADIAL="$2"; shift 2 ;;
        --cutoff) CUTOFF="$2"; shift 2 ;;
        --envelope-exponent) ENVELOPE_EXPONENT="$2"; shift 2 ;;
        --num-before-skip) NUM_BEFORE_SKIP="$2"; shift 2 ;;
        --num-after-skip) NUM_AFTER_SKIP="$2"; shift 2 ;;
        --num-output-layers) NUM_OUTPUT_LAYERS="$2"; shift 2 ;;
        --max-num-neighbors) MAX_NUM_NEIGHBORS="$2"; shift 2 ;;
        --cv-folds) CV_FOLDS="$2"; shift 2 ;;
        --cv-test-fold) CV_TEST_FOLD="$2"; shift 2 ;;
        --cv-stratify) CV_STRATIFY=1; shift ;;
        --no-cv-grouped) CV_GROUPED=0; shift ;;
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
        --ckpt) CKPT_PATH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ ${#TARGET_FIELDS[@]} -ne ${#TARGET_WEIGHTS[@]} ]; then
    echo "Error: Number of target fields (${#TARGET_FIELDS[@]}) doesn't match number of target weights (${#TARGET_WEIGHTS[@]})"
    exit 1
fi

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

CMD="${CMD} --hidden_channels ${HIDDEN_CHANNELS}"
CMD="${CMD} --num_blocks ${NUM_BLOCKS}"
CMD="${CMD} --int_emb_size ${INT_EMB_SIZE}"
CMD="${CMD} --basis_emb_size ${BASIS_EMB_SIZE}"
CMD="${CMD} --out_emb_channels ${OUT_EMB_CHANNELS}"
CMD="${CMD} --num_spherical ${NUM_SPHERICAL}"
CMD="${CMD} --num_radial ${NUM_RADIAL}"
CMD="${CMD} --cutoff ${CUTOFF}"
CMD="${CMD} --envelope_exponent ${ENVELOPE_EXPONENT}"
CMD="${CMD} --num_before_skip ${NUM_BEFORE_SKIP}"
CMD="${CMD} --num_after_skip ${NUM_AFTER_SKIP}"
CMD="${CMD} --num_output_layers ${NUM_OUTPUT_LAYERS}"
CMD="${CMD} --max_num_neighbors ${MAX_NUM_NEIGHBORS}"

if [ ${CV_FOLDS} -gt 0 ]; then
    CMD="${CMD} --cv_folds ${CV_FOLDS}"
    CMD="${CMD} --cv_test_fold ${CV_TEST_FOLD}"
    if [ ${CV_STRATIFY} -eq 1 ]; then
        CMD="${CMD} --cv_stratify"
    fi
    if [ ${CV_GROUPED} -eq 0 ]; then
        CMD="${CMD} --no-cv_grouped"
    fi
else
    if [ -n "$VAL_CSV" ] && [ -n "$TEST_CSV" ]; then
        CMD="${CMD} --val_csv \"${VAL_CSV}\""
        CMD="${CMD} --test_csv \"${TEST_CSV}\""
    else
        CMD="${CMD} --train_ratio ${TRAIN_RATIO}"
        CMD="${CMD} --val_ratio ${VAL_RATIO}"
        CMD="${CMD} --test_ratio ${TEST_RATIO}"
    fi
fi

CMD="${CMD} --save_best_model --save_predictions"

if [ ${#TARGET_FIELDS[@]} -gt 0 ]; then
    CMD="${CMD} --reaction_target_fields"
    for field in "${TARGET_FIELDS[@]}"; do
        CMD="${CMD} \"${field}\""
    done
fi

if [ ${#TARGET_WEIGHTS[@]} -gt 0 ]; then
    CMD="${CMD} --target_weights"
    for weight in "${TARGET_WEIGHTS[@]}"; do
        CMD="${CMD} ${weight}"
    done
fi

CMD="${CMD} --reaction_file_suffixes \"${REACTANT_SUFFIX}\" \"${TS_SUFFIX}\" \"${PRODUCT_SUFFIX}\""

if [ ${#INPUT_FEATURES[@]} -gt 0 ]; then
    CMD="${CMD} --input_features"
    for feature in "${INPUT_FEATURES[@]}"; do
        CMD="${CMD} \"${feature}\""
    done
fi

# 👇 强制使用 CPU
CMD="${CMD} --no-cuda"

if [ -n "$CKPT_PATH" ]; then
    CMD="${CMD} --ckpt_path \"${CKPT_PATH}\""
fi

echo "Executing command: ${CMD}"
mkdir -p "${OUTPUT_DIR}"
bash -c "${CMD}"
