#!/bin/bash
set -e

PYTHON_ENV="CUDA_VISIBLE_DEVICES=0 python"
SCRIPT_DIR=$(dirname "$(realpath "$0")")
INFER_SCRIPT="${SCRIPT_DIR}/deep/cli/inference.py"
OUT_DIR="${SCRIPT_DIR}/results/inference"

DATASET="XTB"
READOUT="mean"
MODEL_TYPE="dimenet++"
BATCH_SIZE=32
NODE_DIM=128
RANDOM_SEED=42
NUM_WORKERS=4

REACTION_ROOT="${SCRIPT_DIR}/dataset/DATASET_DA_F"
INFER_CSV="${SCRIPT_DIR}/dataset/DATASET_DA_F/dataset_xtb_final.csv"
TARGET_FIELDS=("G(TS)" "DrG")
REACTANT_SUFFIX="_reactant.xyz"
TS_SUFFIX="_ts.xyz"
PRODUCT_SUFFIX="_product.xyz"
INPUT_FEATURES=("G(TS)_xtb" "DrG_xtb")


CKPT_PATH="${SCRIPT_DIR}/results/xtb_multi/XTB_dimenet++_mean_seed42_20250401_052941/checkpoints/best-epoch=0009-val_total_loss=0.1499.ckpt"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -r|--readout) READOUT="$2"; shift 2 ;;
        -b|--batch) BATCH_SIZE="$2"; shift 2 ;;
        -n|--node-dim) NODE_DIM="$2"; shift 2 ;;
        -s|--seed) RANDOM_SEED="$2"; shift 2 ;;
        -o|--out-dir) OUT_DIR="$2"; shift 2 ;;
        --reaction-root) REACTION_ROOT="$2"; shift 2 ;;
        --infer-csv) INFER_CSV="$2"; shift 2 ;;
        --model-type) MODEL_TYPE="$2"; shift 2 ;;
        --workers) NUM_WORKERS="$2"; shift 2 ;;
        --target-fields)
            TARGET_FIELDS=()
            IFS=' ' read -ra TEMP_FIELDS <<< "$2"
            for field in "${TEMP_FIELDS[@]}"; do
                TARGET_FIELDS+=("$field")
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
        --cuda) CUDA="--cuda"; shift ;;
        --no-cuda) CUDA="--no-cuda"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ ${#TARGET_FIELDS[@]} -lt 1 ]; then
    echo "Error: At least one target field must be specified"
    exit 1
fi

CMD="${PYTHON_ENV} ${INFER_SCRIPT}"
CMD="${CMD} --dataset ${DATASET}"
CMD="${CMD} --readout ${READOUT}"
CMD="${CMD} --model_type ${MODEL_TYPE}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --node_latent_dim ${NODE_DIM}"
CMD="${CMD} --num_workers ${NUM_WORKERS}"
CMD="${CMD} --random_seed ${RANDOM_SEED}"
CMD="${CMD} --out_dir ${OUT_DIR}"
CMD="${CMD} --reaction_dataset_root ${REACTION_ROOT}"
CMD="${CMD} --infer_csv \"${INFER_CSV}\""
CMD="${CMD} --ckpt_path \"${CKPT_PATH}\""

if [ ${#TARGET_FIELDS[@]} -gt 0 ]; then
    CMD="${CMD} --reaction_target_fields"
    for field in "${TARGET_FIELDS[@]}"; do
        CMD="${CMD} \"${field}\""
    done
fi

CMD="${CMD} --reaction_file_suffixes \"${REACTANT_SUFFIX}\" \"${TS_SUFFIX}\" \"${PRODUCT_SUFFIX}\""

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

echo "Executing command: ${CMD}"

mkdir -p "${OUT_DIR}"
bash -c "${CMD}"