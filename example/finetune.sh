#!/bin/bash
# freeze_backbone_finetune.sh
# This script performs transfer learning by freezing the backbone and fine-tuning
# only the readout and prediction layers on reaction dataset

# Exit on any error
set -e

# Set default environment and paths
PYTHON_ENV="python"  # Change to your conda environment if needed
SCRIPT_DIR=$(dirname "$(realpath "$0")")
FINETUNE_SCRIPT="${SCRIPT_DIR}/deep/cli/finetune.py"
MODEL_PATH="${SCRIPT_DIR}/results/xtb/dimenet++/XTB/42/0/mean/XTB_target0_dimenet++_mean_seed42_20250304_114228/checkpoints/best-epoch=0000-val_total_loss=0.1673.ckpt"   # Change to your pre-trained model path
OUTPUT_DIR="${SCRIPT_DIR}/results/finetune/freeze_backbone"

# Default parameters
DATASET="XTB"
TARGET_ID=0
BATCH_SIZE=16
EPOCHS=1
MIN_EPOCHS=1
EARLY_STOPPING=10
LR=0.0001
OPTIMIZER="adamw"
SCHEDULER="cosine"
WARMUP_EPOCHS=2
MIN_LR=0.0000001
WEIGHT_DECAY=0.0001
DROPOUT=0.1
MAX_NUM_ATOMS=100

# XTB dataset specific parameters
REACTION_ROOT="${SCRIPT_DIR}/dataset/DATASET_DA"
REACTION_CSV="${SCRIPT_DIR}/dataset/DATASET_DA/DA_dataset_cleaned.csv"

# Create command with all parameters
CMD="${PYTHON_ENV} ${FINETUNE_SCRIPT}"
CMD="${CMD} --model_path ${MODEL_PATH}"
CMD="${CMD} --dataset ${DATASET}"
CMD="${CMD} --target_id ${TARGET_ID}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --eval_batch_size ${BATCH_SIZE}"
CMD="${CMD} --max_epochs ${EPOCHS}"
CMD="${CMD} --min_epochs ${MIN_EPOCHS}"
CMD="${CMD} --early_stopping_patience ${EARLY_STOPPING}"
CMD="${CMD} --lr ${LR}"
CMD="${CMD} --optimizer ${OPTIMIZER}"
CMD="${CMD} --scheduler ${SCHEDULER}"
CMD="${CMD} --warmup_epochs ${WARMUP_EPOCHS}"
CMD="${CMD} --min_lr ${MIN_LR}"
CMD="${CMD} --weight_decay ${WEIGHT_DECAY}"
CMD="${CMD} --gradient_clip_val 1.0"
CMD="${CMD} --output_dir ${OUTPUT_DIR}"
CMD="${CMD} --reaction_dataset_root ${REACTION_ROOT}"
CMD="${CMD} --reaction_dataset_csv ${REACTION_CSV}"
CMD="${CMD} --max_num_atoms ${MAX_NUM_ATOMS}"
CMD="${CMD} --save_best_model --save_last_model --save_predictions --save_visualizations"
CMD="${CMD} --progress_bar"
CMD="${CMD} --freeze_backbone"  # This is the key parameter for this script
CMD="${CMD} --cuda"
CMD="${CMD} --precision 16"  # Use mixed precision for faster training
CMD="${CMD} --random_seed 42"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Print command
echo "========================================================================================="
echo "Running fine-tuning with FROZEN BACKBONE on reaction dataset:"
echo "- Dataset:       XTB"
echo "- Pretrained model: ${MODEL_PATH}"
echo "- Batch size:    ${BATCH_SIZE}"
echo "- Epochs:        ${EPOCHS} (min: ${MIN_EPOCHS})"
echo "- Learning rate: ${LR}"
echo "- Optimizer:     ${OPTIMIZER}"
echo "- Scheduler:     ${SCHEDULER} (warmup: ${WARMUP_EPOCHS})"
echo "- Strategy:      Freeze backbone (transfer learning)"
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
    echo "Fine-tuning with frozen backbone completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "========================================================================================="
else
    echo "========================================================================================="
    echo "Fine-tuning failed with exit code $?"
    echo "========================================================================================="
    exit 1
fi