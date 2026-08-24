#!/bin/bash
# Plan B training and evaluation script: Updated Run B + bridges
# Uses train_eagle3_vlm_online.sh wrapper with environment variables (torchrun multi-GPU)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ============================================================================
# Shared config
# ============================================================================
TARGET_MODEL_NAME_OR_PATH="HuggingFaceTB/SmolVLM-256M-Instruct"
TRAIN_DATA_PATH="${REPO_ROOT}/dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl"
EVAL_DATA_PATH="${REPO_ROOT}/dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl"

# Training hyperparams
TRAIN_MODE="nccl"                    # Use NCCL backend with torchrun (4 GPUs)
NUM_TRAIN_EPOCHS=2
SAVE_STRATEGY="epoch"
EVAL_STRATEGY="epoch"
LOAD_FROM_CACHE_FILE="true"
LOGGING_STEPS=10

# Eval params
EVAL_NUM_SAMPLES=256
EVAL_NUM_SPEC_TOKENS=3
EVAL_MODEL_MAX_LENGTH=4096

# ============================================================================
# Helper functions
# ============================================================================

log_section() {
    echo ""
    echo "================================================================================"
    echo ">>> $1"
    echo "================================================================================"
    echo ""
}

log_step() {
    echo ">>> $1"
}

# ============================================================================
# EXPERIMENT 1: Updated Plan B (progressive_fc_draft_feedback)
# ============================================================================

# log_section "EXPERIMENT 1: Updated Plan B (progressive_fc_draft_feedback) - TRAINING COMMENTED OUT"
#
# OUTPUT_UPDATED_B="${REPO_ROOT}/outputs/smolvlm-256m-eagle3-updated-runb"
# mkdir -p "$OUTPUT_UPDATED_B"
#
# log_step "Training Updated Plan B (2 epochs, epoch-based save/eval)..."
# DRAFT_MODEL_CONFIG_PATH="angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-draft-feedback.json" \
# OUTPUT_DIR="$OUTPUT_UPDATED_B" \
# TARGET_MODEL_NAME_OR_PATH="$TARGET_MODEL_NAME_OR_PATH" \
# TRAIN_DATA_PATH="$TRAIN_DATA_PATH" \
# EVAL_DATA_PATH="$EVAL_DATA_PATH" \
# TRAIN_MODE="$TRAIN_MODE" \
# NUM_TRAIN_EPOCHS="$NUM_TRAIN_EPOCHS" \
# SAVE_STRATEGY="$SAVE_STRATEGY" \
# EVAL_STRATEGY="$EVAL_STRATEGY" \
# LOAD_FROM_CACHE_FILE="$LOAD_FROM_CACHE_FILE" \
# LOGGING_STEPS="$LOGGING_STEPS" \
# bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# log_step "Training Updated Plan B complete. Checkpoint: $OUTPUT_UPDATED_B"

OUTPUT_UPDATED_B="${REPO_ROOT}/outputs/smolvlm-256m-eagle3-updated-runb"

# ============================================================================
# EVALUATION: Updated Plan B (temp=0 and temp=1)
# ============================================================================

log_section "EVALUATION: Updated Plan B"

# Find latest checkpoint
LATEST_CKPT_UPDATED_B=$(find "$OUTPUT_UPDATED_B" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
if [ -z "$LATEST_CKPT_UPDATED_B" ]; then
    LATEST_CKPT_UPDATED_B="$OUTPUT_UPDATED_B"
fi

log_step "Using checkpoint: $LATEST_CKPT_UPDATED_B"

# Prepare draft config for vLLM eval
log_step "Preparing draft config for vLLM eval (Updated Plan B)..."
python scripts/speculative/smolvlm/prepare_draft_config_for_vllm_eval.py \
    --draft_model "$LATEST_CKPT_UPDATED_B" \
    --draft_model_config_path "angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-draft-feedback.json" \
    --dry_run

# Eval temp=0
log_step "Evaluating Updated Plan B (temp=0)..."
python tools/eval_smolvlm_eagle3_acceptance.py \
    --target_model "$TARGET_MODEL_NAME_OR_PATH" \
    --draft_model "$LATEST_CKPT_UPDATED_B" \
    --draft_model_config_path "angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-draft-feedback.json" \
    --data_path "$EVAL_DATA_PATH" \
    --output_file "${OUTPUT_UPDATED_B}/eval_results_temp0.json" \
    --num_samples "$EVAL_NUM_SAMPLES" \
    --num_spec_tokens "$EVAL_NUM_SPEC_TOKENS" \
    --model_max_length "$EVAL_MODEL_MAX_LENGTH" \
    --torch_dtype bfloat16 \
    --chat_template_type smolvlm

log_step "Updated Plan B evaluation complete."

# ============================================================================
# EXPERIMENT 2: Plan B with Early Exit Bridges
# ============================================================================

log_section "EXPERIMENT 2: Plan B with Early Exit Bridges"

OUTPUT_BRIDGES="${REPO_ROOT}/outputs/smolvlm-256m-eagle3-runb-bridges"
mkdir -p "$OUTPUT_BRIDGES"

log_step "Training Plan B with Bridges (2 epochs, epoch-based save/eval)..."
DRAFT_MODEL_CONFIG_PATH="angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-bridges.json" \
OUTPUT_DIR="$OUTPUT_BRIDGES" \
TARGET_MODEL_NAME_OR_PATH="$TARGET_MODEL_NAME_OR_PATH" \
TRAIN_DATA_PATH="$TRAIN_DATA_PATH" \
EVAL_DATA_PATH="$EVAL_DATA_PATH" \
TRAIN_MODE="$TRAIN_MODE" \
NUM_TRAIN_EPOCHS="$NUM_TRAIN_EPOCHS" \
SAVE_STRATEGY="$SAVE_STRATEGY" \
EVAL_STRATEGY="$EVAL_STRATEGY" \
LOAD_FROM_CACHE_FILE="$LOAD_FROM_CACHE_FILE" \
LOGGING_STEPS="$LOGGING_STEPS" \
bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

log_step "Training Plan B with Bridges complete. Checkpoint: $OUTPUT_BRIDGES"

# ============================================================================
# EVALUATION: Plan B with Bridges (temp=0 and temp=1)
# ============================================================================

log_section "EVALUATION: Plan B with Bridges"

# Find latest checkpoint
LATEST_CKPT_BRIDGES=$(find "$OUTPUT_BRIDGES" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
if [ -z "$LATEST_CKPT_BRIDGES" ]; then
    LATEST_CKPT_BRIDGES="$OUTPUT_BRIDGES"
fi

log_step "Using checkpoint: $LATEST_CKPT_BRIDGES"

# Prepare draft config for vLLM eval
log_step "Preparing draft config for vLLM eval (Bridges)..."
python scripts/speculative/smolvlm/prepare_draft_config_for_vllm_eval.py \
    --draft_model "$LATEST_CKPT_BRIDGES" \
    --draft_model_config_path "angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-bridges.json" \
    --dry_run

# Eval temp=0
log_step "Evaluating Plan B with Bridges (temp=0)..."
python tools/eval_smolvlm_eagle3_acceptance.py \
    --target_model "$TARGET_MODEL_NAME_OR_PATH" \
    --draft_model "$LATEST_CKPT_BRIDGES" \
    --draft_model_config_path "angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-bridges.json" \
    --data_path "$EVAL_DATA_PATH" \
    --output_file "${OUTPUT_BRIDGES}/eval_results_temp0.json" \
    --num_samples "$EVAL_NUM_SAMPLES" \
    --num_spec_tokens "$EVAL_NUM_SPEC_TOKENS" \
    --model_max_length "$EVAL_MODEL_MAX_LENGTH" \
    --torch_dtype bfloat16 \
    --chat_template_type smolvlm

log_step "Plan B with Bridges evaluation complete."

# ============================================================================
# Summary
# ============================================================================

log_section "EXPERIMENTS COMPLETE"

log_step "Results Summary:"
echo "  Updated Plan B (progressive_fc_draft_feedback):"
echo "    Output directory: $OUTPUT_UPDATED_B"
echo "    Eval results (temp=0): ${OUTPUT_UPDATED_B}/eval_results_temp0.json"
echo ""
echo "  Plan B with Early Exit Bridges:"
echo "    Output directory: $OUTPUT_BRIDGES"
echo "    Eval results (temp=0): ${OUTPUT_BRIDGES}/eval_results_temp0.json"
echo ""

log_step "To view results:"
echo "  cat ${OUTPUT_UPDATED_B}/eval_results_temp0.json"
echo "  cat ${OUTPUT_BRIDGES}/eval_results_temp0.json"
