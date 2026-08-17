#!/bin/bash
# Offline HiViS-style baseline for SmolVLM / Idefics3.
#
# Stages:
#   STAGE=generate  create pruned hidden-state .ckpt data
#   STAGE=stage1    train progressive Eagle3 on the pruned offline data
#   STAGE=stage2    warm-start from stage1 and enable branch-distill top-k loss
#   STAGE=all       run generate, stage1, then stage2

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "${ROOT}/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

STAGE=${STAGE:-all}
LAUNCH=${LAUNCH:-torchrun}
NPROC=${NPROC:-4}
DIST_BACKEND=${DIST_BACKEND:-nccl}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES

CONFIG_DIR=angelslim/compressor/speculative/train/configs
TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolVLM-256M-Instruct}
STAGE1_DRAFT_CONFIG_PATH=${STAGE1_DRAFT_CONFIG_PATH:-$CONFIG_DIR/smolvlm-256m-eagle3-progressive-hivis.json}
STAGE2_DRAFT_CONFIG_PATH=${STAGE2_DRAFT_CONFIG_PATH:-$CONFIG_DIR/smolvlm-256m-eagle3-progressive-hivis-branch-distill-w03.json}
RAW_TRAIN_DATA_PATH=${RAW_TRAIN_DATA_PATH:-dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl}
RAW_EVAL_DATA_PATH=${RAW_EVAL_DATA_PATH:-dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl}
HIDDEN_ROOT=${HIDDEN_ROOT:-dataset/smolvlm_256m_hivis_offline}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/smolvlm_256m_hivis_offline_baseline}
STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-$OUTPUT_ROOT/stage1}
STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR:-$OUTPUT_ROOT/stage2_branch_distill_w03}
STAGE2_WARMSTART=${STAGE2_WARMSTART:-$STAGE1_OUTPUT_DIR}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-1}
NUM_PROC=${NUM_PROC:-4}
SAMPLE_NUM=${SAMPLE_NUM:-}
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-true}
OVERWRITE_HIDDEN=${OVERWRITE_HIDDEN:-false}

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
MAX_STEPS=${MAX_STEPS:--1}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.5}
SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
SAVE_STEPS=${SAVE_STEPS:-5000}
EVAL_STRATEGY=${EVAL_STRATEGY:-no}
EVAL_STEPS=${EVAL_STEPS:-5000}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}
CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-smolvlm}
EMBED_WEIGHT_KEY=${EMBED_WEIGHT_KEY:-model.text_model.embed_tokens.weight}
LM_HEAD_KEY=${LM_HEAD_KEY:-lm_head.weight}

run_generate() {
  local args=(
    tools/generate_smolvlm_hivis_offline_data.py
    --target_model_name_or_path "$TARGET_MODEL_NAME_OR_PATH"
    --draft_model_config_path "$STAGE1_DRAFT_CONFIG_PATH"
    --train_data_path "$RAW_TRAIN_DATA_PATH"
    --eval_data_path "$RAW_EVAL_DATA_PATH"
    --output_dir "$HIDDEN_ROOT"
    --split all
    --model_max_length "$MODEL_MAX_LENGTH"
    --batch_size "$GEN_BATCH_SIZE"
    --num_proc "$NUM_PROC"
    --load_from_cache_file "$LOAD_FROM_CACHE_FILE"
    --chat_template_type "$CHAT_TEMPLATE_TYPE"
  )
  [[ -n "$SAMPLE_NUM" ]] && args+=(--sample_num "$SAMPLE_NUM")
  [[ "$OVERWRITE_HIDDEN" == "true" ]] && args+=(--overwrite)
  python "${args[@]}"
}

run_train() {
  local config_path="$1"
  local output_dir="$2"
  local warmstart="${3:-}"
  local args=(
    tools/train_eagle3_offline.py
    --modal_type VLM
    --target_model_name_or_path "$TARGET_MODEL_NAME_OR_PATH"
    --draft_model_config_path "$config_path"
    --train_hidden_path "$HIDDEN_ROOT/train"
    --eval_hidden_path "$HIDDEN_ROOT/eval"
    --output_dir "$output_dir"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --max_steps "$MAX_STEPS"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --warmup_ratio "$WARMUP_RATIO"
    --lr_scheduler_type "$LR_SCHEDULER_TYPE"
    --max_grad_norm "$MAX_GRAD_NORM"
    --model_max_length "$MODEL_MAX_LENGTH"
    --embed_weight_key "$EMBED_WEIGHT_KEY"
    --lm_head_key "$LM_HEAD_KEY"
    --chat_template_type "$CHAT_TEMPLATE_TYPE"
    --save_strategy "$SAVE_STRATEGY"
    --save_steps "$SAVE_STEPS"
    --eval_strategy "$EVAL_STRATEGY"
    --eval_steps "$EVAL_STEPS"
    --logging_steps 100
    --bf16
    --report_to none
  )
  [[ -n "$warmstart" ]] && args+=(--draft_model_name_or_path "$warmstart")
  [[ -n "$DEEPSPEED_CONFIG" ]] && args+=(--deepspeed "$DEEPSPEED_CONFIG")

  case "$LAUNCH" in
    torchrun)
      torchrun --nproc_per_node="$NPROC" "${args[@]}"
      ;;
    python)
      unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT \
        TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS \
        TORCHELASTIC_USE_AGENT_STORE 2>/dev/null || true
      python "${args[@]}"
      ;;
    *)
      echo "ERROR: LAUNCH must be torchrun or python, got: $LAUNCH" >&2
      exit 1
      ;;
  esac
}

echo "=== SmolVLM HiViS offline baseline ==="
echo "  stage=$STAGE launch=$LAUNCH nproc=$NPROC cuda=$CUDA_VISIBLE_DEVICES"
echo "  target=$TARGET_MODEL_NAME_OR_PATH"
echo "  hidden=$HIDDEN_ROOT output=$OUTPUT_ROOT"

case "$STAGE" in
  generate)
    run_generate
    ;;
  stage1)
    run_train "$STAGE1_DRAFT_CONFIG_PATH" "$STAGE1_OUTPUT_DIR"
    ;;
  stage2)
    run_train "$STAGE2_DRAFT_CONFIG_PATH" "$STAGE2_OUTPUT_DIR" "$STAGE2_WARMSTART"
    ;;
  all)
    run_generate
    run_train "$STAGE1_DRAFT_CONFIG_PATH" "$STAGE1_OUTPUT_DIR"
    run_train "$STAGE2_DRAFT_CONFIG_PATH" "$STAGE2_OUTPUT_DIR" "$STAGE2_WARMSTART"
    ;;
  *)
    echo "ERROR: STAGE must be generate, stage1, stage2, or all (got: $STAGE)" >&2
    exit 1
    ;;
esac
