#!/bin/bash
# Online Eagle3 training for SmolVLM / Idefics3.
#
# Default: DeepSpeed ZeRO-3, 4 GPU, 2 epochs, save every 5k steps
#   bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# TRAIN_MODE=nccl|deepspeed|gloo|python  (default: deepspeed)
#   TRAIN_MODE=nccl      bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#   TRAIN_MODE=gloo      bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#   TRAIN_MODE=python    bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# Overrides: TRAIN_DATA_PATH, EVAL_DATA_PATH, OUTPUT_DIR, DRAFT_MODEL_CONFIG_PATH,
#   DRAFT_MODEL_NAME_OR_PATH, MAX_STEPS, PROGRESSIVE_TARGET_HS_WARMUP_STEPS, ...

set -euo pipefail

TRAIN_MODE=${TRAIN_MODE:-deepspeed}
case "${TRAIN_MODE}" in
  nccl)
    LAUNCH=${LAUNCH:-torchrun}
    DIST_BACKEND=${DIST_BACKEND:-nccl}
    NPROC=${NPROC:-4}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    ;;
  deepspeed)
    LAUNCH=${LAUNCH:-torchrun}
    DIST_BACKEND=${DIST_BACKEND:-nccl}
    NPROC=${NPROC:-4}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-angelslim/compressor/speculative/train/configs/deepspeed_zero3.json}
    ;;
  gloo)
    LAUNCH=${LAUNCH:-torchrun}
    DIST_BACKEND=${DIST_BACKEND:-gloo}
    NPROC=${NPROC:-4}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    ;;
  python)
    LAUNCH=${LAUNCH:-python}
    EQUIV_NPROC=${EQUIV_NPROC:-4}
    NPROC=${NPROC:-1}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    NUM_PROC=${NUM_PROC:-1}
    ;;
  *)
    echo "ERROR: TRAIN_MODE must be nccl, deepspeed, gloo, or python (got: ${TRAIN_MODE})" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "${ROOT}/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

CONFIG_DIR=angelslim/compressor/speculative/train/configs
TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolVLM-256M-Instruct}
DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:-$CONFIG_DIR/smolvlm-256m-eagle3.json}
DRAFT_MODEL_NAME_OR_PATH=${DRAFT_MODEL_NAME_OR_PATH:-}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-output/smolvlm_256m_eagle3_online}
EMBED_WEIGHT_KEY=${EMBED_WEIGHT_KEY:-model.text_model.embed_tokens.weight}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-smolvlm}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
MAX_STEPS=${MAX_STEPS:--1}
PROGRESSIVE_TARGET_HS_WARMUP_STEPS=${PROGRESSIVE_TARGET_HS_WARMUP_STEPS:-0}
SAMPLE_NUM=${SAMPLE_NUM:-}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}

# Draft dtype identical across launchers (config JSON, usually bf16).
# Non-DeepSpeed paths use FP32MasterWeightOptimizer in Eagle3Trainer so Adam
# moments match ZeRO's FP32 optimizer state (plain bf16 Adam plateaus).
DRAFT_MODEL_DTYPE=${DRAFT_MODEL_DTYPE:-config}

LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-true}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
SAVE_STEPS=${SAVE_STEPS:-5000}

LAUNCH=${LAUNCH:-torchrun}
DIST_BACKEND=${DIST_BACKEND:-nccl}
NPROC=${NPROC:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
EQUIV_NPROC=${EQUIV_NPROC:-}
if [[ -z "${GRADIENT_ACCUMULATION_STEPS+x}" ]]; then
  if [[ -n "${EQUIV_NPROC}" ]] && [[ "${LAUNCH}" == "python" || "${NPROC}" == "1" ]]; then
    GRADIENT_ACCUMULATION_STEPS="${EQUIV_NPROC}"
  else
    GRADIENT_ACCUMULATION_STEPS=1
  fi
fi
NUM_PROC=${NUM_PROC:-4}

export CUDA_VISIBLE_DEVICES

ARGS=(
  tools/train_eagle3_online.py
  --modal_type VLM
  --target_model_name_or_path "${TARGET_MODEL_NAME_OR_PATH}"
  --draft_model_config_path "${DRAFT_MODEL_CONFIG_PATH}"
  --train_data_path "${TRAIN_DATA_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num_proc "${NUM_PROC}"
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE}"
  --save_strategy "${SAVE_STRATEGY}"
  --learning_rate 1e-4
  --weight_decay 0.0
  --warmup_ratio 0.05
  --lr_scheduler_type "constant"
  --logging_steps 100
  --model_max_length "${MODEL_MAX_LENGTH}"
  --embed_weight_key "${EMBED_WEIGHT_KEY}"
  --chat_template_type "${CHAT_TEMPLATE_TYPE}"
  --bf16
  --report_to none
  --run_name smolvlm-256m-eagle3-angelslim
)

[[ -n "${EVAL_DATA_PATH}" ]] && ARGS+=(--eval_data_path "${EVAL_DATA_PATH}")
[[ -n "${DRAFT_MODEL_NAME_OR_PATH}" ]] && ARGS+=(--draft_model_name_or_path "${DRAFT_MODEL_NAME_OR_PATH}")
[[ -n "${SAMPLE_NUM}" ]] && ARGS+=(--sample_num "${SAMPLE_NUM}")
[[ -n "${DEEPSPEED_CONFIG}" ]] && ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
[[ -n "${SAVE_STEPS}" ]] && ARGS+=(--save_steps "${SAVE_STEPS}")
[[ -n "${MAX_STEPS}" && "${MAX_STEPS}" != "-1" ]] && ARGS+=(--max_steps "${MAX_STEPS}")
[[ "${PROGRESSIVE_TARGET_HS_WARMUP_STEPS}" != "0" ]] && \
  ARGS+=(--progressive_target_hs_warmup_steps "${PROGRESSIVE_TARGET_HS_WARMUP_STEPS}")
[[ "${DRAFT_MODEL_DTYPE}" != "config" ]] && ARGS+=(--draft_model_dtype "${DRAFT_MODEL_DTYPE}")
[[ "${LAUNCH}" == "torchrun" ]] && ARGS+=(--ddp_backend "${DIST_BACKEND}")

echo "=== SmolVLM Eagle3 train ==="
echo "  TRAIN_MODE=${TRAIN_MODE}  LAUNCH=${LAUNCH}  NPROC=${NPROC}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  deepspeed=${DEEPSPEED_CONFIG:-none}  dtype=${DRAFT_MODEL_DTYPE}"

case "${LAUNCH}" in
  torchrun)
    torchrun --nproc_per_node="${NPROC}" "${ARGS[@]}"
    ;;
  python)
    unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT \
      TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS \
      TORCHELASTIC_USE_AGENT_STORE 2>/dev/null || true
    python "${ARGS[@]}"
    ;;
  *)
    echo "ERROR: LAUNCH must be torchrun or python, got: ${LAUNCH}" >&2
    exit 1
    ;;
esac
