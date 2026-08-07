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
# Draft knobs (see comments near assignments below):
#   DRAFT_MODEL_CONFIG_PATH   — architecture JSON (eagle3 | progressive | hawk | ...)
#   DRAFT_MODEL_NAME_OR_PATH  — optional draft checkpoint to warm-start from
#   DRAFT_MODEL_DTYPE         — config | float16 | bfloat16 | float32
# Other overrides: TRAIN_DATA_PATH, EVAL_DATA_PATH, OUTPUT_DIR, MAX_STEPS, ...

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

# Target (frozen) VLM that produces hidden states / teacher logits.
# Option: HF id or local path (default HuggingFaceTB/SmolVLM-256M-Instruct).
TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolVLM-256M-Instruct}

# Draft architecture blueprint (JSON). Required. Builds the draft from scratch.
# Options under $CONFIG_DIR:
#   smolvlm-256m-eagle3.json              — stock Eagle3 (fused_fc, 1 draft layer)
#   smolvlm-256m-eagle3-progressive.json  — progressive_staged, 3 layers, init from target
#   smolvlm-256m-eagle3-progressive-uninit.json — same progressive, no layer init
#   smolvlm-256m-hawk.json                — hawk (w1/w2 H-fusion, 3 layers)
# Example: DRAFT_MODEL_CONFIG_PATH=$CONFIG_DIR/smolvlm-256m-hawk.json
DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:-$CONFIG_DIR/smolvlm-256m-eagle3.json}

# Optional warm-start weights for the draft (not the target).
# Options:
#   "" (default)              — random / config init only (no pretrained draft)
#   path/to/checkpoint-NNNN   — load draft weights from a prior train run
#   path/to/draft_dir         — load from a saved draft model directory / HF id
# Skipped automatically if OUTPUT_DIR already has checkpoint-* (HF resume wins).
# Example: DRAFT_MODEL_NAME_OR_PATH=output/smolvlm_256m_hawk/checkpoint-30000
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

# Draft parameter dtype before Trainer/optimizer creation.
# Options: config | float16 | bfloat16 | float32
#   config (default) — use "dtype" from the draft JSON (usually bfloat16)
#   float32          — useful for plain DDP/NCCL so Adam gets FP32 moments
# Non-DeepSpeed paths also use FP32MasterWeightOptimizer so Adam moments match
# ZeRO's FP32 optimizer state (plain bf16 Adam plateaus).
# Example: DRAFT_MODEL_DTYPE=float32
DRAFT_MODEL_DTYPE=${DRAFT_MODEL_DTYPE:-config}

# Tokenization map/filter cache under <train_jsonl_dir>/.map_cache/ (not OUTPUT_DIR).
# true = reuse those files across restarts; false = remap. Delete .map_cache after
# changing preprocess code. Default true.
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
echo "  draft_config=${DRAFT_MODEL_CONFIG_PATH}"
echo "  draft_warmstart=${DRAFT_MODEL_NAME_OR_PATH:-none}  dtype=${DRAFT_MODEL_DTYPE}"
echo "  deepspeed=${DEEPSPEED_CONFIG:-none}  output=${OUTPUT_DIR}"

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
