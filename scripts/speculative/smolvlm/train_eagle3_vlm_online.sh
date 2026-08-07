#!/bin/bash
# Online Eagle3 training for SmolVLM / Idefics3.
#
# Default recipe (this machine): DeepSpeed ZeRO-3, 4 GPU, 2 epochs, save every 5k steps
#   per_device_bs=1, grad_accum=1, NPROC=4 → effective batch = 4
#   MAX_STEPS=-1 so epochs are not overridden.
#
#   bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# ---------------------------------------------------------------------------
# A) torchrun + NCCL, 4 GPU  (no DeepSpeed)
# ---------------------------------------------------------------------------
#   TRAIN_MODE=nccl bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# ---------------------------------------------------------------------------
# A2) torchrun + NCCL + DeepSpeed ZeRO-3, 4 GPU  (default)
# ---------------------------------------------------------------------------
#   TRAIN_MODE=deepspeed bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# ---------------------------------------------------------------------------
# B) torchrun + Gloo, 4 GPU  (use when NCCL is broken; same DDP math)
# ---------------------------------------------------------------------------
#   TRAIN_MODE=gloo bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# ---------------------------------------------------------------------------
# C) plain python, 1 GPU  (no process group; grad_accum=4 via EQUIV_NPROC)
# ---------------------------------------------------------------------------
#   TRAIN_MODE=python bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# Related — vLLM data gen on 1 GPU (not this script; throughput only):
#   GPU_NUM=1 CUDA_VISIBLE_DEVICES=0 bash scripts/speculative/smolvlm/run_vllm_server.sh
#   MAX_CLIENTS=1 NUM_THREADS=8 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
#
# Other overrides:
#   TRAIN_DATA_PATH=... EVAL_DATA_PATH=... OUTPUT_DIR=... DRAFT_MODEL_CONFIG_PATH=...
#   DRAFT_MODEL_NAME_OR_PATH=...   # warm-start from a trained draft into a new run
#   PROGRESSIVE_TARGET_HS_WARMUP_STEPS=100  # teacher-force progressive aux first
#   MAX_STEPS=10000  # if set (>0 / not -1), overrides epochs
#
# Hawk (progressive H-fusion):
#   DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-hawk.json \
#     OUTPUT_DIR=output/smolvlm_256m_hawk \
#     bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

set -euo pipefail

# Optional preset: TRAIN_MODE=nccl|deepspeed|gloo|python (default: deepspeed ZeRO-3 / 4 GPU)
TRAIN_MODE=${TRAIN_MODE:-deepspeed}
case "${TRAIN_MODE}" in
  "" ) ;;
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

ANGEL_ENV_BIN=${ANGEL_ENV_BIN:-/home/hyang/miniconda3/envs/angel/bin}
PYTHON_BIN=${PYTHON_BIN:-${ANGEL_ENV_BIN}/python}
TORCHRUN_BIN=${TORCHRUN_BIN:-${ANGEL_ENV_BIN}/torchrun}

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
# If > 0 and not -1, overrides epochs (HF optimizer steps). Default -1 → use epochs.
MAX_STEPS=${MAX_STEPS:--1}
PROGRESSIVE_TARGET_HS_WARMUP_STEPS=${PROGRESSIVE_TARGET_HS_WARMUP_STEPS:-0}
SAMPLE_NUM=${SAMPLE_NUM:-}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}
# For plain NCCL/DDP, float32 draft params give torch AdamW fp32 moments,
# matching the important ZeRO-3 optimizer-state behavior more closely.
if [[ -z "${DRAFT_MODEL_DTYPE:-}" ]]; then
  if [[ "${TRAIN_MODE}" == "deepspeed" ]]; then
    DRAFT_MODEL_DTYPE=config
  else
    DRAFT_MODEL_DTYPE=float32
  fi
fi
# Reuse HF datasets preprocess cache across restarts (true/false).
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-true}
# Save draft for vLLM eval. Default: every 5k optimizer steps.
# Set SAVE_STRATEGY=no only for throwaway smoke runs.
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
SAVE_STEPS=${SAVE_STEPS:-5000}

# --- Launch / distributed knobs ---
# LAUNCH=torchrun|python  (python = single process, no process group / no NCCL)
LAUNCH=${LAUNCH:-torchrun}
# DIST_BACKEND=nccl|gloo  (only used with LAUNCH=torchrun)
DIST_BACKEND=${DIST_BACKEND:-nccl}
NPROC=${NPROC:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
# If EQUIV_NPROC is set and GRADIENT_ACCUMULATION_STEPS is unset, match a W-way
# DDP effective batch: effective_batch = per_device_bs * NPROC * grad_accum
# For LAUNCH=python (1 process): grad_accum defaults to EQUIV_NPROC.
EQUIV_NPROC=${EQUIV_NPROC:-}
if [[ -z "${GRADIENT_ACCUMULATION_STEPS+x}" ]]; then
  if [[ -n "${EQUIV_NPROC}" ]]; then
    if [[ "${LAUNCH}" == "python" || "${NPROC}" == "1" ]]; then
      GRADIENT_ACCUMULATION_STEPS="${EQUIV_NPROC}"
    else
      # Multi-GPU DDP already multiplies by NPROC; keep accum=1 unless overridden.
      GRADIENT_ACCUMULATION_STEPS=1
    fi
  else
    GRADIENT_ACCUMULATION_STEPS=1
  fi
fi
# HF datasets map workers. Broken mp servers: NUM_PROC=1 (or 0).
NUM_PROC=${NUM_PROC:-4}

export CUDA_VISIBLE_DEVICES

EVAL_ARGS=()
if [[ -n "${EVAL_DATA_PATH}" ]]; then
  EVAL_ARGS+=(--eval_data_path "${EVAL_DATA_PATH}")
fi

DRAFT_WARM_START_ARGS=()
if [[ -n "${DRAFT_MODEL_NAME_OR_PATH}" ]]; then
  DRAFT_WARM_START_ARGS+=(--draft_model_name_or_path "${DRAFT_MODEL_NAME_OR_PATH}")
fi

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_NUM}" ]]; then
  SAMPLE_ARGS+=(--sample_num "${SAMPLE_NUM}")
fi

DS_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  DS_ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

SAVE_ARGS=(--save_strategy "${SAVE_STRATEGY}")
if [[ -n "${SAVE_STEPS}" ]]; then
  SAVE_ARGS+=(--save_steps "${SAVE_STEPS}")
fi

MAX_STEPS_ARGS=()
if [[ -n "${MAX_STEPS}" && "${MAX_STEPS}" != "-1" ]]; then
  MAX_STEPS_ARGS+=(--max_steps "${MAX_STEPS}")
fi

PROGRESSIVE_TARGET_HS_WARMUP_ARGS=()
if [[ "${PROGRESSIVE_TARGET_HS_WARMUP_STEPS}" != "0" ]]; then
  PROGRESSIVE_TARGET_HS_WARMUP_ARGS+=(
    --progressive_target_hs_warmup_steps "${PROGRESSIVE_TARGET_HS_WARMUP_STEPS}"
  )
fi

DRAFT_MODEL_DTYPE_ARGS=()
if [[ "${DRAFT_MODEL_DTYPE}" != "config" ]]; then
  DRAFT_MODEL_DTYPE_ARGS+=(--draft_model_dtype "${DRAFT_MODEL_DTYPE}")
fi

DDP_ARGS=()
if [[ "${LAUNCH}" == "torchrun" ]]; then
  DDP_ARGS+=(--ddp_backend "${DIST_BACKEND}")
fi

TRAIN_CMD=(
  tools/train_eagle3_online.py
  --modal_type VLM
  --target_model_name_or_path "${TARGET_MODEL_NAME_OR_PATH}"
  --draft_model_config_path "${DRAFT_MODEL_CONFIG_PATH}"
  "${DRAFT_MODEL_DTYPE_ARGS[@]}"
  "${DRAFT_WARM_START_ARGS[@]}"
  --train_data_path "${TRAIN_DATA_PATH}"
  "${EVAL_ARGS[@]}"
  --output_dir "${OUTPUT_DIR}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  "${MAX_STEPS_ARGS[@]}"
  "${PROGRESSIVE_TARGET_HS_WARMUP_ARGS[@]}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num_proc "${NUM_PROC}"
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE}"
  "${SAVE_ARGS[@]}"
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
  "${DDP_ARGS[@]}"
  "${SAMPLE_ARGS[@]}"
  "${DS_ARGS[@]}"
)

EFFECTIVE_BATCH=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [[ "${LAUNCH}" == "torchrun" ]]; then
  EFFECTIVE_BATCH=$((EFFECTIVE_BATCH * NPROC))
fi

echo "=== SmolVLM Eagle3 train launch ==="
echo "  LAUNCH=${LAUNCH}  DIST_BACKEND=${DIST_BACKEND}  NPROC=${NPROC}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  angel_env_bin=${ANGEL_ENV_BIN}"
echo "  per_device_bs=${PER_DEVICE_TRAIN_BATCH_SIZE}  grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "  effective_batch≈${EFFECTIVE_BATCH}  (EQUIV_NPROC=${EQUIV_NPROC:-unset})"
echo "  epochs=${NUM_TRAIN_EPOCHS}  max_steps=${MAX_STEPS:-unset}"
echo "  DRAFT_MODEL_DTYPE=${DRAFT_MODEL_DTYPE}"
echo "  progressive_target_hs_warmup_steps=${PROGRESSIVE_TARGET_HS_WARMUP_STEPS}"
echo "  deepspeed_config=${DEEPSPEED_CONFIG:-none}"
echo "  NUM_PROC(datasets map)=${NUM_PROC}"

case "${LAUNCH}" in
  torchrun)
    echo "Running: ${TORCHRUN_BIN} --nproc_per_node=${NPROC} ${TRAIN_CMD[*]}"
    "${TORCHRUN_BIN}" --nproc_per_node="${NPROC}" "${TRAIN_CMD[@]}"
    ;;
  python)
    if [[ "${NPROC}" != "1" ]]; then
      echo "WARNING: LAUNCH=python ignores NPROC=${NPROC} (single process)." >&2
    fi
    # Clear leftover torchrun env so HF does not think we are distributed.
    unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT \
      TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS \
      TORCHELASTIC_USE_AGENT_STORE 2>/dev/null || true
    echo "Running: ${PYTHON_BIN} ${TRAIN_CMD[*]}"
    "${PYTHON_BIN}" "${TRAIN_CMD[@]}"
    ;;
  *)
    echo "ERROR: LAUNCH must be torchrun or python, got: ${LAUNCH}" >&2
    exit 1
    ;;
esac
