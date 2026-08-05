#!/bin/bash
# Online Eagle3 training for SmolVLM / Idefics3.
#
# Reference recipe (this machine): per_device_bs=1, grad_accum=1, NPROC=4, 2 epochs
# → effective batch = 4. Recipe-equivalent across modes below (not bit-exact).
#
# ---------------------------------------------------------------------------
# A) torchrun + NCCL, 4 GPU  (default backend)
# ---------------------------------------------------------------------------
#   NPROC=4 NUM_TRAIN_EPOCHS=2 CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# ---------------------------------------------------------------------------
# B) torchrun + Gloo, 4 GPU  (use when NCCL is broken; same DDP math)
# ---------------------------------------------------------------------------
#   DIST_BACKEND=gloo NPROC=4 NUM_TRAIN_EPOCHS=2 CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# ---------------------------------------------------------------------------
# C) plain python, 1 GPU  (no process group; grad_accum=4 via EQUIV_NPROC)
# ---------------------------------------------------------------------------
#   LAUNCH=python EQUIV_NPROC=4 NUM_TRAIN_EPOCHS=2 CUDA_VISIBLE_DEVICES=0 NUM_PROC=1 \
#     bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# Or set TRAIN_MODE=nccl|gloo|python (fills the knobs above; still honor overrides):
#   TRAIN_MODE=gloo   NUM_TRAIN_EPOCHS=2 bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#   TRAIN_MODE=python NUM_TRAIN_EPOCHS=2 bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# Related — vLLM data gen on 1 GPU (not this script; throughput only):
#   GPU_NUM=1 CUDA_VISIBLE_DEVICES=0 bash scripts/speculative/smolvlm/run_vllm_server.sh
#   MAX_CLIENTS=1 NUM_THREADS=8 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
#
# Other overrides:
#   TRAIN_DATA_PATH=... EVAL_DATA_PATH=... OUTPUT_DIR=... DRAFT_MODEL_CONFIG_PATH=...
#
# Hawk (progressive H-fusion) config:
#   DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-hawk.json \
#     OUTPUT_DIR=output/smolvlm_256m_hawk NUM_TRAIN_EPOCHS=2 \
#     bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

set -euo pipefail

# Optional preset: TRAIN_MODE=nccl|gloo|python
TRAIN_MODE=${TRAIN_MODE:-}
case "${TRAIN_MODE}" in
  "" ) ;;
  nccl)
    LAUNCH=${LAUNCH:-torchrun}
    DIST_BACKEND=${DIST_BACKEND:-nccl}
    NPROC=${NPROC:-4}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
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
    echo "ERROR: TRAIN_MODE must be nccl, gloo, or python (got: ${TRAIN_MODE})" >&2
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
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-dataset/smolvlm_256m_target_gen/data_0-36.jsonl}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-output/smolvlm_256m_eagle3_online}
EMBED_WEIGHT_KEY=${EMBED_WEIGHT_KEY:-model.text_model.embed_tokens.weight}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-smolvlm}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
SAMPLE_NUM=${SAMPLE_NUM:-}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}
# Reuse HF datasets preprocess cache across restarts (true/false).
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-true}
# Save draft for vLLM eval. Use "epoch" (or steps) so OUTPUT_DIR has config.json + weights.
# Set SAVE_STRATEGY=no only for throwaway smoke runs.
SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
SAVE_STEPS=${SAVE_STEPS:-}

# --- Launch / distributed knobs ---
# LAUNCH=torchrun|python  (python = single process, no process group / no NCCL)
LAUNCH=${LAUNCH:-torchrun}
# DIST_BACKEND=nccl|gloo  (only used with LAUNCH=torchrun)
DIST_BACKEND=${DIST_BACKEND:-nccl}
NPROC=${NPROC:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
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

DDP_ARGS=()
if [[ "${LAUNCH}" == "torchrun" ]]; then
  DDP_ARGS+=(--ddp_backend "${DIST_BACKEND}")
fi

TRAIN_CMD=(
  tools/train_eagle3_online.py
  --modal_type VLM
  --target_model_name_or_path "${TARGET_MODEL_NAME_OR_PATH}"
  --draft_model_config_path "${DRAFT_MODEL_CONFIG_PATH}"
  --train_data_path "${TRAIN_DATA_PATH}"
  "${EVAL_ARGS[@]}"
  --output_dir "${OUTPUT_DIR}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
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
  --logging_steps 1
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
echo "  per_device_bs=${PER_DEVICE_TRAIN_BATCH_SIZE}  grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "  effective_batch≈${EFFECTIVE_BATCH}  (EQUIV_NPROC=${EQUIV_NPROC:-unset})"
echo "  NUM_PROC(datasets map)=${NUM_PROC}"

case "${LAUNCH}" in
  torchrun)
    echo "Running: torchrun --nproc_per_node=${NPROC} ${TRAIN_CMD[*]}"
    torchrun --nproc_per_node="${NPROC}" "${TRAIN_CMD[@]}"
    ;;
  python)
    if [[ "${NPROC}" != "1" ]]; then
      echo "WARNING: LAUNCH=python ignores NPROC=${NPROC} (single process)." >&2
    fi
    # Clear leftover torchrun env so HF does not think we are distributed.
    unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT \
      TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS \
      TORCHELASTIC_USE_AGENT_STORE 2>/dev/null || true
    echo "Running: python ${TRAIN_CMD[*]}"
    python "${TRAIN_CMD[@]}"
    ;;
  *)
    echo "ERROR: LAUNCH must be torchrun or python, got: ${LAUNCH}" >&2
    exit 1
    ;;
esac
