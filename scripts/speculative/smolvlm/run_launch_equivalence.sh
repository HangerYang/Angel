#!/usr/bin/env bash
# Compare launch backends with the same training recipe (not bit-exact).
#
# Default recipe — 10k optimizer steps, effective batch = 4:
#   NCCL/Gloo 4-GPU: per_device_bs=1, NPROC=4, grad_accum=1
#   Python 1-GPU:    per_device_bs=1, EQUIV_NPROC=4 → grad_accum=4
#
#   bash scripts/speculative/smolvlm/run_launch_equivalence.sh
#
# Overrides:
#   MAX_STEPS=1000 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
#     bash scripts/speculative/smolvlm/run_launch_equivalence.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

MAX_STEPS="${MAX_STEPS:-10000}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"  # unused when MAX_STEPS>0; kept for HF
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
EQUIV_NPROC="${EQUIV_NPROC:-4}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl}"
DRAFT_MODEL_CONFIG_PATH="${DRAFT_MODEL_CONFIG_PATH:-angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3.json}"
# One checkpoint at the end of the step budget.
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS}}"
GPUS="${GPUS:-0,1,2,3}"
SINGLE_GPU="${SINGLE_GPU:-0}"

if [[ ! -f "${TRAIN_DATA_PATH}" ]]; then
  echo "ERROR: missing TRAIN_DATA_PATH=${TRAIN_DATA_PATH}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_DATA_PATH}" ]]; then
  echo "ERROR: missing EVAL_DATA_PATH=${EVAL_DATA_PATH}" >&2
  exit 1
fi

TAG="steps${MAX_STEPS}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}"

run_one() {
  local name="$1"
  shift
  local out="output/equiv_${name}_${TAG}"
  local log="logs/equiv_${name}_${TAG}.log"
  mkdir -p logs output
  echo "================================================================"
  echo "[equiv] START ${name}  OUTPUT_DIR=${out}"
  echo "================================================================"
  env "$@" \
    MAX_STEPS="${MAX_STEPS}" \
    NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH}" \
    EVAL_DATA_PATH="${EVAL_DATA_PATH}" \
    DRAFT_MODEL_CONFIG_PATH="${DRAFT_MODEL_CONFIG_PATH}" \
    SAVE_STRATEGY="${SAVE_STRATEGY}" \
    SAVE_STEPS="${SAVE_STEPS}" \
    OUTPUT_DIR="${out}" \
    bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh \
    2>&1 | tee "${log}"
  echo "[equiv] DONE ${name}  log=${log}  out=${out}"
}

echo "[equiv] recipe: max_steps=${MAX_STEPS} per_device_bs=${PER_DEVICE_TRAIN_BATCH_SIZE} equiv_nproc=${EQUIV_NPROC}"
echo "[equiv] effective_batch≈$((PER_DEVICE_TRAIN_BATCH_SIZE * EQUIV_NPROC)) for all runs"
echo "[equiv] train=${TRAIN_DATA_PATH}"
echo "[equiv] eval=${EVAL_DATA_PATH}"
echo "[equiv] config=${DRAFT_MODEL_CONFIG_PATH}"

# 1) NCCL 4-GPU
run_one nccl \
  TRAIN_MODE=nccl NPROC="${EQUIV_NPROC}" CUDA_VISIBLE_DEVICES="${GPUS}"

# 2) Gloo 4-GPU
run_one gloo \
  TRAIN_MODE=gloo NPROC="${EQUIV_NPROC}" CUDA_VISIBLE_DEVICES="${GPUS}"

# 3) Python 1-GPU (EQUIV_NPROC → grad_accum so effective batch matches)
run_one python1gpu \
  TRAIN_MODE=python EQUIV_NPROC="${EQUIV_NPROC}" \
  CUDA_VISIBLE_DEVICES="${SINGLE_GPU}" NUM_PROC=1

echo
echo "[equiv] all runs finished."
echo "  output/equiv_nccl_${TAG}"
echo "  output/equiv_gloo_${TAG}"
echo "  output/equiv_python1gpu_${TAG}"
echo "Logs under logs/equiv_*_${TAG}.log"
