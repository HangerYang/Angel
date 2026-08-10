#!/usr/bin/env bash
# Queue (3 runs):
#   [1] real_hawk LoRA normal (2 epochs, draft-HS feedback, no GT warmup)
#   [2] hawk GT warmup only (2 epochs) → saves end ckpt (+ warmup_end/)
#   [3] hawk regular (2 epochs) warm-started from [2]'s warmup_end ckpt
#
# Usage:
#   bash scripts/speculative/smolvlm/run.sh
#   TRAIN_MODE=nccl NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/speculative/smolvlm/run.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

CONFIG_DIR=angelslim/compressor/speculative/train/configs
TRAIN_MODE="${TRAIN_MODE:-nccl}"
NPROC="${NPROC:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"

REAL_HAWK_EPOCHS="${REAL_HAWK_EPOCHS:-2}"
HAWK_WARMUP_EPOCHS="${HAWK_WARMUP_EPOCHS:-2}"
HAWK_REGULAR_EPOCHS="${HAWK_REGULAR_EPOCHS:-2}"
# Cover the entire warmup phase (trainer gates on global_step < this).
HAWK_WARMUP_STEPS="${HAWK_WARMUP_STEPS:-999999999}"

REAL_HAWK_OUT="${REAL_HAWK_OUT:-output/smolvlm_256m_real_hawk_nccl}"
HAWK_WARMUP_OUT="${HAWK_WARMUP_OUT:-output/smolvlm_256m_hawk_warmup}"
HAWK_REGULAR_OUT="${HAWK_REGULAR_OUT:-output/smolvlm_256m_hawk_nccl_v2}"

COMMON=(
  TRAIN_MODE="${TRAIN_MODE}"
  NPROC="${NPROC}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  TRAIN_DATA_PATH="${TRAIN_DATA_PATH}"
  EVAL_DATA_PATH="${EVAL_DATA_PATH}"
  EVAL_STRATEGY="${EVAL_STRATEGY}"
  EVAL_STEPS="${EVAL_STEPS}"
  SAVE_STEPS="${SAVE_STEPS}"
)

latest_checkpoint() {
  local dir="$1"
  local ckpt
  ckpt="$(find "${dir}" -maxdepth 1 -type d -name 'checkpoint-*' \
    | sort -t- -k2 -n | tail -1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "ERROR: no checkpoint-* under ${dir}" >&2
    exit 1
  fi
  printf '%s\n' "${ckpt}"
}

# [1] real_hawk normal: draft-HS feedback from step 0 (no target-HS warmup).
echo "======== [1/3] real_hawk (LoRA) normal ${REAL_HAWK_EPOCHS}ep $(date -Is) ========"
env "${COMMON[@]}" \
  NUM_TRAIN_EPOCHS="${REAL_HAWK_EPOCHS}" \
  TARGET_HS_WARMUP_STEPS=0 \
  DRAFT_MODEL_CONFIG_PATH="${CONFIG_DIR}/smolvlm-256m-real-hawk.json" \
  OUTPUT_DIR="${REAL_HAWK_OUT}" \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

# [2] hawk GT warmup only. Epoch saves so the end-of-warmup model is on disk.
echo "======== [2/3] hawk GT warmup ${HAWK_WARMUP_EPOCHS}ep $(date -Is) ========"
env "${COMMON[@]}" \
  NUM_TRAIN_EPOCHS="${HAWK_WARMUP_EPOCHS}" \
  TARGET_HS_WARMUP_STEPS="${HAWK_WARMUP_STEPS}" \
  SAVE_STRATEGY=epoch \
  DRAFT_MODEL_CONFIG_PATH="${CONFIG_DIR}/smolvlm-256m-hawk.json" \
  OUTPUT_DIR="${HAWK_WARMUP_OUT}" \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

WARMUP_CKPT="$(latest_checkpoint "${HAWK_WARMUP_OUT}")"
WARMUP_END="${HAWK_WARMUP_OUT}/warmup_end"
rm -rf "${WARMUP_END}"
cp -a "${WARMUP_CKPT}" "${WARMUP_END}"
echo "======== warmup end ckpt: ${WARMUP_CKPT} → ${WARMUP_END} ========"

# [3] hawk regular: fresh run warm-started from warmup_end (no HF resume into warmup dir).
echo "======== [3/3] hawk regular ${HAWK_REGULAR_EPOCHS}ep from ${WARMUP_END} $(date -Is) ========"
env "${COMMON[@]}" \
  NUM_TRAIN_EPOCHS="${HAWK_REGULAR_EPOCHS}" \
  TARGET_HS_WARMUP_STEPS=0 \
  DRAFT_MODEL_CONFIG_PATH="${CONFIG_DIR}/smolvlm-256m-hawk.json" \
  DRAFT_MODEL_NAME_OR_PATH="${WARMUP_END}" \
  OUTPUT_DIR="${HAWK_REGULAR_OUT}" \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

echo "======== ALL DONE $(date -Is) ========"
