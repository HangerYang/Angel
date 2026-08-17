#!/usr/bin/env bash
# Train progressive layers_1_15_23 + image-key attn-match (w=0.1, all unroll
# steps, full-position / no cross-step KV), then eval temp=0 and temp=1.
#
#   bash scripts/speculative/smolvlm/run_attn_match_allsteps_w01.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

export PATH="/home/hyang/miniconda3/envs/angel/bin:${PATH}"

CONFIG_DIR="angelslim/compressor/speculative/train/configs"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/progressive_layer_group_tests}"

ATTN_RUN_ID="${ATTN_RUN_ID:-layers_1_15_23_attn_match_img_allsteps}"
ATTN_CONFIG="${ATTN_CONFIG:-${CONFIG_DIR}/smolvlm-256m-eagle3-progressive-layers-1-15-23-attn-match-allsteps.json}"
ATTN_OUT="${OUTPUT_ROOT}/${ATTN_RUN_ID}"

TRAIN_MODE="${TRAIN_MODE:-nccl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAX_STEPS="${MAX_STEPS:-66466}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
TARGET_HS_WARMUP_STEPS="${TARGET_HS_WARMUP_STEPS:-0}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-true}"

DATASETS="${DATASETS:-lmms-lab/textvqa MMMU/MMMU Lin-Chen/MMStar opendatalab/OmniDocBench HuggingFaceH4/MATH-500 lmms-lab/COCO-Caption}"
NUM_PROMPTS="${NUM_PROMPTS:-80}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

mkdir -p "${OUTPUT_ROOT}" "${ATTN_OUT}/logs"

echo "=== Attn-match allsteps w=0.1 ==="
echo "config=${ATTN_CONFIG}"
echo "out=${ATTN_OUT}"
echo "gpus=${CUDA_VISIBLE_DEVICES} mode=${TRAIN_MODE} max_steps=${MAX_STEPS}"
python3 - <<PY
import json
c=json.load(open("${ATTN_CONFIG}"))
keys=[k for k in c if k.startswith("attn_match")]
print("attn_match:", {k:c[k] for k in keys})
PY

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  if [[ -d "${ATTN_OUT}/checkpoint-66466" ]]; then
    echo "======== SKIP TRAIN ${ATTN_RUN_ID} (checkpoint-66466 exists) ========"
  else
    echo
    echo "======== TRAIN ${ATTN_RUN_ID} ========"
    TRAIN_MODE="${TRAIN_MODE}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    DRAFT_MODEL_CONFIG_PATH="${ATTN_CONFIG}" \
    OUTPUT_DIR="${ATTN_OUT}" \
    MAX_STEPS="${MAX_STEPS}" \
    NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
    SAVE_STRATEGY="${SAVE_STRATEGY}" \
    SAVE_STEPS="${SAVE_STEPS}" \
    MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH}" \
    TARGET_HS_WARMUP_STEPS="${TARGET_HS_WARMUP_STEPS}" \
    LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE}" \
      bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
  fi
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  for TEMP in 0 1; do
    echo
    echo "======== EVAL ${ATTN_RUN_ID} temp=${TEMP} ========"
    DRAFT_MODEL="${ATTN_OUT}" \
    DRAFT_MODEL_CONFIG_PATH="${ATTN_CONFIG}" \
    RUN_NAME="${ATTN_RUN_ID}" \
    OUT_ROOT="${ATTN_OUT}/eval_temp${TEMP}" \
    DATASETS="${DATASETS}" \
    NUM_PROMPTS="${NUM_PROMPTS}" \
    OUTPUT_LEN="${OUTPUT_LEN}" \
    NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    TEMP="${TEMP}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      bash scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh
  done
fi

echo
echo "=== ALL DONE ==="
echo "  attn: ${ATTN_OUT}"
echo "  eval: ${ATTN_OUT}/eval_temp0  ${ATTN_OUT}/eval_temp1"
