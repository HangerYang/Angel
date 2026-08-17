#!/usr/bin/env bash
# 1) Resume good progressive layers_1_15_23 from ckpt-40000 → full 2 epochs (~66466).
# 2) Train progressive layers_1_15_23 + attn-match (w=0.1) for 2 epochs.
# 3) Eval both at temp=0.
#
#   bash scripts/speculative/smolvlm/run_progressive_1_15_23_finish_and_attn_match.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

export PATH="/home/hyang/miniconda3/envs/angel/bin:${PATH}"

CONFIG_DIR="angelslim/compressor/speculative/train/configs"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/progressive_layer_group_tests}"

FINISH_RUN_ID="layers_1_15_23"
FINISH_CONFIG="${CONFIG_DIR}/smolvlm-256m-eagle3-progressive-layers-1-15-23.json"
FINISH_OUT="${OUTPUT_ROOT}/${FINISH_RUN_ID}"

ATTN_RUN_ID="${ATTN_RUN_ID:-layers_1_15_23_attn_match_img}"
ATTN_CONFIG="${CONFIG_DIR}/smolvlm-256m-eagle3-progressive-layers-1-15-23-attn-match.json"
ATTN_OUT="${OUTPUT_ROOT}/${ATTN_RUN_ID}"

TRAIN_MODE="${TRAIN_MODE:-nccl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Full 2-epoch budget for this dataset/batch (matches prior progressive_nccl / hawk runs).
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
TEMP="${TEMP:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "${OUTPUT_ROOT}"

echo "=== Progressive 1/15/23 finish + attn-match ==="
echo "finish: resume ${FINISH_OUT} → max_steps=${MAX_STEPS} epochs=${NUM_TRAIN_EPOCHS}"
echo "attn:   ${ATTN_OUT} config=${ATTN_CONFIG} (attn_match_loss_weight=0.1)"
echo "gpus=${CUDA_VISIBLE_DEVICES} mode=${TRAIN_MODE}"

# ---------------------------------------------------------------------------
# Job 1: finish layers_1_15_23 to 2 epochs
# ---------------------------------------------------------------------------
if [[ ! -d "${FINISH_OUT}/checkpoint-40000" ]]; then
  echo "ERROR: expected resume ckpt ${FINISH_OUT}/checkpoint-40000" >&2
  exit 1
fi

if [[ -d "${FINISH_OUT}/checkpoint-66466" ]]; then
  echo "======== SKIP TRAIN ${FINISH_RUN_ID} (checkpoint-66466 exists) ========"
else
  echo
  echo "======== TRAIN ${FINISH_RUN_ID} (resume → 2 epochs) ========"
  TRAIN_MODE="${TRAIN_MODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  DRAFT_MODEL_CONFIG_PATH="${FINISH_CONFIG}" \
  OUTPUT_DIR="${FINISH_OUT}" \
  MAX_STEPS="${MAX_STEPS}" \
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
  SAVE_STRATEGY="${SAVE_STRATEGY}" \
  SAVE_STEPS="${SAVE_STEPS}" \
  MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH}" \
  TARGET_HS_WARMUP_STEPS="${TARGET_HS_WARMUP_STEPS}" \
  LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE}" \
    bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  echo
  echo "======== EVAL ${FINISH_RUN_ID} (2-epoch) ========"
  DRAFT_MODEL="${FINISH_OUT}" \
  DRAFT_MODEL_CONFIG_PATH="${FINISH_CONFIG}" \
  RUN_NAME="${FINISH_RUN_ID}_2ep" \
  OUT_ROOT="${FINISH_OUT}/eval_temp0_2ep" \
  DATASETS="${DATASETS}" \
  NUM_PROMPTS="${NUM_PROMPTS}" \
  OUTPUT_LEN="${OUTPUT_LEN}" \
  NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  TEMP="${TEMP}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
    bash scripts/speculative/smolvlm/eval_acceptance_suite.sh
fi

# ---------------------------------------------------------------------------
# Job 2: attn-match train (2 epochs from scratch)
# ---------------------------------------------------------------------------
if [[ -d "${ATTN_OUT}/checkpoint-66466" ]]; then
  echo "======== SKIP TRAIN ${ATTN_RUN_ID} (checkpoint-66466 exists) ========"
else
  echo
  echo "======== TRAIN ${ATTN_RUN_ID} (2 epochs, attn_match w=0.1) ========"
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

if [[ "${SKIP_EVAL}" != "1" ]]; then
  echo
  echo "======== EVAL ${ATTN_RUN_ID} ========"
  DRAFT_MODEL="${ATTN_OUT}" \
  DRAFT_MODEL_CONFIG_PATH="${ATTN_CONFIG}" \
  RUN_NAME="${ATTN_RUN_ID}" \
  OUT_ROOT="${ATTN_OUT}/eval_temp0" \
  DATASETS="${DATASETS}" \
  NUM_PROMPTS="${NUM_PROMPTS}" \
  OUTPUT_LEN="${OUTPUT_LEN}" \
  NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  TEMP="${TEMP}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
    bash scripts/speculative/smolvlm/eval_acceptance_suite.sh
fi

echo
echo "=== ALL DONE ==="
echo "  finished: ${FINISH_OUT}"
echo "  attn:     ${ATTN_OUT}"
