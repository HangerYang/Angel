#!/usr/bin/env bash
# Train progressive branch-distill w=0.3 for 2 epochs, then eval temp 0/1.
#
#   bash scripts/speculative/smolvlm/run_branch_distill_w03.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PATH="/home/hyang/miniconda3/envs/angel/bin:${PATH}"

CFG="${CFG:-angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-branch-distill-w03.json}"
OUT="${OUT:-output/progressive_default_tests/progressive_branch_distill_w03}"
MAX_STEPS="${MAX_STEPS:-66466}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DATASETS="${DATASETS:-lmms-lab/textvqa MMMU/MMMU Lin-Chen/MMStar opendatalab/OmniDocBench HuggingFaceH4/MATH-500 lmms-lab/COCO-Caption}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

mkdir -p "${OUT}/logs"

echo "=== Branch distill w=0.3 ==="
echo "config=${CFG}"
echo "out=${OUT}"
python3 - <<PY
import json
c=json.load(open("${CFG}"))
print("branch_distill:", {k:c[k] for k in c if k.startswith("branch_distill")})
PY

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  if [[ -d "${OUT}/checkpoint-66466" ]]; then
    echo "======== SKIP TRAIN (checkpoint-66466 exists) ========"
  else
    echo "======== TRAIN ${OUT} ========"
    TRAIN_MODE=nccl \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    DRAFT_MODEL_CONFIG_PATH="${CFG}" \
    OUTPUT_DIR="${OUT}" \
    MAX_STEPS="${MAX_STEPS}" \
    NUM_TRAIN_EPOCHS=2 \
    SAVE_STEPS=5000 \
    MODEL_MAX_LENGTH=4096 \
    TARGET_HS_WARMUP_STEPS=0 \
    LOAD_FROM_CACHE_FILE=true \
      bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
  fi
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  for TEMP in 0 1; do
    echo "======== EVAL temp=${TEMP} ========"
    DRAFT_MODEL="${OUT}" \
    DRAFT_MODEL_CONFIG_PATH="${CFG}" \
    RUN_NAME="$(basename "${OUT}")" \
    OUT_ROOT="${OUT}/eval_temp${TEMP}" \
    DATASETS="${DATASETS}" \
    NUM_PROMPTS=80 \
    OUTPUT_LEN=1024 \
    NUM_SPEC_TOKENS=4 \
    MAX_NUM_SEQS=1 \
    TEMP="${TEMP}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      bash scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh
  done
fi

echo "=== ALL DONE === ${OUT}"
