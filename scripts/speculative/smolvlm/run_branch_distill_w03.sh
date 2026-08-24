#!/usr/bin/env bash
# Train/eval a branch-change distillation SmolVLM Eagle3 config.
#
# Default is the included non-progressive top1-w03 config. Override CFG/OUT to
# run any other branch config in this branch, for example:
#
#   CFG=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top2-curr-r33k.json \
#   OUT=my_angel/eagle/branch-change-top2-curr-r33k \
#     bash scripts/speculative/smolvlm/run_branch_distill_w03.sh
#
# Set SKIP_TRAIN=1 or SKIP_EVAL=1 to run only one side.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PATH="/home/hyang/miniconda3/envs/angel/bin:${PATH}"

CONFIG_DIR="angelslim/compressor/speculative/train/configs"
CFG="${CFG:-${CONFIG_DIR}/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top1-w03.json}"
OUT="${OUT:-my_angel/eagle/branch-change-top1-w03}"
MAX_STEPS="${MAX_STEPS:-66466}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DATASETS="${DATASETS:-lmms-lab/textvqa MMMU/MMMU Lin-Chen/MMStar opendatalab/OmniDocBench HuggingFaceH4/MATH-500 lmms-lab/COCO-Caption}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

if [[ ! -f "${CFG}" ]]; then
  echo "ERROR: CFG does not exist: ${CFG}" >&2
  exit 1
fi

mkdir -p "${OUT}/logs"

echo "=== Branch-change distillation ==="
echo "config=${CFG}"
echo "out=${OUT}"
python3 - <<PYJSON
import json
c=json.load(open("${CFG}"))
print("branch_distill:", {k:c[k] for k in c if k.startswith("branch_distill")})
print("mode:", c.get("eagle_aux_injection_mode"), "layers:", c.get("num_hidden_layers"))
PYJSON

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  if compgen -G "${OUT}/checkpoint-*" > /dev/null; then
    echo "======== SKIP TRAIN (checkpoint exists under ${OUT}) ========"
  else
    echo "======== TRAIN ${OUT} ========"
    TRAIN_MODE=nccl \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    DRAFT_MODEL_CONFIG_PATH="${CFG}" \
    OUTPUT_DIR="${OUT}" \
    MAX_STEPS="${MAX_STEPS}" \
    NUM_TRAIN_EPOCHS=2 \
    SAVE_STRATEGY=steps \
    SAVE_STEPS=5000 \
    EVAL_STRATEGY=no \
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
