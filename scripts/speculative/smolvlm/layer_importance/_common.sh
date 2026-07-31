#!/usr/bin/env bash
# Shared env for layer-importance runners.
# Override any of these when launching on larger data elsewhere.
#
# Required for real runs:
#   DATA_PATH=/path/to/large.jsonl
#
# Optional:
#   MODEL_PATH=HuggingFaceTB/SmolVLM-256M-Instruct
#   OUTPUT_DIR=./outputs
#   MAX_SAMPLES=   (empty = all)
#   MAX_LENGTH=2048
#   DEVICE=cuda
#   DTYPE=bfloat16
#   NUM_PROC=8
#   PYTHON=python

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-HuggingFaceTB/SmolVLM-256M-Instruct}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/dataset/smolvlm_256m_target_gen/data_0-36.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
DEVICE="${DEVICE:-}"
DTYPE="${DTYPE:-bfloat16}"
NUM_PROC="${NUM_PROC:-8}"
# Use the python that has torch+transformers. Examples:
#   PYTHON=/path/to/env/bin/python
#   conda activate angel   # then default `python` works
# Optional:
#   EVAL_RANDOM=1          # also score random 3-layer sets
#   MAX_SAMPLES=
PYTHON="${PYTHON:-python}"

mkdir -p "${OUTPUT_DIR}"

run_metrics() {
  local metrics="$1"
  shift || true
  local args=(
    --model-path "${MODEL_PATH}"
    --data-path "${DATA_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --metrics "${metrics}"
    --max-length "${MAX_LENGTH}"
    --dtype "${DTYPE}"
    --num-proc "${NUM_PROC}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    args+=(--max-samples "${MAX_SAMPLES}")
  fi
  if [[ -n "${DEVICE}" ]]; then
    args+=(--device "${DEVICE}")
  fi
  if [[ "${EVAL_RANDOM:-}" == "1" || "${EVAL_RANDOM:-}" == "true" ]]; then
    args+=(--eval-random)
  fi
  echo "[layer_importance] metrics=${metrics}"
  echo "[layer_importance] data=${DATA_PATH}"
  echo "[layer_importance] out=${OUTPUT_DIR}"
  "${PYTHON}" "${SCRIPT_DIR}/analyze_target_layers.py" "${args[@]}" "$@"
}
