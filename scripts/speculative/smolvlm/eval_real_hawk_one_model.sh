#!/usr/bin/env bash
# One-model real_hawk eval (shared target W + LoRA only on draft steps).
#
# This is the speedup number to report for the one-model story — NOT merged
# 2-model vLLM hawk tok/s.
#
# Usage:
#   DRAFT_MODEL=output/smolvlm_256m_real_hawk_nccl/checkpoint-XXXX \
#     bash scripts/speculative/smolvlm/eval_real_hawk_one_model.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -f "${ROOT}/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

# Prefer angel env if available
if [[ -f /home/hyang/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hyang/miniconda3/etc/profile.d/conda.sh
  conda activate angel 2>/dev/null || true
fi

CONFIG_DIR=angelslim/compressor/speculative/train/configs

# TARGET_MODEL — frozen HF target VLM. HF id or local path.
#   Default: HuggingFaceTB/SmolVLM-256M-Instruct
TARGET_MODEL="${TARGET_MODEL:-HuggingFaceTB/SmolVLM-256M-Instruct}"

# DRAFT_MODEL — real_hawk checkpoint dir (required). Must contain config.json + weights/LoRA.
#   Example: output/smolvlm_256m_real_hawk_nccl/checkpoint-30000
DRAFT_MODEL="${DRAFT_MODEL:?Set DRAFT_MODEL to a real_hawk checkpoint dir}"

# DRAFT_MODEL_CONFIG_PATH — real_hawk train JSON (LoRA r/alpha/targets + aux layer ids).
#   Default: .../smolvlm-256m-real-hawk.json (alias: layer-skip-lora.json)
DRAFT_MODEL_CONFIG_PATH="${DRAFT_MODEL_CONFIG_PATH:-${CONFIG_DIR}/smolvlm-256m-real-hawk.json}"

# DATASET — HF dataset id or local jsonl. Examples: lmms-lab/textvqa | path/to.jsonl
DATASET="${DATASET:-lmms-lab/textvqa}"

# NUM_PROMPTS — # examples. Typical: 20 (default) | 80
NUM_PROMPTS="${NUM_PROMPTS:-20}"

# NUM_SPEC_TOKENS — draft depth K. Typical: 3|4|5
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}"

# MAX_NEW_TOKENS — generation length for this HF one-model harness. Typical: 64|256
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

# CUDA_VISIBLE_DEVICES — GPU id(s). Example: 0
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# OUTPUT_FILE — summary json path. Default: {DRAFT_MODEL}/eval_one_model/summary.json
OUTPUT_FILE="${OUTPUT_FILE:-${DRAFT_MODEL}/eval_one_model/summary.json}"

# SKIP_BASELINE — skip non-speculative target-only baseline timing.
#   Options: 0 (default, run baseline+spec) | 1 (spec only)
SKIP_BASELINE="${SKIP_BASELINE:-0}"

export CUDA_VISIBLE_DEVICES

EXTRA=()
if [[ "${SKIP_BASELINE}" == "1" ]]; then
  EXTRA+=(--skip_baseline)
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"

echo "=== One-model real_hawk eval ==="
echo "  draft=${DRAFT_MODEL}"
echo "  dataset=${DATASET}  n=${NUM_PROMPTS}  K=${NUM_SPEC_TOKENS}"
echo "  output=${OUTPUT_FILE}"

python3 tools/hf_one_model_real_hawk_eval.py \
  --target_model "${TARGET_MODEL}" \
  --draft_model "${DRAFT_MODEL}" \
  --draft_model_config_path "${DRAFT_MODEL_CONFIG_PATH}" \
  --dataset "${DATASET}" \
  --num_prompts "${NUM_PROMPTS}" \
  --num_spec_tokens "${NUM_SPEC_TOKENS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --output_file "${OUTPUT_FILE}" \
  "${EXTRA[@]}"

echo "Results: ${OUTPUT_FILE}"
