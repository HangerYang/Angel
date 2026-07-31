#!/usr/bin/env bash
# Offline vLLM Eagle3 eval for SmolVLM (+ optional trained draft).
#
# Reads draft depth / aux layers from the checkpoint (1- or multi-layer):
#   num_hidden_layers
#   eagle_aux_hidden_state_layer_ids   (vLLM)
#   aux_hidden_states_layer_ids        (train; used to derive eagle_aux if needed)
#
# Examples:
#   # Eagle3 (requires a saved draft dir)
#   DRAFT_MODEL=output/smolvlm_256m_eagle3_online \
#     bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
#
#   # Baseline (no speculative decoding)
#   USE_EAGLE=0 bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
#
#   # Local jsonl + small smoke
#   DATASET=dataset/smolvlm_256m_target_gen/data_0-36.jsonl NUM_PROMPTS=4 \
#     DRAFT_MODEL=output/smolvlm_256m_eagle3_online \
#     bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
#
# See scripts/speculative/smolvlm/README.md § Eval + "Where to update vLLM".

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -f "${ROOT}/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

CONFIG_DIR=angelslim/compressor/speculative/train/configs
TARGET_MODEL="${TARGET_MODEL:-HuggingFaceTB/SmolVLM-256M-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-output/smolvlm_256m_eagle3_online}"
DRAFT_MODEL_CONFIG_PATH="${DRAFT_MODEL_CONFIG_PATH:-${CONFIG_DIR}/smolvlm-256m-eagle3.json}"
DATASET="${DATASET:-lmms-lab/textvqa}"
OUTPUT_FILE="${OUTPUT_FILE:-results/smolvlm-256m-eagle3-eval.jsonl}"
USE_EAGLE="${USE_EAGLE:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-80}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
TEMP="${TEMP:-0}"
TP="${TP:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEBUG_ARGS=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  DEBUG_ARGS+=(--debug)
fi
if [[ "${PDB:-0}" == "1" ]]; then
  DEBUG_ARGS+=(--pdb)
fi

export CUDA_VISIBLE_DEVICES

CMD=(
  python3 tools/vllm_offline_eagle3_vlm_batch.py
  --target_model "${TARGET_MODEL}"
  --dataset "${DATASET}"
  --num_prompts "${NUM_PROMPTS}"
  --temp "${TEMP}"
  --max_num_seqs "${MAX_NUM_SEQS}"
  --max_model_len "${MAX_MODEL_LEN}"
  --output_len "${OUTPUT_LEN}"
  --tp "${TP}"
  --output_file "${OUTPUT_FILE}"
)

if [[ "${USE_EAGLE}" == "1" ]]; then
  if [[ ! -d "${DRAFT_MODEL}" ]]; then
    echo "ERROR: DRAFT_MODEL directory not found: ${DRAFT_MODEL}" >&2
    echo "Train a draft first (and ensure it is saved), e.g.:" >&2
    echo "  SAVE_STRATEGY=epoch bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh" >&2
    exit 1
  fi

  PREPARE_ARGS=(
    --draft_model "${DRAFT_MODEL}"
  )
  if [[ -n "${DRAFT_MODEL_CONFIG_PATH}" && -f "${DRAFT_MODEL_CONFIG_PATH}" ]]; then
    PREPARE_ARGS+=(--draft_model_config_path "${DRAFT_MODEL_CONFIG_PATH}")
  fi
  python3 scripts/speculative/smolvlm/prepare_draft_config_for_vllm_eval.py \
    "${PREPARE_ARGS[@]}"

  CMD+=(
    --draft_model "${DRAFT_MODEL}"
    --use_eagle
    --num_spec_tokens "${NUM_SPEC_TOKENS}"
  )
else
  echo "USE_EAGLE=0 — baseline eval (no draft / speculative decoding)"
fi

if [[ ${#DEBUG_ARGS[@]} -gt 0 ]]; then
  CMD+=("${DEBUG_ARGS[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Results: ${OUTPUT_FILE}"
