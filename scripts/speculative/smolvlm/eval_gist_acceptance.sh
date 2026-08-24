#!/usr/bin/env bash
# Oracle-gist acceptance eval for SmolVLM EAGLE-3 drafts.
#
# Wraps tools/eval_smolvlm_eagle3_acceptance.py, the only evaluator that can
# supply the oracle gist. The vLLM path (eval_eagle3_vlm_batch.sh) cannot: it
# has no way to build a "what the target is about to say" vector at decode
# time, and a gist-conditioned draft has a 3-stream layer 0 that vLLM's 2-stream
# EAGLE-3 kernel will not load.
#
# GIST_MODE is read from the draft config so the oracle matches training.
# Override it only to measure a deliberate train/eval mismatch.
#
# Output (default): {DRAFT_MODEL}/eval_gist_acceptance/acceptance.json
#
# Examples:
#   DRAFT_MODEL=output/smolvlm_256m_eagle3_oracle_whole_gist_2ep/checkpoint-33232 \
#     DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-3.1-oracle-whole-gist.json \
#     bash scripts/speculative/smolvlm/eval_gist_acceptance.sh
#
#   # non-gist baseline on the same data, for the acceptance delta
#   DRAFT_MODEL=output/smolvlm_256m_eagle3_3.1/checkpoint-33232 \
#     DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-3.1.json \
#     bash scripts/speculative/smolvlm/eval_gist_acceptance.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${DRY_RUN:-0}" != "1" && -f "${ROOT}/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

PYTHON=${PYTHON:-python}

# DRAFT_MODEL — trained draft checkpoint dir (contains model.safetensors). Required,
# except for DRY_RUN=1 validation.
DRAFT_MODEL=${DRAFT_MODEL:-}
if [[ -z "${DRAFT_MODEL}" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "ERROR: set DRAFT_MODEL to a draft checkpoint directory" >&2
  exit 1
fi

# DRAFT_MODEL_CONFIG_PATH — the train-time draft JSON. Required; this is where
#   gist_conditioning / gist_mode / gist_embedding_dim are read from.
DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:?set DRAFT_MODEL_CONFIG_PATH}

TARGET_MODEL=${TARGET_MODEL:-HuggingFaceTB/SmolVLM-256M-Instruct}
DATA_PATH=${DATA_PATH:-dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl}

# NUM_SAMPLES — eval examples. 128 is the tool default; 512+ tightens the CI.
NUM_SAMPLES=${NUM_SAMPLES:-128}

# NUM_SPEC_TOKENS — speculative draft length the acceptance histogram covers.
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-7}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-smolvlm}
NUM_PROC=${NUM_PROC:-4}
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-1}

# GIST_MODE — "" (default) takes the mode from the draft config, which is what
#   keeps eval consistent with training. Set remaining|whole only on purpose.
GIST_MODE=${GIST_MODE:-}

OUTPUT_FILE=${OUTPUT_FILE:-${DRAFT_MODEL%/}/eval_gist_acceptance/acceptance.json}

ARGS=(
  tools/eval_smolvlm_eagle3_acceptance.py
  --target_model "${TARGET_MODEL}"
  --draft_model "${DRAFT_MODEL}"
  --draft_model_config_path "${DRAFT_MODEL_CONFIG_PATH}"
  --data_path "${DATA_PATH}"
  --output_file "${OUTPUT_FILE}"
  --num_samples "${NUM_SAMPLES}"
  --num_spec_tokens "${NUM_SPEC_TOKENS}"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --chat_template_type "${CHAT_TEMPLATE_TYPE}"
  --num_proc "${NUM_PROC}"
)
[[ -n "${GIST_MODE}" ]] && ARGS+=(--gist_mode "${GIST_MODE}")
[[ "${LOAD_FROM_CACHE_FILE}" == "1" ]] && ARGS+=(--load_from_cache_file)

echo "=== SmolVLM oracle-gist acceptance eval ==="
echo "  draft=${DRAFT_MODEL}"
echo "  config=${DRAFT_MODEL_CONFIG_PATH}"
echo "  data=${DATA_PATH}  num_samples=${NUM_SAMPLES}"
echo "  gist_mode=${GIST_MODE:-from-config}"
echo "  output=${OUTPUT_FILE}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: command validated, not launching acceptance eval."
  printf '  %q' "${PYTHON}" "${ARGS[@]}"
  echo
  exit 0
fi

"${PYTHON}" "${ARGS[@]}"
