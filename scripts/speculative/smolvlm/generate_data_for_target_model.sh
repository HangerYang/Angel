#!/usr/bin/env bash
# Resample conversations with SmolVLM via the local vLLM OpenAI server.
# Uses the mixed text/VL JSONL (openai_vl format) by default.
# Default: talk to 4 server replicas (ports BASE_PORT .. BASE_PORT+3):
#   MAX_CLIENTS=4 NUM_THREADS=32 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
#
# Single-GPU server:
#   MAX_CLIENTS=1 NUM_THREADS=8 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
# (Same outputs; slower wall-clock — not a training-step multiplier.)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

# Prefer this checkout over any incomplete site-packages angelslim
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${ROOT}/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

export DATA_NAME_OR_PATH="${DATA_NAME_OR_PATH:-${ROOT}/dataset/mixed_text_vl_36/mixed_text_vl_36.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/dataset/smolvlm_256m_target_gen}"
export DATA_FORMAT="${DATA_FORMAT:-openai_vl}"
export DATA_SHARD_SIZE="${DATA_SHARD_SIZE:-50000}"
export BASE_PORT="${BASE_PORT:-6000}"
export NUM_THREADS="${NUM_THREADS:-32}"
# Match GPU_NUM used in run_vllm_server.sh (1 server → MAX_CLIENTS=1).
export MAX_CLIENTS="${MAX_CLIENTS:-4}"
export MAX_TOKENS="${MAX_TOKENS:-512}"

mkdir -p "${OUTPUT_DIR}"

echo "DATA_NAME_OR_PATH=${DATA_NAME_OR_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DATA_FORMAT=${DATA_FORMAT}  BASE_PORT=${BASE_PORT}"

# Sanity: server must be up
if ! curl -sf "http://127.0.0.1:${BASE_PORT}/v1/models" >/dev/null; then
  echo "ERROR: no vLLM server on port ${BASE_PORT}. Run:" >&2
  echo "  bash scripts/speculative/smolvlm/run_vllm_server.sh" >&2
  exit 1
fi

python3 ./tools/generate_data_for_target_model.py \
  --data_name_or_path "${DATA_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --data_format "${DATA_FORMAT}" \
  --data_shard_size "${DATA_SHARD_SIZE}" \
  --base_port "${BASE_PORT}" \
  --num_threads "${NUM_THREADS}" \
  --max_clients "${MAX_CLIENTS}" \
  --max_tokens "${MAX_TOKENS}"

echo "Done. Outputs under ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}" || true
