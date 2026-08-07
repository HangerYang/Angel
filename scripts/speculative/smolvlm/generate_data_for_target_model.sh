#!/usr/bin/env bash
# Resample conversations with SmolVLM via the local vLLM OpenAI server.
#
# Default input: preprocessed ShareGPT 70k + LLaVA 70k mix (openai_vl),
# with absolute image paths under dataset/preprocessed/llava_images/.
#
# Multi-GPU (match run_vllm_server.sh GPU_NUM):
#   MAX_CLIENTS=4 NUM_THREADS=32 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
#
# Single-GPU server:
#   MAX_CLIENTS=1 NUM_THREADS=8 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

# Prefer this checkout over any incomplete site-packages angelslim
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${ROOT}/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

export DATA_NAME_OR_PATH="${DATA_NAME_OR_PATH:-${ROOT}/dataset/preprocessed/mixed_sharegpt_llava665k_70k70k.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/dataset/smolvlm_256m_target_gen_mixed_70k70k}"
export DATA_FORMAT="${DATA_FORMAT:-openai_vl}"
export DATA_SHARD_SIZE="${DATA_SHARD_SIZE:-50000}"
export BASE_PORT="${BASE_PORT:-6000}"
export NUM_THREADS="${NUM_THREADS:-32}"
# Match GPU_NUM used in run_vllm_server.sh (1 server → MAX_CLIENTS=1).
export MAX_CLIENTS="${MAX_CLIENTS:-4}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"

mkdir -p "${OUTPUT_DIR}"

echo "DATA_NAME_OR_PATH=${DATA_NAME_OR_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DATA_FORMAT=${DATA_FORMAT}  BASE_PORT=${BASE_PORT}  MAX_CLIENTS=${MAX_CLIENTS}"
echo "NUM_THREADS=${NUM_THREADS}  MAX_TOKENS=${MAX_TOKENS}  DATA_SHARD_SIZE=${DATA_SHARD_SIZE}"

if [[ ! -f "${DATA_NAME_OR_PATH}" ]]; then
  echo "ERROR: missing input jsonl: ${DATA_NAME_OR_PATH}" >&2
  exit 1
fi

# Fail fast if image paths still point off-box or files are missing.
python3 - <<'PY'
import json, os, sys
from pathlib import Path

path = Path(os.environ["DATA_NAME_OR_PATH"])
bad_prefix = "/home/nilay/"
n = n_img = miss = stale = 0
examples = []
with path.open() as f:
    for line in f:
        if not line.strip():
            continue
        n += 1
        row = json.loads(line)
        for turn in row.get("conversations", []) or []:
            content = turn.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not (isinstance(part, dict) and part.get("type") == "image"):
                    continue
                img = part.get("image")
                if not img:
                    continue
                n_img += 1
                if img.startswith(bad_prefix):
                    stale += 1
                    if len(examples) < 5:
                        examples.append(("stale", img))
                elif not Path(img).is_file():
                    miss += 1
                    if len(examples) < 5:
                        examples.append(("missing", img))
print(f"verified lines={n} images={n_img} missing={miss} stale_nilay={stale}")
if examples:
    for kind, p in examples:
        print(f"  {kind}: {p}", file=sys.stderr)
if stale or miss:
    sys.exit(1)
PY

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
