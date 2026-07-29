#!/usr/bin/env bash
# Launch vLLM OpenAI server(s) for HuggingFaceTB/SmolVLM-256M-Instruct.
# Default: 4 GPU replicas (dp=4 style data-parallel serving).
#   GPU_NUM=4 BASE_PORT=6000 bash scripts/speculative/smolvlm/run_vllm_server.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${ROOT}/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

export MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolVLM-256M-Instruct}"
export MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH:-${MODEL_NAME}}"
export GPU_NUM="${GPU_NUM:-4}"
export BASE_PORT="${BASE_PORT:-6000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"

mkdir -p ./logs
LOG_TAG="$(echo "${MODEL_NAME}" | sed 's/\//-/g')"

echo "Starting ${GPU_NUM} vLLM server(s) for ${MODEL_LOCAL_PATH} (ports ${BASE_PORT}+)"
for i in $(seq 0 $((GPU_NUM - 1))); do
  PORT=$((BASE_PORT + i))
  LOG="./logs/${LOG_TAG}_${i}.log"
  echo "CUDA_VISIBLE_DEVICES=${i} -> port ${PORT}  log=${LOG}"
  CUDA_VISIBLE_DEVICES="${i}" nohup vllm serve "${MODEL_LOCAL_PATH}" \
    --port "${PORT}" \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --limit-mm-per-prompt '{"image":1}' \
    --enforce-eager \
    >"${LOG}" 2>&1 &
  echo $! >"./logs/${LOG_TAG}_${i}.pid"
done

echo "Waiting for server readiness on port ${BASE_PORT}..."
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${BASE_PORT}/v1/models" >/dev/null 2>&1; then
    echo "Server ready: http://127.0.0.1:${BASE_PORT}/v1"
    curl -s "http://127.0.0.1:${BASE_PORT}/v1/models"
    echo
    exit 0
  fi
  sleep 2
done

echo "ERROR: server did not become ready. Tail of log:" >&2
tail -n 60 "./logs/${LOG_TAG}_0.log" >&2 || true
exit 1
