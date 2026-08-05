#!/usr/bin/env bash
# Launch vLLM OpenAI server(s) for HuggingFaceTB/SmolVLM-256M-Instruct.
#
# Multi-GPU data-parallel replicas (default 4):
#   GPU_NUM=4 BASE_PORT=6000 bash scripts/speculative/smolvlm/run_vllm_server.sh
#
# Single GPU (broken multi-process / only one free GPU):
#   GPU_NUM=1 CUDA_VISIBLE_DEVICES=0 bash scripts/speculative/smolvlm/run_vllm_server.sh
# Then generate with matching clients:
#   MAX_CLIENTS=1 NUM_THREADS=8 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
#
# Equivalence: 1 vs 4 servers changes throughput only (same model samples).
# Data gen takes ~GPU_NUM× longer on 1 GPU for the same dataset size — not
# related to training step counts.
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
# Comma-separated device list. If unset, uses 0..GPU_NUM-1.
CUDA_VISIBLE_DEVICES_LIST="${CUDA_VISIBLE_DEVICES:-}"

mkdir -p ./logs
LOG_TAG="$(echo "${MODEL_NAME}" | sed 's/\//-/g')"

if [[ -n "${CUDA_VISIBLE_DEVICES_LIST}" ]]; then
  IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES_LIST}"
  if [[ "${#DEVICES[@]}" -lt "${GPU_NUM}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES has ${#DEVICES[@]} devices but GPU_NUM=${GPU_NUM}" >&2
    exit 1
  fi
else
  DEVICES=()
  for i in $(seq 0 $((GPU_NUM - 1))); do
    DEVICES+=("${i}")
  done
fi

echo "Starting ${GPU_NUM} vLLM server(s) for ${MODEL_LOCAL_PATH} (ports ${BASE_PORT}+)"
for i in $(seq 0 $((GPU_NUM - 1))); do
  DEV="${DEVICES[$i]}"
  PORT=$((BASE_PORT + i))
  LOG="./logs/${LOG_TAG}_${i}.log"
  echo "CUDA_VISIBLE_DEVICES=${DEV} -> port ${PORT}  log=${LOG}"
  CUDA_VISIBLE_DEVICES="${DEV}" nohup vllm serve "${MODEL_LOCAL_PATH}" \
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
    echo "Generate data with MAX_CLIENTS=${GPU_NUM} (match server count), e.g.:"
    echo "  MAX_CLIENTS=${GPU_NUM} bash scripts/speculative/smolvlm/generate_data_for_target_model.sh"
    exit 0
  fi
  sleep 2
done

echo "ERROR: server did not become ready. Tail of log:" >&2
tail -n 60 "./logs/${LOG_TAG}_0.log" >&2 || true
exit 1
