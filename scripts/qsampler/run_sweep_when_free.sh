#!/usr/bin/env bash
# Wait for the GPUs to free up, then run the num_queries sweep back to back.
#
# The EAGLE banded_mix_fc job owns all 4 GPUs; this parks until no
# train_eagle3_online.py process remains, then trains N=4, 8, 16 for 2 epochs each.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

echo "[queue] $(date -Is) waiting for train_eagle3_online.py to exit..."
while pgrep -f "tools/train_eagle3_online.py" > /dev/null; do
  sleep 60
done
echo "[queue] $(date -Is) GPUs free, starting Q-Sampler sweep"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

for N in ${SWEEP:-4 8 16}; do
  LOG="logs/qsampler_n${N}.log"
  echo "[queue] $(date -Is) === num_queries=${N} -> ${LOG} ==="
  START=$(date +%s)
  NUM_QUERIES="${N}" \
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}" \
  OUTPUT_DIR="output/qsampler-n${N}" \
    bash scripts/qsampler/train_qsampler.sh > "${LOG}" 2>&1
  RC=$?
  ELAPSED=$(( $(date +%s) - START ))
  echo "[queue] $(date -Is) num_queries=${N} exited rc=${RC} after ${ELAPSED}s"
  # A run that dies in under five minutes is a systematic fault, not a flake --
  # stop rather than burning the remaining sweep points on the same error.
  if [ "${RC}" -ne 0 ] && [ "${ELAPSED}" -lt 300 ]; then
    echo "[queue] $(date -Is) ABORTING sweep: N=${N} failed fast. Tail of ${LOG}:"
    tail -30 "${LOG}"
    exit 1
  fi
done
echo "[queue] $(date -Is) sweep complete"
