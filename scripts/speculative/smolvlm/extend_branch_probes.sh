#!/usr/bin/env bash
# Continue the 5k branch-change probes to a deeper common step.
#
#   TARGET=50000 bash scripts/speculative/smolvlm/extend_branch_probes.sh
#
# Each probe dir already holds checkpoint-38233, so HF resumes it: optimizer
# state, constant LR, data order and global_step all carry over. Raising
# MAX_STEPS is the only change.
#
# Why deeper: the paired branch-vs-baseline dCE is not stationary. On the
# completed top2-curr-r33k run it grows from -0.0058 over 33k-38k to -0.0277
# over 63k-66k, so a 5k read sees only ~21% of the asymptotic effect. A common
# stop at 50000 puts every arm at ~79%.
#
# All arms stop at the SAME step so they stay directly comparable, and the
# baseline has paired rows at every one of those steps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PATH=/home/hyang/miniconda3/envs/angel/bin:${PATH}

CONFIG_DIR=angelslim/compressor/speculative/train/configs
TARGET="${TARGET:-50000}"
TS=$(date +%Y%m%d_%H%M%S)
ARMS="${ARMS:-probeA-top2-w01 probeB-top3-w01 probeC-top3-w03}"

for arm in ${ARMS}; do
  name="branch-change-${arm}-r33k"
  out="my_angel/eagle/${name}"
  cfg="${CONFIG_DIR}/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-${arm}-r33k.json"

  last=$(ls -d "${out}"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1 || true)
  if [[ -z "${last}" ]]; then
    echo "ERROR: ${out} has no checkpoint to resume" >&2; exit 1
  fi
  if (( last >= TARGET )); then
    echo "======== SKIP ${name}: already at ${last} >= ${TARGET} ========"
    continue
  fi

  echo
  echo "################################################################"
  echo "# ${name}   (${last} -> ${TARGET})"
  echo "# $(date '+%F %T')"
  echo "################################################################"

  TRAIN_MODE=nccl \
  DRAFT_MODEL_CONFIG_PATH="${cfg}" \
  MAX_STEPS="${TARGET}" \
  SAVE_STRATEGY=steps SAVE_STEPS=5000 EVAL_STRATEGY=no \
  OUTPUT_DIR="${out}" \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh \
    2>&1 | tee "logs/${name}.extend${TARGET}.${TS}.log"
done

echo
echo "=== probe extension to ${TARGET} done (ts=${TS}) ==="
echo "compare with: python my_angel/compare_branch_probes.py"
