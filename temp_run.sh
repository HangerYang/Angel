#!/usr/bin/env bash
# EAGLE 3.1 rank-2 branch-change runs, continued from the branch-free
# banded_mix_fc_3.1 baseline at step 33233. Each run dir was seeded with that
# checkpoint, so HF resumes it: optimizer state, constant LR, data order and
# global_step all carry over, and training ends at 66466 like every other run.
#
#   bash temp_run.sh
#
# Branch weight: 0 through 33233, linear to 0.1 by 38233, flat after.
set -euo pipefail

cd /home/hyang/AngelSlim
export PATH=/home/hyang/miniconda3/envs/angel/bin:${PATH}
mkdir -p logs my_angel/eagle

TS=$(date +%Y%m%d_%H%M%S)
CONFIG_DIR=angelslim/compressor/speculative/train/configs

run_train() {
  local name="$1" cfg="$2"
  local out="my_angel/eagle/${name}"
  [[ -d "${out}/checkpoint-33233" ]] || {
    echo "ERROR: ${out} is not seeded with checkpoint-33233" >&2; exit 1; }
  echo
  echo "################################################################"
  echo "# train ${name} -> ${out}   (resume from 33233 -> 66466)"
  echo "# config: ${cfg}"
  echo "# $(date '+%F %T')"
  echo "################################################################"
  TRAIN_MODE=nccl \
  DRAFT_MODEL_CONFIG_PATH="${cfg}" \
  SAVE_STRATEGY=steps SAVE_STEPS=5000 EVAL_STRATEGY=no \
  OUTPUT_DIR="${out}" \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh \
    2>&1 | tee "logs/${name}.train.${TS}.log"
}

run_train branch-change-top2-curr-r33k \
  "${CONFIG_DIR}/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top2-curr-r33k.json"
run_train branch-change-top2-curr-synth-r33k \
  "${CONFIG_DIR}/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top2-curr-synth-r33k.json"

echo
echo "=== branch experiment queue done (ts=${TS}) ==="
