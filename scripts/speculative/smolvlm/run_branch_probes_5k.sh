#!/usr/bin/env bash
# Three 5k-step branch-change probes, all continued from the branch-free
# banded_mix_fc_3.1 baseline at step 33233 and all with NO weight ramp
# (branch_distill_ramp_steps=0), so full weight applies from the first step.
#
#   bash scripts/speculative/smolvlm/run_branch_probes_5k.sh
#
#   A  target_top_k=2  w=0.1   isolates the ramp vs top2-curr-r33k
#   B  target_top_k=3  w=0.1   adds rank-3 supervision on top of A
#   C  target_top_k=3  w=0.3   B at 3x weight
#
# Each run dir is seeded with the baseline's checkpoint-33233, so HF resumes it:
# optimizer state, constant LR, data order and global_step all carry over.
# MAX_STEPS=38233 stops each probe after exactly 5000 optimizer steps.
#
# Judge these on train CE -- mean(train/ploss_0..6) -- not on the reported
# `loss`, which folds in branch_weight * branch_loss and is not comparable
# across different w. See compare_branch_probes.py.
#
# Set SKIP_SEED=1 to reuse run dirs that are already seeded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PATH=/home/hyang/miniconda3/envs/angel/bin:${PATH}
mkdir -p logs my_angel/eagle

CONFIG_DIR=angelslim/compressor/speculative/train/configs
SEED_CKPT="${SEED_CKPT:-my_angel/eagle/smolvlm-256m-eagle3-banded-mix-fc-3.1/checkpoint-33233}"
MAX_STEPS="${MAX_STEPS:-38233}"
TS=$(date +%Y%m%d_%H%M%S)

[[ -f "${SEED_CKPT}/model.safetensors" ]] || {
  echo "ERROR: seed checkpoint missing: ${SEED_CKPT}" >&2; exit 1; }

run_probe() {
  local name="$1" cfg="$2"
  local out="my_angel/eagle/${name}"

  if [[ "${SKIP_SEED:-0}" != "1" && ! -d "${out}/checkpoint-33233" ]]; then
    echo "seeding ${out}/checkpoint-33233 from ${SEED_CKPT}"
    mkdir -p "${out}"
    cp -r "${SEED_CKPT}" "${out}/checkpoint-33233"
  fi
  [[ -d "${out}/checkpoint-33233" ]] || {
    echo "ERROR: ${out} is not seeded with checkpoint-33233" >&2; exit 1; }

  echo
  echo "################################################################"
  echo "# ${name}   (33233 -> ${MAX_STEPS}, no ramp)"
  echo "# config: ${cfg}"
  echo "# $(date '+%F %T')"
  echo "################################################################"
  python3 -c "
import json; c=json.load(open('${cfg}'))
print('  ', {k: c[k] for k in c if k.startswith('branch_distill')})"

  TRAIN_MODE=nccl \
  DRAFT_MODEL_CONFIG_PATH="${cfg}" \
  MAX_STEPS="${MAX_STEPS}" \
  SAVE_STRATEGY=steps SAVE_STEPS=5000 EVAL_STRATEGY=no \
  OUTPUT_DIR="${out}" \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh \
    2>&1 | tee "logs/${name}.train.${TS}.log"
}

run_probe branch-change-probeA-top2-w01-r33k \
  "${CONFIG_DIR}/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-probeA-top2-w01-r33k.json"
run_probe branch-change-probeB-top3-w01-r33k \
  "${CONFIG_DIR}/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-probeB-top3-w01-r33k.json"
run_probe branch-change-probeC-top3-w03-r33k \
  "${CONFIG_DIR}/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-probeC-top3-w03-r33k.json"

echo
echo "=== branch probes done (ts=${TS}) ==="
echo "compare with: python my_angel/compare_branch_probes.py"
