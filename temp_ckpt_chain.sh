#!/usr/bin/env bash
# Acceptance every 15k steps (plus the final checkpoint) for the two
# branch-change runs, with matching held-out diagnostics for w03.
set -euo pipefail
cd /home/hyang/AngelSlim
export PATH=/home/hyang/miniconda3/envs/angel/bin:${PATH}
CFGD=angelslim/compressor/speculative/train/configs

while pgrep -f eval_acceptance_suite_dp.sh >/dev/null; do sleep 30; done

EVERY=15000 RUN=branch-change-top1-w01 \
  CFG=${CFGD}/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top1-w01.json \
  bash temp_ckpt_accept.sh

EVERY=15000 RUN=branch-change-top1-w03 SAMPLES=200 bash temp_ckpt_diag.sh
EVERY=15000 RUN=branch-change-top1-w03 \
  CFG=${CFGD}/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top1-w03.json \
  bash temp_ckpt_accept.sh
echo "=== chain done: $(date '+%F %T') ==="
