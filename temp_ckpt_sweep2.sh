#!/usr/bin/env bash
# Same checkpoint sweep for change-w03 and the mix-fc baseline, after the
# change-w01 acceptance sweep releases the GPUs.
set -euo pipefail
cd /home/hyang/AngelSlim
export PATH=/home/hyang/miniconda3/envs/angel/bin:${PATH}
CFGD=angelslim/compressor/speculative/train/configs

while pgrep -f temp_ckpt_accept.sh >/dev/null; do sleep 60; done

RUN=branch-change-top1-w03 SAMPLES=200 bash temp_ckpt_diag.sh
RUN=smolvlm-256m-eagle3-banded-mix-fc-3.1 SAMPLES=200 bash temp_ckpt_diag.sh

RUN=branch-change-top1-w03 \
  CFG=${CFGD}/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top1-w03.json \
  bash temp_ckpt_accept.sh
RUN=smolvlm-256m-eagle3-banded-mix-fc-3.1 \
  CFG=${CFGD}/smolvlm-256m-eagle3-banded-mix-fc-3.1.json \
  bash temp_ckpt_accept.sh
echo "=== sweep2 done ==="
