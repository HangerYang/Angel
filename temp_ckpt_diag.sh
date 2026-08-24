#!/usr/bin/env bash
# Branch-vs-main diagnostic across every saved checkpoint of one run.
# 4 checkpoints at a time, one GPU each.
set -euo pipefail
cd /home/hyang/AngelSlim
export PATH=/home/hyang/miniconda3/envs/angel/bin:${PATH}
export PYTHONPATH=/home/hyang/AngelSlim

RUN=${RUN:-branch-change-top1-w01}
CFG=${CFG:-angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-banded-mix-fc-3.1.json}
SAMPLES=${SAMPLES:-200}
OUTDIR=${OUTDIR:-$HOME/tmp/ckpt_diag/${RUN}}
mkdir -p "${OUTDIR}"

STEPS=$(ls -d my_angel/eagle/${RUN}/checkpoint-* | sed 's/.*checkpoint-//' | sort -n)
# EVERY: keep one checkpoint per this many steps (the final one always stays).
EVERY=${EVERY:-10000}
LAST=$(echo "${STEPS}" | tail -1)
STEPS=$(for s in ${STEPS}; do
  if [[ $((s % EVERY)) -eq 0 || "${s}" == "${LAST}" ]]; then echo "${s}"; fi
done)
gpu=0
for s in ${STEPS}; do
  out="${OUTDIR}/step_${s}.json"
  [[ -f "${out}" ]] && { echo "skip ${s}"; continue; }
  CUDA_VISIBLE_DEVICES=${gpu} python tools/branch_rank_diagnostic.py \
    --draft_model_config_path "${CFG}" \
    --draft_ckpt my_angel/eagle/${RUN}/checkpoint-${s} \
    --vocab_cache my_angel/eagle/${RUN}/vocab_mapping_cache.pt \
    --tag "${RUN}@${s}" --num_samples "${SAMPLES}" --groups 2 \
    --scratch_dir "$HOME/tmp/branch_diag_gpu${gpu}" \
    --out "${out}" > "${OUTDIR}/step_${s}.log" 2>&1 &
  gpu=$(( (gpu + 1) % 4 ))
  [[ ${gpu} -eq 0 ]] && wait
done
wait
echo "=== checkpoint diagnostics done: ${OUTDIR} ==="
