#!/usr/bin/env bash
# Acceptance length at every saved checkpoint (temp 0, 8-dataset atd suite).
set -euo pipefail
cd /home/hyang/AngelSlim
export PATH=/home/hyang/miniconda3/envs/angel/bin:${PATH}

RUN=${RUN:-branch-change-top1-w01}
CFG=${CFG:-angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top1-w01.json}
DATASETS="HuggingFaceH4/MATH-500 opendatalab/OmniDocBench MMMU/MMMU lmms-lab/COCO-Caption Lin-Chen/MMStar lmms-lab/chartqa lmms-lab/textvqa ai4math/mathvista"

STEPS=$(ls -d my_angel/eagle/${RUN}/checkpoint-* | sed 's/.*checkpoint-//' | sort -n)
# EVERY: keep one checkpoint per this many steps (the final one always stays).
EVERY=${EVERY:-10000}
LAST=$(echo "${STEPS}" | tail -1)
STEPS=$(for s in ${STEPS}; do
  if [[ $((s % EVERY)) -eq 0 || "${s}" == "${LAST}" ]]; then echo "${s}"; fi
done)
for s in ${STEPS}; do
  out="my_angel/eagle/${RUN}/ckpt_sweep/step_${s}"
  if [[ -f "${out}/mathvista/acceptance_metrics.json" ]]; then echo "skip ${s}"; continue; fi
  echo "### step ${s} -> ${out}  $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  TEMP=0 NUM_PROMPTS=80 NUM_SPEC_TOKENS=4 MAX_NUM_SEQS=1 \
  PROMPT_STYLE=answer_then_describe \
  RUN_NAME="${RUN}-step${s}" \
  DRAFT_MODEL="my_angel/eagle/${RUN}/checkpoint-${s}" \
  DRAFT_MODEL_CONFIG_PATH="${CFG}" \
  OUT_ROOT="${out}" \
  DATASETS="${DATASETS}" \
  bash scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh \
    > "my_angel/eagle/${RUN}/ckpt_sweep/step_${s}.log" 2>&1
done
echo "=== checkpoint acceptance sweep done ==="
