#!/usr/bin/env bash
# EAGLE 3.1 branch-distill eval grid. No vistoken runs here.
#
#   bash temp_eval_grid.sh
#
# Evaluates checkpoint-66466 of the rank-2 + curriculum runs (resumed from the
# baseline at 33233, so they still finish at 66466).
set -euo pipefail

cd /home/hyang/AngelSlim
export PATH=/home/hyang/miniconda3/envs/angel/bin:${PATH}
mkdir -p logs

TS=$(date +%Y%m%d_%H%M%S)
CONFIG_DIR=angelslim/compressor/speculative/train/configs
DATASETS="HuggingFaceH4/MATH-500 opendatalab/OmniDocBench MMMU/MMMU lmms-lab/COCO-Caption Lin-Chen/MMStar lmms-lab/chartqa lmms-lab/textvqa ai4math/mathvista"
FORCE="${FORCE:-0}"

run_cell() {
  local name="$1" cfg="$2" temp="$3" style="$4"
  local ckpt="my_angel/eagle/${name}/checkpoint-${CKPT_STEP:-66466}"
  local out="my_angel/eagle/${name}/rerun_atd/temp${temp}"
  [[ -f "${ckpt}/model.safetensors" ]] || { echo "ERROR: missing ${ckpt}/model.safetensors" >&2; exit 1; }
  echo
  echo "################################################################"
  echo "# ${name} | rerun_atd/temp${temp} | prompt_style=${style} | GPU 0"
  echo "# $(date '+%F %T')"
  echo "################################################################"
  CUDA_VISIBLE_DEVICES=0 \
  FORCE="${FORCE}" \
  TEMP="${temp}" NUM_PROMPTS=80 NUM_SPEC_TOKENS=4 MAX_NUM_SEQS=1 \
  PROMPT_STYLE="${style}" \
  RUN_NAME="${name}" \
  DRAFT_MODEL="${ckpt}" \
  DRAFT_MODEL_CONFIG_PATH="${cfg}" \
  OUT_ROOT="${out}" \
  DATASETS="${DATASETS}" \
  bash scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh \
    2>&1 | tee "logs/${name}.rerun_atd_temp${temp}.${TS}.log"
}

run_model() {
  local name="$1" cfg="$2"
  run_cell "${name}" "${cfg}" 0 answer_then_describe
  run_cell "${name}" "${cfg}" 1 answer_then_describe
}

run_model branch-change-top2-curr-r33k \
  "${CONFIG_DIR}/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top2-curr-r33k.json"
run_model branch-change-top2-curr-synth-r33k \
  "${CONFIG_DIR}/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top2-curr-synth-r33k.json"

echo
echo "=== branch eval grid done (ts=${TS}) ==="
