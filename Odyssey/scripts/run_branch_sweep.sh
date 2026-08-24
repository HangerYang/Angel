#!/usr/bin/env bash
# Odyssey: sweep resync branches over the vLLM eval, one branch per run.
#
#   bash Odyssey/scripts/run_branch_sweep.sh
#
# Env:
#   GAMMA          num_speculative_tokens (default 8; the plan asks for 8-16)
#   BRANCHES       space-separated subset of: baseline block stale_corr stale_skip recompute
#   DATASETS       benchmark ids (default: a checkable-ground-truth pair)
#   NUM_PROMPTS    prompts per benchmark (default 40)
#   TEMP           0 (greedy, default) or 1 (needed for distributional checks)
#   IGNORE_EOS     0 (default, natural generation) or 1 (equal work per run).
#                  1 makes wall-clock comparable but measures acceptance mostly
#                  on post-EOS text: on textvqa it drops accept 1.86 -> 1.27.
#   PROMPT_STYLE   raw (default) or answer_then_describe ("ATD").
#                  ATD appends "Then describe the image in detail to justify
#                  your answer", which lengthens MMStar output 8.8 -> 87.0 tok
#                  while keeping the answer at the front, so salvage has room
#                  to compound AND the answer stays extractable.
#   DRAFT_MODEL    draft checkpoint dir
#   GPU            single GPU index (default 0)
#   DRY_RUN        1 prints resolved commands and checks required files, no eval
#
# Each branch writes:
#   Odyssey/results/<tag>/<branch>/<bench>/{results.json,acceptance_metrics.json}
#   Odyssey/logs/<tag>/<branch>.jsonl.<pid>       # per-round events
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ODYSSEY="${ROOT}/Odyssey"
cd "${ROOT}"

source /home/hyang/miniconda3/etc/profile.d/conda.sh
conda activate angel
export PATH="/home/hyang/miniconda3/envs/angel/bin:${PATH}"
source third_party/env.sh 2>/dev/null || true

GAMMA="${GAMMA:-8}"
TEMP="${TEMP:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-40}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
GPU="${GPU:-0}"
DRAFT_MODEL="${DRAFT_MODEL:-my_angel/eagle/baseline_1layer/checkpoint-66466}"
BRANCHES="${BRANCHES:-baseline block stale_corr stale_skip recompute}"
# MATH-500 and MMStar both have checkable ground truth, per the plan's ask for
# final-answer correctness rather than embedding similarity alone.
DATASETS="${DATASETS:-HuggingFaceH4/MATH-500 Lin-Chen/MMStar}"

TAG="${TAG:-g${GAMMA}_t${TEMP}_$(date +%Y%m%d_%H%M%S)}"
RES_ROOT="${ODYSSEY}/results/${TAG}"
LOG_ROOT="${ODYSSEY}/logs/${TAG}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p "${RES_ROOT}" "${LOG_ROOT}"
fi

echo "=== Odyssey branch sweep ==="
echo "tag=${TAG}  gamma=${GAMMA}  temp=${TEMP}  prompts=${NUM_PROMPTS}"
echo "draft=${DRAFT_MODEL}"
echo "branches=${BRANCHES}"
echo "datasets=${DATASETS}"
echo "results=${RES_ROOT}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=1"
  test -f "third_party/patches/vllm-v0.25.0-odyssey-rejection-sampler.patch"
  test -f "scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh"
  PYTHONPATH="${ODYSSEY}:${PYTHONPATH:-}" python3 -m odyssey.test_branches >/dev/null
fi

for branch in ${BRANCHES}; do
  for ds in ${DATASETS}; do
    bench="$(basename "${ds}")"
    out_dir="${RES_ROOT}/${branch}/${bench}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "--- ${branch} / ${bench} ---"
      # "block" is vLLM's own joint-suffix verification on the stock Triton
      # kernels; everything else routes through the Odyssey python sampler.
      unset ODYSSEY_BRANCH
      unset ODYSSEY_REJECTION_METHOD
      if [[ "${branch}" == "block" ]]; then
        export ODYSSEY_REJECTION_METHOD=block
      else
        export ODYSSEY_BRANCH="${branch}"
      fi
      echo "DRY ${branch}/${bench}: ODYSSEY_BRANCH=${ODYSSEY_BRANCH:-} ODYSSEY_REJECTION_METHOD=${ODYSSEY_REJECTION_METHOD:-} DATASET=${ds} OUT=${out_dir}"
      continue
    fi
    mkdir -p "${out_dir}"
    if [[ -f "${out_dir}/acceptance_metrics.json" && "${FORCE:-0}" != "1" ]]; then
      echo "SKIP ${branch}/${bench} (exists)"
      continue
    fi
    echo "--- ${branch} / ${bench} ---"

    # "block" is vLLM's own joint-suffix verification on the stock Triton
    # kernels; everything else routes through the Odyssey python sampler.
    unset ODYSSEY_BRANCH
    unset ODYSSEY_REJECTION_METHOD
    if [[ "${branch}" == "block" ]]; then
      export ODYSSEY_REJECTION_METHOD=block
    else
      export ODYSSEY_BRANCH="${branch}"
    fi


    CUDA_VISIBLE_DEVICES="${GPU}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    ODYSSEY_ROOT="${ODYSSEY}" \
    ODYSSEY_EVENT_LOG="${LOG_ROOT}/${branch}.jsonl" \
    IGNORE_EOS="${IGNORE_EOS:-0}" \
    DRAFT_MODEL="${DRAFT_MODEL}" \
    DRAFT_MODEL_CONFIG_PATH= \
    DATASET="${ds}" \
    NUM_PROMPTS="${NUM_PROMPTS}" \
    OUTPUT_LEN="${OUTPUT_LEN}" \
    NUM_SPEC_TOKENS="${GAMMA}" \
    MAX_NUM_SEQS=1 \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}" \
    TEMP="${TEMP}" \
    PROMPT_STYLE="${PROMPT_STYLE:-raw}" \
    OUTPUT_FILE="${out_dir}/results.json" \
    ACCEPTANCE_METRICS_FILE="${out_dir}/acceptance_metrics.json" \
      bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh \
      >"${LOG_ROOT}/${branch}.${bench}.log" 2>&1 \
      || echo "FAIL ${branch}/${bench} (see ${LOG_ROOT}/${branch}.${bench}.log)"
  done
done

unset ODYSSEY_BRANCH ODYSSEY_REJECTION_METHOD

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo
  echo "dry run complete"
  exit 0
fi

echo
echo "=== branch 3 control: offline re-score of corrected tails ==="
if compgen -G "${LOG_ROOT}/recompute.jsonl.*" >/dev/null; then
  PYTHONPATH="${ODYSSEY}:${PYTHONPATH:-}" python3 -m odyssey.rescore     --events "${LOG_ROOT}/recompute.jsonl.*"     --out "${RES_ROOT}/rescored.jsonl"     --max_events "${MAX_RESCORE:-2000}"     || echo "rescore FAILED"
else
  echo "no recompute events; skipping"
fi

echo
echo "=== analysis ==="
PYTHONPATH="${ODYSSEY}:${PYTHONPATH:-}" python3 -m odyssey.analyze   --events "${LOG_ROOT}/*.jsonl.*"   --results_root "${RES_ROOT}"   --rescored "${RES_ROOT}/rescored.jsonl"   --out "${RES_ROOT}/report.json"

echo
echo "done: ${RES_ROOT}/report.json"
