#!/usr/bin/env bash
# Manual data-parallel eval: different benchmarks on different GPUs.
# At most ONE vLLM job per GPU (queued). Acceptance length unchanged vs serial.
#
#   TEMP=1 \
#   DRAFT_MODEL=output/progressive_layer_group_tests/layers_1_15_23_attn_match_img_w03 \
#   DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-layers-1-15-23-attn-match-w03.json \
#   OUT_ROOT=output/progressive_layer_group_tests/layers_1_15_23_attn_match_img_w03/eval_temp1 \
#   bash scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

source /home/hyang/miniconda3/etc/profile.d/conda.sh
conda activate angel
export PATH="/home/hyang/miniconda3/envs/angel/bin:${PATH}"
# shellcheck disable=SC1091
source third_party/env.sh 2>/dev/null || true

DRAFT_MODEL="${DRAFT_MODEL:-output/progressive_layer_group_tests/layers_1_15_23_attn_match_img_w03}"
DRAFT_MODEL_CONFIG_PATH="${DRAFT_MODEL_CONFIG_PATH:-angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-layers-1-15-23-attn-match-w03.json}"
RUN_NAME="${RUN_NAME:-$(basename "${DRAFT_MODEL%/}")}"
TEMP="${TEMP:-1}"
OUT_ROOT="${OUT_ROOT:-${DRAFT_MODEL%/}/eval_temp${TEMP}}"
NUM_PROMPTS="${NUM_PROMPTS:-80}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPUS_CSV="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"

DATASETS_DEFAULT=(
  lmms-lab/ChartQA
  lmms-lab/VQAv2
  lmms-lab/GQA
  lmms-lab/ScienceQA
  lmms-lab/textvqa
  lmms-lab/MME
  lmms-lab/SEED-Bench
  lmms-lab-encoder/MMVet
  AI4Math/MathVista
  lmms-lab/MMBench
)
if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASETS_ARR=(${DATASETS})
else
  DATASETS_ARR=("${DATASETS_DEFAULT[@]}")
fi

mkdir -p "${OUT_ROOT}"
LOG_DIR="${OUT_ROOT}/_logs"
mkdir -p "${LOG_DIR}"

echo "=== DP eval TEMP=${TEMP} (1 job/GPU queue) ==="
echo "draft=${DRAFT_MODEL}"
echo "out=${OUT_ROOT}"
echo "gpus=${GPUS_CSV}"

# Per-GPU serial queues: round-robin datasets onto GPUs, then run each GPU's queue.
declare -a QUEUE=()
for ((i = 0; i < ${#GPUS[@]}; i++)); do
  QUEUE[$i]=""
done

idx=0
pending=()
metrics_files=()
for ds in "${DATASETS_ARR[@]}"; do
  data_name="$(basename "${ds}")"
  out_dir="${OUT_ROOT}/${data_name}"
  mkdir -p "${out_dir}"
  metrics_file="${out_dir}/acceptance_metrics.json"
  metrics_files+=("${metrics_file}")
  if [[ -f "${metrics_file}" && "${FORCE}" != "1" ]]; then
    echo "SKIP ${ds} (exists)"
    continue
  fi
  slot=$((idx % ${#GPUS[@]}))
  QUEUE[$slot]+="${ds}"$'\n'
  idx=$((idx + 1))
  pending+=("${ds}")
done

run_gpu_queue() {
  local gpu="$1"
  local list="$2"
  local ds data_name out_dir metrics_file log_file
  while IFS= read -r ds; do
    [[ -z "${ds}" ]] && continue
    data_name="$(basename "${ds}")"
    out_dir="${OUT_ROOT}/${data_name}"
    metrics_file="${out_dir}/acceptance_metrics.json"
    log_file="${LOG_DIR}/${data_name}.log"
    echo "GPU ${gpu}: START ${ds}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      PYTHON_BIN="${PYTHON_BIN}" \
      DRAFT_MODEL="${DRAFT_MODEL}" \
      DRAFT_MODEL_CONFIG_PATH="${DRAFT_MODEL_CONFIG_PATH}" \
      DATASET="${ds}" \
      NUM_PROMPTS="${NUM_PROMPTS}" \
      OUTPUT_LEN="${OUTPUT_LEN}" \
      NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS}" \
      MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
      TEMP="${TEMP}" \
      OUTPUT_FILE="${out_dir}/results.json" \
      ACCEPTANCE_METRICS_FILE="${metrics_file}" \
        bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
    ) >"${log_file}" 2>&1
    echo "GPU ${gpu}: DONE ${ds}"
  done <<< "${list}"
}

pids=()
for ((i = 0; i < ${#GPUS[@]}; i++)); do
  [[ -z "${QUEUE[$i]}" ]] && continue
  echo "--- GPU ${GPUS[$i]} queue ---"
  echo "${QUEUE[$i]}" | sed '/^$/d' | sed 's/^/  /'
  run_gpu_queue "${GPUS[$i]}" "${QUEUE[$i]}" &
  pids+=("$!")
done

ec=0
for pid in "${pids[@]:-}"; do
  if ! wait "${pid}"; then
    echo "FAIL pid=${pid}"
    ec=1
  fi
done

"${PYTHON_BIN}" - "${RUN_NAME}" "${metrics_files[@]}" <<'PYSUMMARY'
import json
import sys
from pathlib import Path

run_name = sys.argv[1]
paths = [Path(p) for p in sys.argv[2:]]
print()
print(f"| bench | {run_name} acceptance length | draft acceptance | drafts |")
print("|---|---:|---:|---:|")
for path in paths:
    if not path.exists():
        print(f"| {path.parent.name} | MISSING | n/a | n/a |")
        continue
    m = json.loads(path.read_text())
    bench = Path(str(m.get("dataset", path.parent.name))).name
    acc_len = m.get("mean_acceptance_length")
    draft_rate = m.get("draft_acceptance_rate")
    drafts = m.get("num_drafts")
    acc_len_s = f"{acc_len:.4f}" if isinstance(acc_len, (int, float)) else "n/a"
    draft_rate_s = f"{100 * draft_rate:.2f}%" if isinstance(draft_rate, (int, float)) else "n/a"
    drafts_s = str(drafts) if drafts is not None else "n/a"
    print(f"| {bench} | {acc_len_s} | {draft_rate_s} | {drafts_s} |")
PYSUMMARY

exit "${ec}"
