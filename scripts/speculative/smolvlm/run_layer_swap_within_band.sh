#!/usr/bin/env bash
# Inference-only within-band aux-layer swap for a trained progressive Eagle3 draft.
#
# Does not retrain and does not rewrite the original checkpoint. Each swap gets
# a dir with a patched config.json + symlink to model.safetensors.
#
# Grid (train/HF ids; vLLM = id+1):
#   early {0, 4, 8} × mid {10, 12, 18} × late {20, 23, 25}   → 27
# Extra OFAT around trained [1, 14, 26], plus late 28.
# Full eval flattens (swap × temp × bench) across 4 GPUs (1 vLLM / GPU).
#
#   MODE=setup bash scripts/speculative/smolvlm/run_layer_swap_within_band.sh
#   MODE=full  bash scripts/speculative/smolvlm/run_layer_swap_within_band.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

export PATH="/home/hyang/miniconda3/envs/angel/bin:${PATH}"
PYTHON_BIN="${PYTHON_BIN:-/home/hyang/miniconda3/envs/angel/bin/python}"
# shellcheck disable=SC1091
source "${ROOT}/third_party/env.sh" 2>/dev/null || true
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

MODE="${MODE:-setup}"
SMOKE_EVAL="${SMOKE_EVAL:-0}"
SRC="${SRC:-output/smolvlm_256m_eagle3_progressive_nccl/checkpoint-66466}"
SWAP_ROOT="${SWAP_ROOT:-output/smolvlm_256m_eagle3_progressive_nccl/layer_swap}"
DRAFT_MODEL_CONFIG_PATH="${DRAFT_MODEL_CONFIG_PATH:-angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive.json}"
DESIGNATED_AUX="${DESIGNATED_AUX:-1,14,26}"
GRID_HELPER="${ROOT}/scripts/speculative/smolvlm/materialize_layer_swap_draft.py"

NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
# 0.9 only helps multi-seq KV cache. This sweep is MAX_NUM_SEQS=1 + acceptance length.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.25}"
FORCE="${FORCE:-0}"

LIST_FLAG="--list-all"
IDS_ARE_VLLM=0
if [[ "${MODE}" == "ofat_vllm" ]]; then
  LIST_FLAG="--list-ofat-vllm"
  IDS_ARE_VLLM=1
fi
mapfile -t GRID_ROWS < <("${PYTHON_BIN}" "${GRID_HELPER}" "${LIST_FLAG}")
declare -A SWAP_AUX=()
GRID_NAMES=()
for row in "${GRID_ROWS[@]}"; do
  name="${row%%$'\t'*}"
  aux="${row#*$'\t'}"
  SWAP_AUX["${name}"]="${aux}"
  GRID_NAMES+=("${name}")
done
GRID_NAMES_STR="${GRID_NAMES[*]}"

if [[ "${MODE}" == "smoke" ]]; then
  SWAP_NAMES="${SWAP_NAMES:-e4_m12_l23}"
  TEMPS="${TEMPS:-0}"
  DATASETS="${DATASETS:-lmms-lab/textvqa}"
  NUM_PROMPTS="${NUM_PROMPTS:-1}"
  OUTPUT_LEN="${OUTPUT_LEN:-32}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
elif [[ "${MODE}" == "setup" ]]; then
  SWAP_NAMES="${SWAP_NAMES:-${GRID_NAMES_STR}}"
  TEMPS="${TEMPS:-}"
  DATASETS="${DATASETS:-}"
  NUM_PROMPTS="${NUM_PROMPTS:-0}"
  OUTPUT_LEN="${OUTPUT_LEN:-0}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
elif [[ "${MODE}" == "full" ]]; then
  SWAP_NAMES="${SWAP_NAMES:-${GRID_NAMES_STR}}"
  TEMPS="${TEMPS:-0 1}"
  DATASETS="${DATASETS:-lmms-lab/textvqa echo840/OCRBench Lin-Chen/MMStar MMMU/MMMU HuggingFaceH4/MATH-500}"
  NUM_PROMPTS="${NUM_PROMPTS:-80}"
  OUTPUT_LEN="${OUTPUT_LEN:-1024}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
elif [[ "${MODE}" == "ofat_vllm" ]]; then
  # One-slot swaps on vLLM ids around (2,15,27). 10 configs × 2 temps × 5 benches.
  SWAP_NAMES="${SWAP_NAMES:-${GRID_NAMES_STR}}"
  TEMPS="${TEMPS:-0 1}"
  DATASETS="${DATASETS:-lmms-lab/textvqa echo840/OCRBench Lin-Chen/MMStar MMMU/MMMU HuggingFaceH4/MATH-500}"
  NUM_PROMPTS="${NUM_PROMPTS:-80}"
  OUTPUT_LEN="${OUTPUT_LEN:-1024}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
else
  echo "MODE must be smoke, setup, full, or ofat_vllm, got: ${MODE}" >&2
  exit 1
fi

materialize_one() {
  local name="$1"
  local aux="${SWAP_AUX[$name]:-}"
  local dst="${SWAP_ROOT}/${name}"
  if [[ -z "${aux}" ]]; then
    echo "unknown swap name: ${name}" >&2
    exit 1
  fi
  if [[ "${IDS_ARE_VLLM}" == "1" ]]; then
    "${PYTHON_BIN}" "${GRID_HELPER}" \
      --src "${SRC}" \
      --dst "${dst}" \
      --name "${name}" \
      --eagle_aux_hidden_state_layer_ids "${aux}" \
      --designated_aux_hidden_states_layer_ids "${DESIGNATED_AUX}"
  else
    "${PYTHON_BIN}" "${GRID_HELPER}" \
      --src "${SRC}" \
      --dst "${dst}" \
      --aux_hidden_states_layer_ids "${aux}" \
      --designated_aux_hidden_states_layer_ids "${DESIGNATED_AUX}"
  fi
  "${PYTHON_BIN}" scripts/speculative/smolvlm/prepare_draft_config_for_vllm_eval.py \
    --draft_model "${dst}" \
    --draft_model_config_path "${DRAFT_MODEL_CONFIG_PATH}"
}

echo "=== within-band layer swap (${MODE}) ==="
echo "src=${SRC}"
echo "swap_root=${SWAP_ROOT}"
echo "candidates: early={0,1,4,8} mid={10,12,14,18} late={20,23,25,26,28}"
echo "swaps=${SWAP_NAMES}"
echo "temps=${TEMPS:-none} datasets=${DATASETS:-none}"
echo "num_prompts=${NUM_PROMPTS} output_len=${OUTPUT_LEN}"

mkdir -p "${SWAP_ROOT}"
# shellcheck disable=SC2086
for name in ${SWAP_NAMES}; do
  echo
  echo "======== MATERIALIZE ${name} ========"
  materialize_one "${name}"
done

if [[ "${MODE}" == "setup" ]]; then
  echo
  echo "Materialized ${#GRID_NAMES[@]} swap dirs under ${SWAP_ROOT}"
  exit 0
fi

if [[ "${MODE}" == "smoke" && "${SMOKE_EVAL}" != "1" ]]; then
  echo
  echo "Smoke config check only. To run 1-prompt vLLM:"
  echo "  MODE=smoke SMOKE_EVAL=1 CUDA_VISIBLE_DEVICES=<free-gpu> \\"
  echo "    bash scripts/speculative/smolvlm/run_layer_swap_within_band.sh"
  exit 0
fi

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
# shellcheck disable=SC2206
TEMP_ARR=(${TEMPS})
# shellcheck disable=SC2206
DS_ARR=(${DATASETS})
# shellcheck disable=SC2206
NAME_ARR=(${SWAP_NAMES})

RUN_DIR="${SWAP_ROOT}/_run"
mkdir -p "${RUN_DIR}"
for ((i = 0; i < ${#GPUS[@]}; i++)); do
  : > "${RUN_DIR}/queue.${i}"
done

njobs=0
nskip=0
slot=0
for name in "${NAME_ARR[@]}"; do
  for TEMP in "${TEMP_ARR[@]}"; do
    for ds in "${DS_ARR[@]}"; do
      data_name="$(basename "${ds}")"
      metrics="${SWAP_ROOT}/${name}/eval_temp${TEMP}/${data_name}/acceptance_metrics.json"
      if [[ -f "${metrics}" && "${FORCE}" != "1" ]]; then
        nskip=$((nskip + 1))
        continue
      fi
      echo "${name} ${TEMP} ${ds}" >> "${RUN_DIR}/queue.${slot}"
      slot=$(( (slot + 1) % ${#GPUS[@]} ))
      njobs=$((njobs + 1))
    done
  done
done

echo
echo "Queued ${njobs} evals (skipped ${nskip} existing) across ${#GPUS[@]} GPUs"

run_gpu_queue() {
  local gpu="$1"
  local qfile="$2"
  local logfile="${RUN_DIR}/gpu${gpu}.log"
  local name temp ds data_name out_dir
  : > "${logfile}"
  while read -r name temp ds; do
    [[ -z "${name:-}" ]] && continue
    data_name="$(basename "${ds}")"
    out_dir="${SWAP_ROOT}/${name}/eval_temp${temp}/${data_name}"
    mkdir -p "${out_dir}"
    echo "$(date -Is) GPU ${gpu} START ${name} temp=${temp} ${ds}" | tee -a "${logfile}"
    if (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      PYTHON_BIN="${PYTHON_BIN}" \
      DRAFT_MODEL="${SWAP_ROOT}/${name}" \
      DRAFT_MODEL_CONFIG_PATH="${DRAFT_MODEL_CONFIG_PATH}" \
      DATASET="${ds}" \
      NUM_PROMPTS="${NUM_PROMPTS}" \
      OUTPUT_LEN="${OUTPUT_LEN}" \
      NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS}" \
      MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
      TEMP="${temp}" \
      OUTPUT_FILE="${out_dir}/results.json" \
      ACCEPTANCE_METRICS_FILE="${out_dir}/acceptance_metrics.json" \
        bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
    ) >>"${out_dir}/eval.log" 2>&1; then
      echo "$(date -Is) GPU ${gpu} DONE  ${name} temp=${temp} ${ds}" | tee -a "${logfile}"
    else
      echo "$(date -Is) GPU ${gpu} FAIL  ${name} temp=${temp} ${ds}" | tee -a "${logfile}"
    fi
  done < "${qfile}"
  echo "$(date -Is) GPU ${gpu} QUEUE EMPTY" | tee -a "${logfile}"
}

if [[ "${njobs}" -eq 0 ]]; then
  echo "Nothing to run."
  exit 0
fi

pids=()
for ((i = 0; i < ${#GPUS[@]}; i++)); do
  if [[ ! -s "${RUN_DIR}/queue.${i}" ]]; then
    continue
  fi
  echo "--- GPU ${GPUS[$i]} ---"
  sed 's/^/  /' "${RUN_DIR}/queue.${i}"
  run_gpu_queue "${GPUS[$i]}" "${RUN_DIR}/queue.${i}" &
  pids+=("$!")
done
printf "%s\n" "${pids[@]}" > "${RUN_DIR}/worker.pids"
echo "worker pids: ${pids[*]}"

ec=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    ec=1
  fi
done
echo "$(date -Is) all GPU queues finished ec=${ec}"
exit "${ec}"
