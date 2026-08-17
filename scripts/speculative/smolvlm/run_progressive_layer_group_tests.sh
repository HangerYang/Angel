#!/usr/bin/env bash
# Train and evaluate two progressive Eagle layer-group variants.
#
# Groups:
#   layers_15_23_27: aux_hidden_states_layer_ids=[15,23,27]
#   layers_1_15_23:  aux_hidden_states_layer_ids=[1,15,23]
#
# Full run:
#   bash scripts/speculative/smolvlm/run_progressive_layer_group_tests.sh
#
# Smoke run:
#   MODE=smoke bash scripts/speculative/smolvlm/run_progressive_layer_group_tests.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

export PATH="/home/hyang/miniconda3/envs/angel/bin:${PATH}"

MODE="${MODE:-full}"
CONFIG_DIR="angelslim/compressor/speculative/train/configs"

RUN_IDS=(
  layers_15_23_27
  layers_1_15_23
)

BASELINE_RUN_ID="${BASELINE_RUN_ID:-progressive_nccl_40k}"
BASELINE_DRAFT_MODEL="${BASELINE_DRAFT_MODEL:-output/smolvlm_256m_eagle3_progressive_nccl/checkpoint-40000}"
BASELINE_CONFIG_PATH="${BASELINE_CONFIG_PATH:-${CONFIG_DIR}/smolvlm-256m-eagle3-progressive.json}"

config_for() {
  case "$1" in
    layers_15_23_27)
      echo "${CONFIG_DIR}/smolvlm-256m-eagle3-progressive-layers-15-23-27.json"
      ;;
    layers_1_15_23)
      echo "${CONFIG_DIR}/smolvlm-256m-eagle3-progressive-layers-1-15-23.json"
      ;;
    *)
      echo "bad run id: $1" >&2
      exit 1
      ;;
  esac
}

if [[ "${MODE}" == "smoke" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-output/progressive_layer_group_smoke}"
  TRAIN_MODE="${TRAIN_MODE:-python}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  MAX_STEPS="${MAX_STEPS:-1}"
  SAMPLE_NUM="${SAMPLE_NUM:-1}"
  SAVE_STEPS="${SAVE_STEPS:-1}"
  MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
  OUTPUT_LEN="${OUTPUT_LEN:-32}"
  NUM_PROMPTS="${NUM_PROMPTS:-1}"
elif [[ "${MODE}" == "full" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-output/progressive_layer_group_tests}"
  TRAIN_MODE="${TRAIN_MODE:-nccl}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  MAX_STEPS="${MAX_STEPS:-40000}"
  SAMPLE_NUM="${SAMPLE_NUM:-}"
  SAVE_STEPS="${SAVE_STEPS:-5000}"
  MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
  OUTPUT_LEN="${OUTPUT_LEN:-1024}"
  NUM_PROMPTS="${NUM_PROMPTS:-80}"
else
  echo "MODE must be smoke or full, got: ${MODE}" >&2
  exit 1
fi

SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
DATASETS="${DATASETS:-lmms-lab/textvqa MMMU/MMMU Lin-Chen/MMStar opendatalab/OmniDocBench HuggingFaceH4/MATH-500 lmms-lab/COCO-Caption}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
TEMP="${TEMP:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TARGET_HS_WARMUP_STEPS="${TARGET_HS_WARMUP_STEPS:-0}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-true}"

mkdir -p "${OUTPUT_ROOT}"

echo "=== Progressive Eagle layer-group tests ==="
echo "mode=${MODE}"
echo "output_root=${OUTPUT_ROOT}"
echo "max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS}"
echo "datasets=${DATASETS}"
echo "num_prompts=${NUM_PROMPTS} output_len=${OUTPUT_LEN} temp=${TEMP} K=${NUM_SPEC_TOKENS}"

echo
echo "======== EVAL ${BASELINE_RUN_ID} ========"
echo "draft=${BASELINE_DRAFT_MODEL}"
echo "config=${BASELINE_CONFIG_PATH}"
DRAFT_MODEL="${BASELINE_DRAFT_MODEL}" \
DRAFT_MODEL_CONFIG_PATH="${BASELINE_CONFIG_PATH}" \
RUN_NAME="${BASELINE_RUN_ID}" \
OUT_ROOT="${OUTPUT_ROOT}/${BASELINE_RUN_ID}/eval_temp0" \
DATASETS="${DATASETS}" \
NUM_PROMPTS="${NUM_PROMPTS}" \
OUTPUT_LEN="${OUTPUT_LEN}" \
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
TEMP="${TEMP}" \
PYTHON_BIN="${PYTHON_BIN}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
  bash scripts/speculative/smolvlm/eval_acceptance_suite.sh

for run_id in "${RUN_IDS[@]}"; do
  config_path="$(config_for "${run_id}")"
  output_dir="${OUTPUT_ROOT}/${run_id}"
  eval_root="${output_dir}/eval_temp0"

  echo
  echo "======== TRAIN ${run_id} ========"
  echo "config=${config_path}"
  echo "output=${output_dir}"

  train_env=(
    TRAIN_MODE="${TRAIN_MODE}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    DRAFT_MODEL_CONFIG_PATH="${config_path}"
    OUTPUT_DIR="${output_dir}"
    MAX_STEPS="${MAX_STEPS}"
    SAVE_STRATEGY="${SAVE_STRATEGY}"
    SAVE_STEPS="${SAVE_STEPS}"
    MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH}"
    TARGET_HS_WARMUP_STEPS="${TARGET_HS_WARMUP_STEPS}"
    LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE}"
  )
  [[ -n "${SAMPLE_NUM}" ]] && train_env+=(SAMPLE_NUM="${SAMPLE_NUM}")

  env "${train_env[@]}" bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

  echo
  echo "======== EVAL ${run_id} ========"
  DRAFT_MODEL="${output_dir}" \
  DRAFT_MODEL_CONFIG_PATH="${config_path}" \
  RUN_NAME="${run_id}" \
  OUT_ROOT="${eval_root}" \
  DATASETS="${DATASETS}" \
  NUM_PROMPTS="${NUM_PROMPTS}" \
  OUTPUT_LEN="${OUTPUT_LEN}" \
  NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  TEMP="${TEMP}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
    bash scripts/speculative/smolvlm/eval_acceptance_suite.sh
done
