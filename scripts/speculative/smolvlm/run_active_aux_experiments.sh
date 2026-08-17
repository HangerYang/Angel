#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PATH="/home/hyang/miniconda3/envs/angel/bin:${PATH}"
MODE="${MODE:-smoke}"
CONFIG_DIR=angelslim/compressor/speculative/train/configs
RUN_IDS=(progressive_threshold hawk_feature_match_from_warmup progressive_skew_kl_70p30q)

config_for(){ case "$1" in progressive_threshold) echo ${CONFIG_DIR}/smolvlm-256m-eagle3-progressive-threshold.json;; hawk_feature_match_from_warmup) echo ${CONFIG_DIR}/smolvlm-256m-hawk-feature-match.json;; progressive_skew_kl_70p30q) echo ${CONFIG_DIR}/smolvlm-256m-eagle3-progressive-skew-kl-70p30q.json;; *) echo bad run $1 >&2; exit 1;; esac; }
warmstart_for(){ case "$1" in hawk_feature_match_from_warmup) echo ${HAWK_WARMUP_CKPT:-output/smolvlm_256m_hawk_warmup/checkpoint-66466};; *) echo;; esac; }

if [[ ${MODE} == smoke ]]; then
  OUTPUT_ROOT=${OUTPUT_ROOT:-output/aux_experiment_smoke}; TRAIN_MODE=${TRAIN_MODE:-python}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}; MAX_STEPS=${MAX_STEPS:-1}; SAMPLE_NUM=${SAMPLE_NUM:-1}; SAVE_STEPS=${SAVE_STEPS:-1}; MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}; OUTPUT_LEN=${OUTPUT_LEN:-32}; NUM_PROMPTS=${NUM_PROMPTS:-1}
elif [[ ${MODE} == full ]]; then
  OUTPUT_ROOT=${OUTPUT_ROOT:-output/aux_experiments}; TRAIN_MODE=${TRAIN_MODE:-nccl}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}; MAX_STEPS=${MAX_STEPS:--1}; SAMPLE_NUM=${SAMPLE_NUM:-}; SAVE_STEPS=${SAVE_STEPS:-5000}; MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}; OUTPUT_LEN=${OUTPUT_LEN:-1024}; NUM_PROMPTS=${NUM_PROMPTS:-80}
else echo MODE must be smoke or full >&2; exit 1; fi

SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
DATASETS="${DATASETS:-lmms-lab/textvqa MMMU/MMMU Lin-Chen/MMStar opendatalab/OmniDocBench HuggingFaceH4/MATH-500 lmms-lab/COCO-Caption}"
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-4}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
TEMP=${TEMP:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
TARGET_HS_WARMUP_STEPS=${TARGET_HS_WARMUP_STEPS:-0}
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-true}
mkdir -p "${OUTPUT_ROOT}"
echo "=== Active aux experiments: mode=${MODE}, output=${OUTPUT_ROOT} ==="
echo "datasets=${DATASETS}; prompts=${NUM_PROMPTS}; temp=${TEMP}; K=${NUM_SPEC_TOKENS}"

for run_id in "${RUN_IDS[@]}"; do
  config_path="$(config_for "${run_id}")"
  output_dir="${OUTPUT_ROOT}/${run_id}"
  eval_root="${output_dir}/eval_temp0"
  warmstart="$(warmstart_for "${run_id}")"
  echo
  echo "======== TRAIN ${run_id} ========"
  echo "config=${config_path} output=${output_dir} warmstart=${warmstart:-none}"
  train_env=(TRAIN_MODE="${TRAIN_MODE}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DRAFT_MODEL_CONFIG_PATH="${config_path}" OUTPUT_DIR="${output_dir}" MAX_STEPS="${MAX_STEPS}" SAVE_STRATEGY="${SAVE_STRATEGY}" SAVE_STEPS="${SAVE_STEPS}" MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH}" TARGET_HS_WARMUP_STEPS="${TARGET_HS_WARMUP_STEPS}" LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE}")
  [[ -n "${SAMPLE_NUM}" ]] && train_env+=(SAMPLE_NUM="${SAMPLE_NUM}")
  [[ -n "${warmstart}" ]] && train_env+=(DRAFT_MODEL_NAME_OR_PATH="${warmstart}")
  env "${train_env[@]}" bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
  echo
  echo "======== EVAL ${run_id} ========"
  DRAFT_MODEL="${output_dir}" DRAFT_MODEL_CONFIG_PATH="${config_path}" RUN_NAME="${run_id}" OUT_ROOT="${eval_root}" DATASETS="${DATASETS}" NUM_PROMPTS="${NUM_PROMPTS}" OUTPUT_LEN="${OUTPUT_LEN}" NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS}" MAX_NUM_SEQS="${MAX_NUM_SEQS}" TEMP="${TEMP}" PYTHON_BIN="${PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" bash scripts/speculative/smolvlm/eval_acceptance_suite.sh
done
