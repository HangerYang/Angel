#!/bin/bash
# Online Eagle3 training for SmolVLM / Idefics3.
#
# Example (smoke on generated 36-sample set):
#   bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# Override paths / GPUs as needed:
#   TRAIN_DATA_PATH=... EVAL_DATA_PATH=... CUDA_VISIBLE_DEVICES=0,1 NPROC=2 \
#     bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "${ROOT}/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

CONFIG_DIR=angelslim/compressor/speculative/train/configs
TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolVLM-256M-Instruct}
DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:-$CONFIG_DIR/smolvlm-256m-eagle3.json}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-dataset/smolvlm_256m_target_gen/data_0-36.jsonl}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-output/smolvlm_256m_eagle3_online}
EMBED_WEIGHT_KEY=${EMBED_WEIGHT_KEY:-model.text_model.embed_tokens.weight}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-smolvlm}
NPROC=${NPROC:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
SAMPLE_NUM=${SAMPLE_NUM:-}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}

export CUDA_VISIBLE_DEVICES

EVAL_ARGS=()
if [[ -n "${EVAL_DATA_PATH}" ]]; then
  EVAL_ARGS+=(--eval_data_path "${EVAL_DATA_PATH}")
fi

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_NUM}" ]]; then
  SAMPLE_ARGS+=(--sample_num "${SAMPLE_NUM}")
fi

DS_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  DS_ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

torchrun --nproc_per_node="${NPROC}" tools/train_eagle3_online.py \
    --modal_type VLM \
    --target_model_name_or_path "${TARGET_MODEL_NAME_OR_PATH}" \
    --draft_model_config_path "${DRAFT_MODEL_CONFIG_PATH}" \
    --train_data_path "${TRAIN_DATA_PATH}" \
    "${EVAL_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_proc 4 \
    --save_strategy "no" \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --embed_weight_key "${EMBED_WEIGHT_KEY}" \
    --chat_template_type "${CHAT_TEMPLATE_TYPE}" \
    --bf16 \
    --report_to none \
    --run_name smolvlm-256m-eagle3-angelslim \
    "${SAMPLE_ARGS[@]}" \
    "${DS_ARGS[@]}"
