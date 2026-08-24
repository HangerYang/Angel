#!/usr/bin/env bash
# Q-Sampler distillation launcher (env-var driven, same style as the
# speculative training scripts).
#
#   NUM_QUERIES=4 bash scripts/qsampler/train_qsampler.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

DATA_DIR=${DATA_DIR:-dataset/smolvlm_256m_target_gen_mixed_70k70k}
# Image-only split: the mixed corpus is ~50% text-only ShareGPT, which carries
# no signal for a visual-token sampler. Built by scripts/qsampler/filter_image_rows.py.
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$DATA_DIR/train_images_only.jsonl}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-$DATA_DIR/eval_images_only.jsonl}

NUM_QUERIES=${NUM_QUERIES:-4}
NUM_BLOCKS=${NUM_BLOCKS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-output/qsampler-n${NUM_QUERIES}}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}     # cosine | linear | constant
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}        # decay floor as a fraction of LR
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
# Decoder block 26 of 30 -- late, but not the last (the last is near-collinear
# with the logits, so it would just re-weight the KL term). Same indexing
# convention as aux_hidden_states_layer_ids: block i == hidden_states[i+1].
HIDDEN_LAYERS=${HIDDEN_LAYERS:-26}
HIDDEN_LOSS=${HIDDEN_LOSS:-cosine}
LAMBDA_HIDDEN=${LAMBDA_HIDDEN:-0.3}
NPROC=${NPROC:-4}
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "Q-Sampler: 64 -> ${NUM_QUERIES} tokens/tile | out=${OUTPUT_DIR} | epochs=${NUM_TRAIN_EPOCHS}"

torchrun --nproc_per_node="${NPROC}" tools/train_qsampler.py \
  --train_data_path "${TRAIN_DATA_PATH}" \
  --eval_data_path "${EVAL_DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_queries "${NUM_QUERIES}" \
  --num_blocks "${NUM_BLOCKS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --lr_scheduler "${LR_SCHEDULER}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --min_lr_ratio "${MIN_LR_RATIO}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --hidden_layers "${HIDDEN_LAYERS}" \
  --hidden_loss "${HIDDEN_LOSS}" \
  --lambda_hidden "${LAMBDA_HIDDEN}" \
  ${EXTRA_ARGS}
