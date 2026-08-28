#!/bin/bash
# Official HiViS Qwen2.5-VL-3B baseline recipe.
# Generates the offline HiViS cache, then trains Stage 1 for 2 epochs and
# Stage 2 for 1 epoch, over all 8 GPUs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export STAGE=${STAGE:-all}
export GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
export DATA_FILE=${DATA_FILE:-$ROOT/dataset/preprocessed/mixed_sharegpt_llava665k_70k70k_b64.jsonl}
export HIVIS_DATA_ROOT=${HIVIS_DATA_ROOT:-$ROOT/dataset/hivis_qwen25vl_3b_generated}
export OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/output/hivis_official/qwen25vl_3b_stage1_2ep_stage2_1ep}
export STAGE1_DIR=${STAGE1_DIR:-$OUTPUT_ROOT/stage1}
export STAGE2_DIR=${STAGE2_DIR:-$OUTPUT_ROOT/stage2}
export STAGE1_CKPT=${STAGE1_CKPT:-$STAGE1_DIR/state_0}
export STAGE1_EPOCHS=${STAGE1_EPOCHS:-2}
export STAGE2_EPOCHS=${STAGE2_EPOCHS:-1}

bash "$ROOT/scripts/speculative/qwen2_5_vl/train_official_hivis_qwen3b.sh"
