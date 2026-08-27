#!/bin/bash
# Official HiViS SmolVLM-256M baseline recipe.
# Generates the offline HiViS cache, then trains Stage 1 for 2 epochs and
# Stage 2 for 1 epoch.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export STAGE=${STAGE:-all}
export GPUS=${GPUS:-"0 1 2 3"}
export DATA_FILE=${DATA_FILE:-$ROOT/dataset/preprocessed/mixed_sharegpt_llava665k_70k70k_b64.jsonl}
export HIVIS_DATA_ROOT=${HIVIS_DATA_ROOT:-$ROOT/dataset/hivis_smolvlm_256m_generated}
export OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/output/hivis_official/smolvlm_256m_stage1_2ep_stage2_1ep}
export STAGE1_DIR=${STAGE1_DIR:-$OUTPUT_ROOT/stage1}
export STAGE2_DIR=${STAGE2_DIR:-$OUTPUT_ROOT/stage2}
export STAGE1_CKPT=${STAGE1_CKPT:-$STAGE1_DIR/state_0}
export STAGE1_EPOCHS=${STAGE1_EPOCHS:-2}
export STAGE2_EPOCHS=${STAGE2_EPOCHS:-1}

bash "$ROOT/scripts/speculative/smolvlm/train_official_hivis_smolvlm.sh"
