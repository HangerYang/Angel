#!/usr/bin/env bash
#
# ViSpec data-generation + training pipeline for SmolVLM-256M (Idefics3 arch).
#
# Evaluation is not part of this repo: drafts trained here are evaluated through
# HiViS's harness, not ViSpec's, so ViSpec's evaluation code was removed.
#
# Usage:
#   bash run_smolvlm.sh                 # all stages
#   bash run_smolvlm.sh 1 1.1 1.2       # data generation only
#   bash run_smolvlm.sh 2               # training only
#
# Data:
#   * Stage 1.1: Aeala/ShareGPT_Vicuna_unfiltered (HF auto-download)
#   * Stage 1.2: LLaVA-Pretrain at LLAVA_DATA_PATH (default: /data/llava_datasets/...)

set -euo pipefail
ulimit -n 1048576 || true

# ─── Hardware ────────────────────────────────────────────────────────────────
# Physical GPU ids used for data-gen workers and training/eval (via CUDA_VISIBLE_DEVICES).
GPU_IDS=(2 3 4 5 6 7)
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")"

# ─── Paths & model ───────────────────────────────────────────────────────────
BASE_MODEL="HuggingFaceTB/SmolVLM-256M-Instruct"
CONFIG="vispec/train/smolvlm_256M_config.json"
LLAVA_DATA_PATH="/data/llava_datasets/data/LLaVA-Pretrain"

TEXT_DATA_ROOT="vispec_data/smolvlm/text"
MM_DATA_ROOT="vispec_data/smolvlm/multimodal"
CKPT_STAGE1="vispec_data/smolvlm/ckpt_stage1"
CKPT_STAGE2="vispec_data/smolvlm/ckpt_stage2"
LOG_DIR="vispec_data/smolvlm/logs"

# ViSpec default sample range (68000 examples)
START=0
END=67999
STAGE1_EPOCH_STATE="state_20"
STAGE2_EPOCH_STATE="state_20"

NUM_Q=2
DEPTH=3
TOP_K=8
TOTAL_TOKEN=30
MTP_STEPS=1
MAX_LEN=4096

mkdir -p "${TEXT_DATA_ROOT}" "${MM_DATA_ROOT}" "${CKPT_STAGE1}" "${CKPT_STAGE2}" "${LOG_DIR}"

# ─── Stage selection ────────────────────────────────────────────────────────
STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
  STAGES=(1 2)
fi
run_stage() {
  local target="$1"
  for s in "${STAGES[@]}"; do
    [[ "$s" == "$target" || "$s" == "${target%%.*}" ]] && return 0
  done
  return 1
}

GPU_IDS_STR="${GPU_IDS[*]}"

# ─── Stage 1.1: text-only data ────────────────────────────────────────────────
if run_stage 1.1; then
  echo "=== Stage 1.1: text-only data generation (GPUs: ${GPU_IDS_STR}) ==="
  python -m vispec.ge_data.allocation_idefics3_shargpt \
    --outdir="${TEXT_DATA_ROOT}" \
    --start="${START}" --end="${END}" \
    --model="${BASE_MODEL}" \
    --gpu_ids ${GPU_IDS_STR}
fi

# ─── Stage 1.2: multimodal data ───────────────────────────────────────────────
if run_stage 1.2; then
  echo "=== Stage 1.2: multimodal data generation (GPUs: ${GPU_IDS_STR}) ==="
  echo "LLaVA-Pretrain: ${LLAVA_DATA_PATH}"
  python -m vispec.ge_data.allocation_idefics3_pretrain_gen \
    --outdir="${MM_DATA_ROOT}" \
    --start="${START}" --end="${END}" \
    --model="${BASE_MODEL}" \
    --temperature=1.0 \
    --datapath="${LLAVA_DATA_PATH}" \
    --gpu_ids ${GPU_IDS_STR}
fi

# ─── Stage 2.1: initial (text) training ───────────────────────────────────────
if run_stage 2.1; then
  echo "=== Stage 2.1: initial draft training (GPUs: ${CUDA_VISIBLE_DEVICES}) ==="
  accelerate launch --multi_gpu --mixed_precision=bf16 \
    -m vispec.train.main \
    --cpdir="${CKPT_STAGE1}" \
    --basepath="${BASE_MODEL}" \
    --begin-epoch=0 \
    --bs=1 \
    --configpath="${CONFIG}" \
    --lr=3e-5 \
    --max-len="${MAX_LEN}" \
    --num-workers=8 \
    --tmpdir="${TEXT_DATA_ROOT}"
fi

# ─── Stage 2.2: ViSpec training (multimodal) ──────────────────────────────────
if run_stage 2.2; then
  echo "=== Stage 2.2: ViSpec training (GPUs: ${CUDA_VISIBLE_DEVICES}) ==="
  accelerate launch --multi_gpu --mixed_precision=bf16 \
    -m vispec.train.main_mtp \
    --cpdir="${CKPT_STAGE2}" \
    --basepath="${BASE_MODEL}" \
    --begin-epoch=0 \
    --bs=1 \
    --configpath="${CONFIG}" \
    --loadpath="${CKPT_STAGE1}/${STAGE1_EPOCH_STATE}/model.safetensors" \
    --lr=3e-6 \
    --max-len="${MAX_LEN}" \
    --mtp-steps="${MTP_STEPS}" \
    --num-q="${NUM_Q}" \
    --num-workers=8 \
    --tmpdir="${MM_DATA_ROOT}" \
    --use-ours=True
fi


echo "All requested stages complete."
