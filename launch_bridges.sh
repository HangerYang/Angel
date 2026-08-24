#!/usr/bin/env bash
# Detached launcher for the Plan B + early-exit-bridges run.
# Run via: setsid nohup bash launch_bridges.sh >/dev/null 2>&1 &
set -euo pipefail

REPO_ROOT="/home/hyang/AngelSlim"
cd "$REPO_ROOT"

export PATH=/home/hyang/miniconda3/envs/angel/bin:$PATH

DRAFT_MODEL_CONFIG_PATH="angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-bridges.json" \
OUTPUT_DIR="${REPO_ROOT}/outputs/smolvlm-256m-eagle3-runb-bridges" \
TARGET_MODEL_NAME_OR_PATH="HuggingFaceTB/SmolVLM-256M-Instruct" \
TRAIN_DATA_PATH="${REPO_ROOT}/dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl" \
EVAL_DATA_PATH="${REPO_ROOT}/dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl" \
TRAIN_MODE="nccl" \
NUM_TRAIN_EPOCHS=2 \
SAVE_STRATEGY="epoch" \
EVAL_STRATEGY="epoch" \
LOAD_FROM_CACHE_FILE="true" \
LOGGING_STEPS=10 \
bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh \
  >> "${REPO_ROOT}/outputs/smolvlm-256m-eagle3-runb-bridges/train.log" 2>&1
