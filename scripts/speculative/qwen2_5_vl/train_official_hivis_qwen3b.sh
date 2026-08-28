#!/bin/bash
# Official HiViS training setup for Qwen2.5-VL-3B, in the AngelSLIM checkout.
# Mirrors scripts/speculative/smolvlm/train_official_hivis_smolvlm.sh --
# same two-stage recipe, same shared mixed dataset, different target model
# and draft config.
#
# Stages:
#   STAGE=generate          generate official HiViS .ckpt data from the mixed JSONL
#                           (one pass: loads the target model once and routes
#                           each record to the text or multimodal output dir
#                           based on its own content -- see ge_data_qwen.py)
#   STAGE=stage1            run official hivis.train.main_mix
#   STAGE=stage2            run official hivis.train.main_mix_topk_dyn_res
#   STAGE=all               generate, stage1, stage2
#
# Also: stage2 here points at the same text-data-dir as stage1 (no /non_code
# subfolder) -- ge_data_qwen.py never creates a non_code split, so pointing
# stage2 at "$TEXT_CKPT_DIR/non_code" the way the SmolVLM script does would
# raise FileNotFoundError the first time stage2 actually runs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HIVIS_ROOT="${HIVIS_ROOT:-$ROOT/HiViS}"
cd "$ROOT"

export PYTHONPATH="$ROOT:$HIVIS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# hivis conda env has the torch/torchvision/transformers/accelerate versions
# this pipeline was validated against. Prepend it to PATH ONLY if it exists
# on this machine -- on a server managed with something other than this one
# conda env (uv, a different conda install, ...), silently do nothing and
# rely on whatever the caller already has active. Override the expected path
# via HIVIS_CONDA_ENV, or set HIVIS_CONDA_ENV="" to always skip this.
HIVIS_CONDA_ENV="${HIVIS_CONDA_ENV:-/home/hyang/anaconda3/envs/hivis}"
if [[ -n "$HIVIS_CONDA_ENV" && -d "$HIVIS_CONDA_ENV/bin" ]]; then
  export PATH="$HIVIS_CONDA_ENV/bin:$PATH"
fi
if [[ "${DRY_RUN:-0}" != "1" && -f "$ROOT/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/third_party/env.sh"
fi

STAGE=${STAGE:-all}
TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-/home/hyang/HiViS/models/Qwen2.5-VL-3B-Instruct}
DATA_FILE=${DATA_FILE:-$ROOT/dataset/preprocessed/mixed_sharegpt_llava665k_70k70k_b64.jsonl}
HIVIS_DATA_ROOT=${HIVIS_DATA_ROOT:-$ROOT/dataset/hivis_qwen25vl_3b_generated}
TEXT_CKPT_DIR=${TEXT_CKPT_DIR:-$HIVIS_DATA_ROOT/sharegpt}
MULTIMODAL_CKPT_DIR=${MULTIMODAL_CKPT_DIR:-$HIVIS_DATA_ROOT/llava_v1_5_mix665k}
CONFIG_PATH=${CONFIG_PATH:-$HIVIS_ROOT/hivis/train/qwen2_5_vl_3B_config.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/output/hivis_official/qwen25vl_3b}
STAGE1_DIR=${STAGE1_DIR:-$OUTPUT_ROOT/stage1}
STAGE2_DIR=${STAGE2_DIR:-$OUTPUT_ROOT/stage2}
STAGE1_CKPT=${STAGE1_CKPT:-$STAGE1_DIR/state_0}

GPUS=(${GPUS:-0 1 2 3})
GPU_CSV=$(IFS=,; echo "${GPUS[*]}")
# Without an accelerate config file, `accelerate launch` with no
# --multi_gpu/--num_processes falls back to a single process --
# CUDA_VISIBLE_DEVICES would then only make GPUS visible, not actually use
# more than the first one. Force real multi-GPU DDP when GPUS has >1 entry.
if [[ ${#GPUS[@]} -gt 1 ]]; then
  ACCELERATE_MULTI_GPU_ARGS=(--multi_gpu --num_processes "${#GPUS[@]}")
else
  ACCELERATE_MULTI_GPU_ARGS=()
fi
START=${START:-0}
# allocation.py clamps this to the data file's actual row count before
# splitting across GPUS, so this sentinel just means "everything from START".
END=${END:-1000000000000}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
NUM_WORKERS=${NUM_WORKERS:-1}
BS_STAGE1=${BS_STAGE1:-4}
BS_STAGE2=${BS_STAGE2:-2}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LR=${LR:-3e-5}
STAGE1_EPOCHS=${STAGE1_EPOCHS:-20}
STAGE2_EPOCHS=${STAGE2_EPOCHS:-10}
FORWARD_NUM_TOTAL=${FORWARD_NUM_TOTAL:-3}
TOPK=${TOPK:-10}
TOPK_W=${TOPK_W:-1.0}
FAIL_FAST=${FAIL_FAST:-false}

run_generate() {
  local args=(
    -m hivis.ge_data.allocation
    --model qwen
    --model-path "$TARGET_MODEL_NAME_OR_PATH"
    --data-file "$DATA_FILE"
    --outdir "$TEXT_CKPT_DIR"
    --multimodal-outdir "$MULTIMODAL_CKPT_DIR"
    --start "$START"
    --end "$END"
    --max-length "$MODEL_MAX_LENGTH"
    --num-workers "$NUM_WORKERS"
    --gpus "${GPUS[@]}"
  )
  [[ "$FAIL_FAST" == "true" ]] && args+=(--fail-fast)
  python "${args[@]}"
}

run_stage1() {
  CUDA_VISIBLE_DEVICES="$GPU_CSV" accelerate launch -m "${ACCELERATE_MULTI_GPU_ARGS[@]}" --mixed_precision=bf16 hivis.train.main_mix \
    --basepath "$TARGET_MODEL_NAME_OR_PATH" \
    --configpath "$CONFIG_PATH" \
    --text-data-dir "$TEXT_CKPT_DIR" \
    --multimodal-data-dir "$MULTIMODAL_CKPT_DIR" \
    --cpdir "$STAGE1_DIR" \
    --lr "$LR" \
    --bs "$BS_STAGE1" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --num-epochs "$STAGE1_EPOCHS" \
    --max-len "$MODEL_MAX_LENGTH"
}

run_stage2() {
  CUDA_VISIBLE_DEVICES="$GPU_CSV" accelerate launch -m "${ACCELERATE_MULTI_GPU_ARGS[@]}" --mixed_precision=bf16 hivis.train.main_mix_topk_dyn_res \
    --basepath "$TARGET_MODEL_NAME_OR_PATH" \
    --configpath "$CONFIG_PATH" \
    --text-data-dir "$TEXT_CKPT_DIR" \
    --multimodal-data-dir "$MULTIMODAL_CKPT_DIR" \
    --ckpt_path "$STAGE1_CKPT" \
    --cpdir "$STAGE2_DIR" \
    --lr "$LR" \
    --bs "$BS_STAGE2" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --num-epochs "$STAGE2_EPOCHS" \
    --max-len "$MODEL_MAX_LENGTH" \
    --forward_num_total "$FORWARD_NUM_TOTAL" \
    --topk "$TOPK" \
    --topk_w "$TOPK_W"
}

echo "=== Official HiViS Qwen2.5-VL-3B setup ==="
echo "  stage=$STAGE target=$TARGET_MODEL_NAME_OR_PATH"
echo "  data_file=$DATA_FILE"
echo "  text_ckpts=$TEXT_CKPT_DIR"
echo "  multimodal_ckpts=$MULTIMODAL_CKPT_DIR"
echo "  output=$OUTPUT_ROOT"
echo "  hivis_root=$HIVIS_ROOT"
echo "  gpus=$GPU_CSV"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: command validated, not launching HiViS generation/training."
  exit 0
fi

case "$STAGE" in
  generate)
    run_generate
    ;;
  stage1)
    run_stage1
    ;;
  stage2)
    run_stage2
    ;;
  all)
    run_generate
    run_stage1
    run_stage2
    ;;
  *)
    echo "ERROR: STAGE must be generate, stage1, stage2, or all (got: $STAGE)" >&2
    exit 1
    ;;
esac
