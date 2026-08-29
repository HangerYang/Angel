#!/bin/bash
# Official HiViS two-stage training, unified across target models.
#
# Replaces the earlier per-model scripts (train_official_hivis_qwen3b.sh,
# train_official_hivis_smolvlm.sh) -- stage1/stage2 were 100% identical
# between them; only the target model path, draft config, and the
# hivis.ge_data.allocation --model flag differed. MODEL picks those.
#
# MODEL=qwen25vl_7b | qwen25vl_3b | smolvlm256m
#
# Stages:
#   STAGE=generate  generate official HiViS .ckpt data from the mixed JSONL
#                   (one pass: loads the target model once and routes each
#                   record based on its own content -- has an image ->
#                   multimodal; text-only -> code/non_code via HiViS's own
#                   is_code_heavy() heuristic -- see ge_data_qwen.py /
#                   ge_data_smolvlm.py)
#   STAGE=stage1    run official hivis.train.main_mix
#   STAGE=stage2    run official hivis.train.main_mix_topk_dyn_res
#   STAGE=all       generate, stage1, stage2
#
# Stage 1 points --text-data-dir at $TEXT_CKPT_DIR (main_mix.py's
# list_files() recurses, so it picks up both code/ and non_code/, and every
# per-GPU <index>/ subfolder generate wrote -- multi-GPU generate output
# needs no separate merge step). Stage 2 points at $TEXT_CKPT_DIR/non_code
# only, matching HiViS's own stage2 recipe.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HIVIS_ROOT="${HIVIS_ROOT:-$ROOT/HiViS}"
cd "$ROOT"

export PYTHONPATH="$ROOT:$HIVIS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# No conda env is assumed or prepended to $PATH here -- activate whatever
# environment (conda, uv venv, ...) you need *before* calling this script.
if [[ "${DRY_RUN:-0}" != "1" && -f "$ROOT/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/third_party/env.sh"
fi

MODEL="${MODEL:?Set MODEL to one of: qwen25vl_7b, qwen25vl_3b, smolvlm256m}"

case "$MODEL" in
  qwen25vl_7b)
    GE_DATA_MODEL=qwen
    DEFAULT_TARGET_MODEL=/home/hyang/HiViS/models/Qwen2.5-VL-7B-Instruct
    DEFAULT_CONFIG_PATH="$HIVIS_ROOT/hivis/train/qwen2.5_vl_7B_config.json"
    ;;
  qwen25vl_3b)
    GE_DATA_MODEL=qwen
    DEFAULT_TARGET_MODEL=/home/hyang/HiViS/models/Qwen2.5-VL-3B-Instruct
    DEFAULT_CONFIG_PATH="$HIVIS_ROOT/hivis/train/qwen2_5_vl_3B_config.json"
    ;;
  smolvlm256m)
    GE_DATA_MODEL=smolvlm
    DEFAULT_TARGET_MODEL=HuggingFaceTB/SmolVLM-256M-Instruct
    DEFAULT_CONFIG_PATH="$HIVIS_ROOT/hivis/train/smolvlm_256m_config.json"
    ;;
  *)
    echo "ERROR: MODEL must be one of: qwen25vl_7b, qwen25vl_3b, smolvlm256m (got: $MODEL)" >&2
    exit 1
    ;;
esac

STAGE=${STAGE:-all}
TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-$DEFAULT_TARGET_MODEL}
DATA_FILE=${DATA_FILE:-$ROOT/dataset/preprocessed/mixed_sharegpt_llava665k_70k70k_b64.jsonl}
HIVIS_DATA_ROOT=${HIVIS_DATA_ROOT:-$ROOT/dataset/hivis_${MODEL}_generated}
TEXT_CKPT_DIR=${TEXT_CKPT_DIR:-$HIVIS_DATA_ROOT/sharegpt}
MULTIMODAL_CKPT_DIR=${MULTIMODAL_CKPT_DIR:-$HIVIS_DATA_ROOT/llava_v1_5_mix665k}
CONFIG_PATH=${CONFIG_PATH:-$DEFAULT_CONFIG_PATH}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/output/hivis_official/$MODEL}
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
# torch.distributed backend for multi-GPU DDP: nccl (default, unset here --
# accelerate/torch pick it on CUDA) or gloo.
DDP_BACKEND=${DDP_BACKEND:-}

run_generate() {
  local args=(
    -m hivis.ge_data.allocation
    --model "$GE_DATA_MODEL"
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
  local args=(
    --basepath "$TARGET_MODEL_NAME_OR_PATH"
    --configpath "$CONFIG_PATH"
    --text-data-dir "$TEXT_CKPT_DIR"
    --multimodal-data-dir "$MULTIMODAL_CKPT_DIR"
    --cpdir "$STAGE1_DIR"
    --lr "$LR"
    --bs "$BS_STAGE1"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --num-epochs "$STAGE1_EPOCHS"
    --max-len "$MODEL_MAX_LENGTH"
  )
  [[ -n "$DDP_BACKEND" ]] && args+=(--ddp-backend "$DDP_BACKEND")
  CUDA_VISIBLE_DEVICES="$GPU_CSV" accelerate launch -m "${ACCELERATE_MULTI_GPU_ARGS[@]}" --mixed_precision=bf16 hivis.train.main_mix "${args[@]}"
}

run_stage2() {
  local args=(
    --basepath "$TARGET_MODEL_NAME_OR_PATH"
    --configpath "$CONFIG_PATH"
    --text-data-dir "$TEXT_CKPT_DIR/non_code"
    --multimodal-data-dir "$MULTIMODAL_CKPT_DIR"
    --ckpt_path "$STAGE1_CKPT"
    --cpdir "$STAGE2_DIR"
    --lr "$LR"
    --bs "$BS_STAGE2"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --num-epochs "$STAGE2_EPOCHS"
    --max-len "$MODEL_MAX_LENGTH"
    --forward_num_total "$FORWARD_NUM_TOTAL"
    --topk "$TOPK"
    --topk_w "$TOPK_W"
  )
  [[ -n "$DDP_BACKEND" ]] && args+=(--ddp-backend "$DDP_BACKEND")
  CUDA_VISIBLE_DEVICES="$GPU_CSV" accelerate launch -m "${ACCELERATE_MULTI_GPU_ARGS[@]}" --mixed_precision=bf16 hivis.train.main_mix_topk_dyn_res "${args[@]}"
}

echo "=== Official HiViS training: MODEL=$MODEL ==="
echo "  stage=$STAGE target=$TARGET_MODEL_NAME_OR_PATH"
echo "  data_file=$DATA_FILE"
echo "  text_ckpts=$TEXT_CKPT_DIR"
echo "  multimodal_ckpts=$MULTIMODAL_CKPT_DIR"
echo "  config=$CONFIG_PATH"
echo "  output=$OUTPUT_ROOT"
echo "  hivis_root=$HIVIS_ROOT"
echo "  gpus=$GPU_CSV"
echo "  ddp_backend=${DDP_BACKEND:-<default: nccl>}"

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
