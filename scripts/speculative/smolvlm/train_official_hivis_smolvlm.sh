#!/bin/bash
# Official HiViS training setup for SmolVLM-256M in the AngelSLIM checkout.
#
# Stages:
#   STAGE=generate          generate official HiViS .ckpt data from the mixed JSONL
#   STAGE=generate_text     generate text-only .ckpt data
#   STAGE=generate_mm       generate multimodal .ckpt data
#   STAGE=stage1            run official hivis.train.main_mix
#   STAGE=stage2            run official hivis.train.main_mix_topk_dyn_res
#   STAGE=all               generate, stage1, stage2

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HIVIS_ROOT="$ROOT/third_party/HiViS"
cd "$ROOT"

export PYTHONPATH="$ROOT:$HIVIS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$ROOT/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/third_party/env.sh"
fi

STAGE=${STAGE:-all}
TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolVLM-256M-Instruct}
DATA_FILE=${DATA_FILE:-$ROOT/dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl}
HIVIS_DATA_ROOT=${HIVIS_DATA_ROOT:-$ROOT/dataset/hivis_smolvlm_256m_generated}
TEXT_CKPT_DIR=${TEXT_CKPT_DIR:-$HIVIS_DATA_ROOT/sharegpt}
MULTIMODAL_CKPT_DIR=${MULTIMODAL_CKPT_DIR:-$HIVIS_DATA_ROOT/llava_v1_5_mix665k}
CONFIG_PATH=${CONFIG_PATH:-$HIVIS_ROOT/hivis/train/smolvlm_256m_config.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/output/hivis_official/smolvlm_256m}
STAGE1_DIR=${STAGE1_DIR:-$OUTPUT_ROOT/stage1}
STAGE2_DIR=${STAGE2_DIR:-$OUTPUT_ROOT/stage2}
STAGE1_CKPT=${STAGE1_CKPT:-$STAGE1_DIR/state_0}

GPUS=(${GPUS:-0 1 2 3})
GPU_CSV=$(IFS=,; echo "${GPUS[*]}")
START=${START:-0}
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

run_generate_one() {
  local data_type="$1"
  local outdir="$2"
  local args=(
    -m hivis.ge_data.allocation
    --model smolvlm
    --data-type "$data_type"
    --model-path "$TARGET_MODEL_NAME_OR_PATH"
    --data-file "$DATA_FILE"
    --outdir "$outdir"
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
  CUDA_VISIBLE_DEVICES="$GPU_CSV" accelerate launch -m --mixed_precision=bf16 hivis.train.main_mix \
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
  CUDA_VISIBLE_DEVICES="$GPU_CSV" accelerate launch -m --mixed_precision=bf16 hivis.train.main_mix_topk_dyn_res \
    --basepath "$TARGET_MODEL_NAME_OR_PATH" \
    --configpath "$CONFIG_PATH" \
    --text-data-dir "$TEXT_CKPT_DIR/non_code" \
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

echo "=== Official HiViS SmolVLM setup ==="
echo "  stage=$STAGE target=$TARGET_MODEL_NAME_OR_PATH"
echo "  data_file=$DATA_FILE"
echo "  text_ckpts=$TEXT_CKPT_DIR"
echo "  multimodal_ckpts=$MULTIMODAL_CKPT_DIR"
echo "  output=$OUTPUT_ROOT"
echo "  gpus=$GPU_CSV"

case "$STAGE" in
  generate)
    run_generate_one text "$TEXT_CKPT_DIR"
    run_generate_one multimodal "$MULTIMODAL_CKPT_DIR"
    ;;
  generate_text)
    run_generate_one text "$TEXT_CKPT_DIR"
    ;;
  generate_mm)
    run_generate_one multimodal "$MULTIMODAL_CKPT_DIR"
    ;;
  stage1)
    run_stage1
    ;;
  stage2)
    run_stage2
    ;;
  all)
    run_generate_one text "$TEXT_CKPT_DIR"
    run_generate_one multimodal "$MULTIMODAL_CKPT_DIR"
    run_stage1
    run_stage2
    ;;
  *)
    echo "ERROR: STAGE must be generate, generate_text, generate_mm, stage1, stage2, or all (got: $STAGE)" >&2
    exit 1
    ;;
esac
