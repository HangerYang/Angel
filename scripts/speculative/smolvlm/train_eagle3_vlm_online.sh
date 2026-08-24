#!/bin/bash
# Online Eagle3 training for SmolVLM / Idefics3.
#
# Default: DeepSpeed ZeRO-3, 4 GPU, 2 epochs, save every 5k steps
#   bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#
# Quick examples:
#   TRAIN_MODE=nccl      bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#   TRAIN_MODE=gloo      bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#   TRAIN_MODE=python    bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
#   DRAFT_MODEL_CONFIG_PATH=.../smolvlm-256m-hawk.json OUTPUT_DIR=output/hawk \
#     bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

set -euo pipefail

# TRAIN_MODE — how to launch multi-GPU training.
#   Options: deepspeed | nccl | gloo | python
#   deepspeed (default) — torchrun + ZeRO-3 JSON (best mem for large drafts)
#   nccl                — plain DDP via torchrun/NCCL (no DeepSpeed)
#   gloo                — DDP via gloo (CPU-collective; rarely needed)
#   python              — single process, 1 GPU (set EQUIV_NPROC to match multi-GPU batch)
TRAIN_MODE=${TRAIN_MODE:-deepspeed}
case "${TRAIN_MODE}" in
  nccl)
    LAUNCH=${LAUNCH:-torchrun}                 # LAUNCH: torchrun | python (override only if you know why)
    DIST_BACKEND=${DIST_BACKEND:-nccl}         # DIST_BACKEND: nccl | gloo (torch DDP backend)
    NPROC=${NPROC:-4}                          # NPROC: # GPU processes for torchrun (e.g. 1|2|4|8)
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}  # CUDA_VISIBLE_DEVICES: comma GPU ids visible to this job
    ;;
  deepspeed)
    LAUNCH=${LAUNCH:-torchrun}
    DIST_BACKEND=${DIST_BACKEND:-nccl}
    NPROC=${NPROC:-4}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    # DEEPSPEED_CONFIG — ZeRO JSON path. Default zero3. Set "" to disable (not usual for TRAIN_MODE=deepspeed).
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-angelslim/compressor/speculative/train/configs/deepspeed_zero3.json}
    ;;
  gloo)
    LAUNCH=${LAUNCH:-torchrun}
    DIST_BACKEND=${DIST_BACKEND:-gloo}
    NPROC=${NPROC:-4}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    ;;
  python)
    LAUNCH=${LAUNCH:-python}
    EQUIV_NPROC=${EQUIV_NPROC:-4}              # EQUIV_NPROC: fake world size → sets grad_accum so 1-GPU matches N-GPU step size
    NPROC=${NPROC:-1}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    NUM_PROC=${NUM_PROC:-1}                    # NUM_PROC: HF datasets map workers (lower on small machines)
    ;;
  *)
    echo "ERROR: TRAIN_MODE must be nccl, deepspeed, gloo, or python (got: ${TRAIN_MODE})" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# Repo must be first so its angelslim/ shadows any installed package.
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# PYTHON — interpreter to use for single-process launch.
#   Override to point at a conda env: PYTHON=/home/hyang/miniconda3/envs/angel/bin/python
PYTHON=${PYTHON:-python}
if [[ -f "${ROOT}/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

CONFIG_DIR=angelslim/compressor/speculative/train/configs

# TARGET_MODEL_NAME_OR_PATH — frozen teacher VLM (HS + logits). HF id or local dir.
#   Default: HuggingFaceTB/SmolVLM-256M-Instruct
TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolVLM-256M-Instruct}

# DRAFT_MODEL_CONFIG_PATH — draft architecture JSON (required). Builds draft from this blueprint.
#   Options under $CONFIG_DIR:
#     smolvlm-256m-eagle3.json                 — stock Eagle3 fused_fc, 1 draft layer
#     smolvlm-256m-eagle3-3.1.json             — stock + EAGLE 3.1 (fc_norm + norm_output)
#     smolvlm-256m-eagle3-banded-mix-fc-3.1.json — EAGLE 3.1 shape, each of the 3 FC
#                                                input streams a learned mix over an aux band
#     smolvlm-256m-eagle3-progressive.json     — progressive_staged, 3 layers, init from target
#     smolvlm-256m-eagle3-progressive-layers-1-15-23-3.1.json — progressive + norm_output
#     smolvlm-256m-eagle3-progressive-uninit.json — progressive, no layer init
#     smolvlm-256m-hawk.json                   — hawk (w1/w2 H-fusion, full layers trainable)
#     smolvlm-256m-real-hawk.json              — real_hawk (frozen target-layer copies + LoRA + fuse)
#     smolvlm-256m-layer-skip-lora.json        — alias of real_hawk
#   Example: DRAFT_MODEL_CONFIG_PATH=$CONFIG_DIR/smolvlm-256m-hawk.json
DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:-$CONFIG_DIR/smolvlm-256m-eagle3.json}

# DRAFT_MODEL_NAME_OR_PATH — optional warm-start weights for the draft (not the target).
#   Options:
#     "" (default)              — config / random init only
#     path/to/checkpoint-NNNN   — load draft weights from a prior run
#     path/to/warmup_end        — e.g. hawk GT-warmup end dir from run.sh
#   Skipped if OUTPUT_DIR already has checkpoint-* (HF resume wins over warm-start).
DRAFT_MODEL_NAME_OR_PATH=${DRAFT_MODEL_NAME_OR_PATH:-}

# TRAIN_DATA_PATH — training jsonl (or HF datasets load path). Must be a *file* or glob for json loader.
#   Default: dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl
#   Wrong: a bare directory (HF FileNotFoundError). Right: .../train.jsonl
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl}

# EVAL_DATA_PATH — held-out jsonl for trainer eval. Empty string skips attaching eval file.
#   Default: .../eval.jsonl. Pair with EVAL_STRATEGY=steps|epoch to actually run eval.
EVAL_DATA_PATH=${EVAL_DATA_PATH:-dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl}

# OUTPUT_DIR — checkpoints + vocab_mapping_cache.pt land here.
#   Example: OUTPUT_DIR=output/smolvlm_256m_hawk_nccl
OUTPUT_DIR=${OUTPUT_DIR:-output/smolvlm_256m_eagle3_online}

# EMBED_WEIGHT_KEY — key inside target weights for draft embed copy.
#   SmolVLM/Idefics3: model.text_model.embed_tokens.weight (default)
EMBED_WEIGHT_KEY=${EMBED_WEIGHT_KEY:-model.text_model.embed_tokens.weight}

# MODEL_MAX_LENGTH — max sequence length for tokenize / pack (tokens). Typical: 2048|4096|8192.
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}

# CHAT_TEMPLATE_TYPE — conversation template id for dataset formatting.
#   Options: smolvlm (default) | others registered in DatasetManager
CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-smolvlm}

# NUM_TRAIN_EPOCHS — full passes over train set. Ignored if MAX_STEPS > 0.
#   Typical: 1|2|4. run.sh hawk warmup/regular uses 2.
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}

# MAX_STEPS — if > 0, stop after this many optimizer steps (overrides epochs).
#   Options: -1 (default, use epochs) | positive int (e.g. 100 for smoke)
MAX_STEPS=${MAX_STEPS:--1}

# TARGET_HS_WARMUP_STEPS — GT target-HS warmup length (optimizer steps).
#   0 (default)     — no warmup; progressive/hawk use draft-HS feedback after step 0
#   N > 0           — for steps 0..N-1 teacher-force shifted GT aux on every speculative substep
#                     progressive_staged → progressive eagle GT; hawk/real_hawk → hawk GT
#   Huge N (e.g. 999999999) — keep entire run in warmup (see run.sh job 2)
#   Alias: PROGRESSIVE_TARGET_HS_WARMUP_STEPS (deprecated)
TARGET_HS_WARMUP_STEPS=${TARGET_HS_WARMUP_STEPS:-${PROGRESSIVE_TARGET_HS_WARMUP_STEPS:-0}}

# SAMPLE_NUM — if set, subsample this many train examples (smoke / debug). Empty = full data.
#   Example: SAMPLE_NUM=64
SAMPLE_NUM=${SAMPLE_NUM:-}

# DEEPSPEED_CONFIG — path to DeepSpeed JSON; empty means omit --deepspeed.
#   Set by TRAIN_MODE=deepspeed by default; override for zero2 etc.
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}

# DRAFT_MODEL_DTYPE — param dtype before Trainer/optimizer creation.
#   Options: config | float16 | bfloat16 | float32
#     config (default) — use "dtype" from draft JSON (usually bfloat16)
#     float32          — helpful for plain DDP/NCCL (Adam moments stay healthy)
DRAFT_MODEL_DTYPE=${DRAFT_MODEL_DTYPE:-config}

# LOAD_FROM_CACHE_FILE — reuse HF datasets map/filter cache under <train_jsonl_dir>/.map_cache/
#   Options: true (default) | false
#   Set false after changing preprocess code; or delete .map_cache manually.
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-true}

# SAVE_STRATEGY — when HF Trainer writes checkpoint-*.
#   Options: steps | epoch | no
#     steps (default) — every SAVE_STEPS
#     epoch           — end of each epoch (use for warmup_end guarantee)
#     no              — never save (smoke only)
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}

# SAVE_STEPS — checkpoint interval when SAVE_STRATEGY=steps. Typical: 1000|5000.
SAVE_STEPS=${SAVE_STEPS:-5000}

# EVAL_STRATEGY — when to run held-out loss eval during training.
#   Options: no (default) | steps | epoch
#   Need EVAL_DATA_PATH set. Example: EVAL_STRATEGY=steps EVAL_STEPS=5000
EVAL_STRATEGY=${EVAL_STRATEGY:-no}

# EVAL_STEPS — eval interval when EVAL_STRATEGY=steps.
EVAL_STEPS=${EVAL_STEPS:-5000}

# LEARNING_RATE — drafter LR. A vistoken row compressor takes its own LR from
# vistoken_compress.lr in the draft config, in a separate param group.
LEARNING_RATE=${LEARNING_RATE:-1e-4}

# LAUNCH / DIST_BACKEND / NPROC / CUDA_VISIBLE_DEVICES — may already be set by TRAIN_MODE case.
LAUNCH=${LAUNCH:-torchrun}
DIST_BACKEND=${DIST_BACKEND:-nccl}
NPROC=${NPROC:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# PER_DEVICE_TRAIN_BATCH_SIZE — per-GPU microbatch. Usually 1 for long VLM sequences.
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}

# EQUIV_NPROC — used with LAUNCH=python to set GRADIENT_ACCUMULATION_STEPS (match multi-GPU batch).
EQUIV_NPROC=${EQUIV_NPROC:-}

# GRADIENT_ACCUMULATION_STEPS — accumulate this many microbatches per optimizer step.
#   Default: 1; or EQUIV_NPROC when single-process equivalence mode.
if [[ -z "${GRADIENT_ACCUMULATION_STEPS+x}" ]]; then
  if [[ -n "${EQUIV_NPROC}" ]] && [[ "${LAUNCH}" == "python" || "${NPROC}" == "1" ]]; then
    GRADIENT_ACCUMULATION_STEPS="${EQUIV_NPROC}"
  else
    GRADIENT_ACCUMULATION_STEPS=1
  fi
fi

# NUM_PROC — HF datasets map workers. Typical: 1..8 (lower if RAM pressure).
NUM_PROC=${NUM_PROC:-4}

export CUDA_VISIBLE_DEVICES

ARGS=(
  tools/train_eagle3_online.py
  --modal_type VLM
  --target_model_name_or_path "${TARGET_MODEL_NAME_OR_PATH}"
  --draft_model_config_path "${DRAFT_MODEL_CONFIG_PATH}"
  --train_data_path "${TRAIN_DATA_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num_proc "${NUM_PROC}"
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE}"
  --save_strategy "${SAVE_STRATEGY}"
  --eval_strategy "${EVAL_STRATEGY}"
  --eval_steps "${EVAL_STEPS}"
  --learning_rate "${LEARNING_RATE}"             # drafter LR; the vistoken compressor gets its own from the draft config
  --weight_decay 0.0
  --warmup_ratio 0.05                           # LR warmup fraction of total steps (not HS warmup)
  --lr_scheduler_type "constant"
  --logging_steps 100
  --model_max_length "${MODEL_MAX_LENGTH}"
  --embed_weight_key "${EMBED_WEIGHT_KEY}"
  --chat_template_type "${CHAT_TEMPLATE_TYPE}"
  --bf16
  --report_to none
  --run_name smolvlm-256m-eagle3-angelslim
)

[[ -n "${EVAL_DATA_PATH}" ]] && ARGS+=(--eval_data_path "${EVAL_DATA_PATH}")
[[ -n "${DRAFT_MODEL_NAME_OR_PATH}" ]] && ARGS+=(--draft_model_name_or_path "${DRAFT_MODEL_NAME_OR_PATH}")
[[ -n "${SAMPLE_NUM}" ]] && ARGS+=(--sample_num "${SAMPLE_NUM}")
[[ -n "${DEEPSPEED_CONFIG}" ]] && ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
[[ -n "${SAVE_STEPS}" ]] && ARGS+=(--save_steps "${SAVE_STEPS}")
[[ -n "${MAX_STEPS}" && "${MAX_STEPS}" != "-1" ]] && ARGS+=(--max_steps "${MAX_STEPS}")
[[ "${TARGET_HS_WARMUP_STEPS}" != "0" ]] && \
  ARGS+=(--target_hs_warmup_steps "${TARGET_HS_WARMUP_STEPS}")
[[ "${DRAFT_MODEL_DTYPE}" != "config" ]] && ARGS+=(--draft_model_dtype "${DRAFT_MODEL_DTYPE}")
[[ "${LAUNCH}" == "torchrun" ]] && ARGS+=(--ddp_backend "${DIST_BACKEND}")

echo "=== SmolVLM Eagle3 train ==="
echo "  TRAIN_MODE=${TRAIN_MODE}  LAUNCH=${LAUNCH}  NPROC=${NPROC}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  draft_config=${DRAFT_MODEL_CONFIG_PATH}"
echo "  draft_warmstart=${DRAFT_MODEL_NAME_OR_PATH:-none}  dtype=${DRAFT_MODEL_DTYPE}"
echo "  target_hs_warmup_steps=${TARGET_HS_WARMUP_STEPS}"
echo "  deepspeed=${DEEPSPEED_CONFIG:-none}  output=${OUTPUT_DIR}"

case "${LAUNCH}" in
  torchrun)
    torchrun --nproc_per_node="${NPROC}" "${ARGS[@]}"
    ;;
  python)
    unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT \
      TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS \
      TORCHELASTIC_USE_AGENT_STORE 2>/dev/null || true
    "${PYTHON}" "${ARGS[@]}"
    ;;
  *)
    echo "ERROR: LAUNCH must be torchrun or python, got: ${LAUNCH}" >&2
    exit 1
    ;;
esac
