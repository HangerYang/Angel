#!/bin/bash
# Pre-warm the oracle gist embedding cache + the final .map_cache, sharded
# across all GPUs, before launching the real multi-GPU training run.
#
# Why this exists: train_eagle3_online.py builds the online dataset
# independently on every torchrun rank with no rank-0-first gating. With
# gist_conditioning=true that means every rank spins up its own
# SentenceTransformer encoder -- all pinned to the *same* hardcoded
# gist_encoder_device (usually "cuda:0"), so they pile onto one physical GPU
# and race-write the same .gist_cache / .map_cache files. This script does
# the encoding once, sharded one-worker-per-GPU (so it actually uses all your
# GPUs), merges the result into the canonical cache location, then builds the
# final .map_cache in a single process -- so the real NPROC>1 launch that
# follows just hits warm caches on every rank instead of recomputing.
#
# Usage:
#   DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-3.1-oracle-remaining-gist-r7.json \
#   TRAIN_DATA_PATH=/home/hyang/AngelSlim/dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl \
#   EVAL_DATA_PATH=/home/hyang/AngelSlim/dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl \
#     bash scripts/speculative/smolvlm/prewarm_gist_cache.sh
#
# Then launch the normal training command (LOAD_FROM_CACHE_FILE=true) -- it
# will find everything cached.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${DRY_RUN:-0}" != "1" && -f "${ROOT}/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

# DRAFT_MODEL_CONFIG_PATH — same draft config JSON you're about to train with.
DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:?set DRAFT_MODEL_CONFIG_PATH}

# TARGET_MODEL_NAME_OR_PATH / TRAIN_DATA_PATH / EVAL_DATA_PATH — must match
# the values you'll pass to train_eagle3_vlm_online.sh exactly (same file
# path/size/mtime feed the cache key).
TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolVLM-256M-Instruct}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:?set TRAIN_DATA_PATH}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-}

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-smolvlm}
MODAL_TYPE=${MODAL_TYPE:-VLM}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}

# PREWARM_GPUS — comma-separated physical GPU ids, one shard worker each.
#   Default: every GPU nvidia-smi reports. DRY_RUN=1 uses a CPU placeholder.
if [[ -z "${PREWARM_GPUS:-}" ]]; then
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    PREWARM_GPUS=0
  else
    PREWARM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
  fi
fi
IFS=',' read -r -a GPU_ARR <<< "${PREWARM_GPUS}"
NUM_SHARDS=${#GPU_ARR[@]}
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "ERROR: no GPUs found (PREWARM_GPUS=${PREWARM_GPUS})" >&2
  exit 1
fi

SAVE_EVERY=${SAVE_EVERY:-2000}
SCRATCH_DIR=${SCRATCH_DIR:-$(dirname "${TRAIN_DATA_PATH}")/.gist_prewarm_scratch}
WARM_MAP_CACHE=${WARM_MAP_CACHE:-true}
CLEAN_SCRATCH=${CLEAN_SCRATCH:-true}

GIST_CONDITIONING=$(python3 -c "import json,sys; print(bool(json.load(open(sys.argv[1])).get('gist_conditioning', False)))" "${DRAFT_MODEL_CONFIG_PATH}")
if [[ "${GIST_CONDITIONING}" != "True" ]]; then
  echo "gist_conditioning is false in ${DRAFT_MODEL_CONFIG_PATH}; nothing to prewarm." >&2
  exit 0
fi

echo "=== Prewarm oracle gist cache ==="
echo "  draft_config=${DRAFT_MODEL_CONFIG_PATH}"
echo "  train=${TRAIN_DATA_PATH}  eval=${EVAL_DATA_PATH:-none}"
echo "  shards=${NUM_SHARDS}  gpus=${PREWARM_GPUS}"
echo "  scratch=${SCRATCH_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: command validated, not launching cache prewarm."
  exit 0
fi

DATA_ARGS=(--data_path "${TRAIN_DATA_PATH}")
[[ -n "${EVAL_DATA_PATH}" ]] && DATA_ARGS+=("${EVAL_DATA_PATH}")

rm -rf "${SCRATCH_DIR}"
mkdir -p "${SCRATCH_DIR}"

PIDS=()
for i in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  shard_dir="${SCRATCH_DIR}/shard_${i}"
  mkdir -p "${shard_dir}"
  echo "  launching shard ${i} on physical GPU ${gpu} -> ${shard_dir}/log.txt"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/tools/prewarm_gist_cache.py" \
    --draft_model_config_path "${DRAFT_MODEL_CONFIG_PATH}" \
    --target_model_name_or_path "${TARGET_MODEL_NAME_OR_PATH}" \
    "${DATA_ARGS[@]}" \
    --gist_cache_dir "${shard_dir}" \
    --device cuda:0 \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --chat_template_type "${CHAT_TEMPLATE_TYPE}" \
    --modal_type "${MODAL_TYPE}" \
    --shuffle_seed "${SHUFFLE_SEED}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${i}" \
    --save_every "${SAVE_EVERY}" \
    > "${shard_dir}/log.txt" 2>&1 &
  PIDS+=($!)
done

echo "Waiting on ${NUM_SHARDS} shard workers (tail -f ${SCRATCH_DIR}/shard_*/log.txt to watch progress)..."
FAILED=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "ERROR: shard ${i} (pid ${PIDS[$i]}) failed; see ${SCRATCH_DIR}/shard_${i}/log.txt" >&2
    FAILED=1
  fi
done
if [[ "${FAILED}" -ne 0 ]]; then
  echo "One or more shard workers failed; not merging. Scratch kept at ${SCRATCH_DIR}." >&2
  exit 1
fi

echo "All shards done. Merging per-shard gist caches..."
python3 - "${DRAFT_MODEL_CONFIG_PATH}" "${TRAIN_DATA_PATH}" "${SCRATCH_DIR}" "${NUM_SHARDS}" <<'PY'
import json
import sys
from pathlib import Path

import torch

draft_config_path, train_data_path, scratch_dir, num_shards = sys.argv[1:5]
num_shards = int(num_shards)

cfg = json.loads(Path(draft_config_path).read_text())
model_name = cfg.get("gist_encoder_model_name_or_path", "Qwen/Qwen3-Embedding-0.6B")
refresh_every = max(1, int(cfg.get("gist_refresh_every", 4)))
safe_name = "".join(
    ch if ch.isalnum() or ch in "._-" else "_" for ch in model_name.rstrip("/").split("/")[-1]
)
filename = f"{safe_name}_r{refresh_every}.pt"

gist_cache_dir = cfg.get("gist_cache_dir")
target_dir = (
    Path(gist_cache_dir).expanduser()
    if gist_cache_dir
    else Path(train_data_path).resolve().parent / ".gist_cache"
)
target_dir.mkdir(parents=True, exist_ok=True)
target_path = target_dir / filename

merged = {}
if target_path.exists():
    merged.update(torch.load(target_path, map_location="cpu"))
    print(f"  base cache already had {len(merged)} entries")

for i in range(num_shards):
    shard_path = Path(scratch_dir) / f"shard_{i}" / filename
    if not shard_path.exists():
        print(f"  WARNING: {shard_path} missing, skipping")
        continue
    shard_cache = torch.load(shard_path, map_location="cpu")
    merged.update(shard_cache)
    print(f"  shard {i}: +{len(shard_cache)} entries ({shard_path})")

torch.save(merged, target_path)
print(f"Merged gist cache: {len(merged)} total entries -> {target_path}")
PY

if [[ "${CLEAN_SCRATCH}" == "true" ]]; then
  echo "Cleaning scratch dir ${SCRATCH_DIR}"
  rm -rf "${SCRATCH_DIR}"
fi

if [[ "${WARM_MAP_CACHE}" == "true" ]]; then
  echo "=== Warming final .map_cache (single process, should be fast now) ==="
  python "${ROOT}/tools/warm_map_cache.py" \
    --draft_model_config_path "${DRAFT_MODEL_CONFIG_PATH}" \
    --target_model_name_or_path "${TARGET_MODEL_NAME_OR_PATH}" \
    --train_data_path "${TRAIN_DATA_PATH}" \
    --eval_data_path "${EVAL_DATA_PATH}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --chat_template_type "${CHAT_TEMPLATE_TYPE}" \
    --modal_type "${MODAL_TYPE}" \
    --shuffle_seed "${SHUFFLE_SEED}"
fi

echo "=== Prewarm complete. Launch training normally with LOAD_FROM_CACHE_FILE=true. ==="
