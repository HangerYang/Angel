#!/usr/bin/env bash
# Offline vLLM Eagle3 eval for SmolVLM (+ optional trained draft).
#
# Reads draft depth / aux layers from the checkpoint (1- or multi-layer):
#   num_hidden_layers
#   eagle_aux_hidden_state_layer_ids   (vLLM)
#   aux_hidden_states_layer_ids        (train; used to derive eagle_aux if needed)
#
# Output (default):
#   {DRAFT_MODEL}/eval/{data}/results.jsonl
#   {DRAFT_MODEL}/eval/{data}_miracle/results.jsonl   # MIRACLE_MODE=1
#   results/baseline/eval/{data}/results.jsonl        # USE_EAGLE=0
#
# Examples:
#   DRAFT_MODEL=output/smolvlm_256m_hawk/checkpoint-30000 \
#     DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-hawk.json \
#     bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
#
#   USE_EAGLE=0 bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
#
#   MIRACLE_MODE=1 DRAFT_MODEL=output/smolvlm_256m_hawk/checkpoint-30000 \
#     DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-hawk.json \
#     bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
#
# After git pull, refresh local vLLM patches (universal):
#   bash third_party/sync_vllm_latest.sh && source third_party/env.sh
#
# See scripts/speculative/smolvlm/README.md § Eval + "Where to update vLLM".

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -f "${ROOT}/third_party/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

CONFIG_DIR=angelslim/compressor/speculative/train/configs

# TARGET_MODEL — frozen target VLM for vLLM. HF id or local path.
#   Default: HuggingFaceTB/SmolVLM-256M-Instruct
TARGET_MODEL="${TARGET_MODEL:-HuggingFaceTB/SmolVLM-256M-Instruct}"

# DRAFT_MODEL — draft checkpoint dir (or parent OUTPUT_DIR; script picks latest checkpoint-*).
#   Options:
#     path/to/checkpoint-NNNN   — preferred (has config.json + weights)
#     path/to/output_dir        — auto-resolves newest checkpoint-*
#   Required when USE_EAGLE=1. Ignored for baseline (USE_EAGLE=0).
DRAFT_MODEL="${DRAFT_MODEL:-output/smolvlm_256m_eagle3_online}"

# DRAFT_MODEL_CONFIG_PATH — train-time draft JSON used to fill missing vLLM fields
#   (num_hidden_layers, eagle_aux_hidden_state_layer_ids, injection mode).
#   Options: same configs as train (eagle3 | progressive | hawk | real_hawk | ...).
#   Example: .../smolvlm-256m-hawk.json
DRAFT_MODEL_CONFIG_PATH="${DRAFT_MODEL_CONFIG_PATH:-${CONFIG_DIR}/smolvlm-256m-eagle3.json}"

# DATASET — HF dataset id or local jsonl path.
#   Examples: lmms-lab/textvqa | lmms-lab/MMMU | path/to/data.jsonl
DATASET="${DATASET:-lmms-lab/textvqa}"

# OUTPUT_FILE — results jsonl path. Empty (default) → auto layout under draft/eval/<data>/...
#   Override anytime: OUTPUT_FILE=results/my_run.jsonl
OUTPUT_FILE="${OUTPUT_FILE:-}"

# USE_EAGLE — enable speculative Eagle3 draft.
#   Options: 1 (default, use draft) | 0 (baseline target-only, no draft)
USE_EAGLE="${USE_EAGLE:-1}"

# MIRACLE_MODE — oracle GT-HS along target trajectory (upper bound).
#   Progressive/hawk steps 1+ match train warmup (L1/L2 GT, L0 last residual).
#   Options: 0 (default, normal eagle) | 1 (miracle capture+use; forces MAX_NUM_SEQS=1)
#   Needs TEMP=0. Alias: ASSISTANCE_MODE (deprecated).
MIRACLE_MODE="${MIRACLE_MODE:-0}"
# Back-compat: old ASSISTANCE_MODE name now means miracle.
if [[ "${ASSISTANCE_MODE:-0}" != "0" && "${MIRACLE_MODE}" == "0" ]]; then
  echo "NOTE: ASSISTANCE_MODE is deprecated; treating as MIRACLE_MODE=1" >&2
  MIRACLE_MODE="${ASSISTANCE_MODE}"
fi

# NUM_PROMPTS — how many dataset examples to eval. Typical: 2 (smoke) | 80 | 200
NUM_PROMPTS="${NUM_PROMPTS:-80}"

# NUM_SPEC_TOKENS — speculative draft depth K (tokens proposed per verify). Typical: 3|4|5
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}"

# MAX_NUM_SEQS — vLLM concurrent sequences. Miracle forces 1. Typical: 1 (acceptance metrics) | 8+
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

# MAX_MODEL_LEN — vLLM context length cap. Must cover prompt+output. Typical: 4096|8192
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# OUTPUT_LEN — max new tokens to generate per prompt. Typical: 32 (smoke) | 256|1024
OUTPUT_LEN="${OUTPUT_LEN:-1024}"

# TEMP — sampling temperature. Use 0 for greedy / miracle. Typical: 0 | 0.7
TEMP="${TEMP:-0}"

# TP — tensor-parallel size across GPUs for one engine. Typical: 1 (SmolVLM-256M)
TP="${TP:-1}"

# GPU_MEMORY_UTILIZATION — vLLM GPU mem fraction. Typical: 0.7|0.9
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# CUDA_VISIBLE_DEVICES — which GPU(s) this eval sees. Example: 0 | 0,1
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# MIRACLE_GT_DIR — fixed root for phase-A GT tokens (shared across drafts).
#   Layout: <MIRACLE_GT_DIR>/<dataset_name>/gt_tokens.json (+ meta.json).
#   Reused if present and meta matches target/temp/output_len with enough prompts.
MIRACLE_GT_DIR="${MIRACLE_GT_DIR:-results/miracle_gt}"

# MIRACLE_HS_DIR — optional dir for phase-B aux tapes ({i:05d}.pt). Empty = under output.
#   Example: MIRACLE_HS_DIR=results/miracle_smoke_hs
MIRACLE_HS_DIR="${MIRACLE_HS_DIR:-}"

# DEBUG — 1 enables --debug (EngineCore in-process for breakpoints). Options: 0|1
# PDB   — 1 enables --pdb (ipdb on breakpoint()). Options: 0|1
DEBUG_ARGS=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  DEBUG_ARGS+=(--debug)
fi
if [[ "${PDB:-0}" == "1" ]]; then
  DEBUG_ARGS+=(--pdb)
fi

export CUDA_VISIBLE_DEVICES
_MIRACLE_ON=0
case "${MIRACLE_MODE,,}" in
  1|true|yes|on) _MIRACLE_ON=1 ;;
esac
if [[ "${_MIRACLE_ON}" == "1" ]]; then
  MIRACLE_MODE=1
  MAX_NUM_SEQS=1
fi

# Trainer often leaves only checkpoint-*/config.json under OUTPUT_DIR.
resolve_draft_model() {
  local d="$1"
  if [[ -f "${d}/config.json" ]]; then
    printf '%s' "${d}"
    return
  fi
  local latest=""
  local c
  for c in "${d}"/checkpoint-*; do
    [[ -d "${c}" && -f "${c}/config.json" ]] || continue
    if [[ -z "${latest}" || "${c}" -nt "${latest}" ]]; then
      latest="${c}"
    fi
  done
  if [[ -n "${latest}" ]]; then
    echo "DRAFT_MODEL has no config.json; using latest checkpoint: ${latest}" >&2
    printf '%s' "${latest}"
    return
  fi
  printf '%s' "${d}"
}

# lmms-lab/textvqa → textvqa ; path/to/foo.jsonl → foo
dataset_folder_name() {
  local ds="$1"
  local base
  base="$(basename "${ds}")"
  base="${base%.jsonl}"
  base="${base%.json}"
  printf '%s' "${base}"
}

EXTRA=()
if [[ "${USE_EAGLE}" == "1" ]]; then
  if [[ ! -d "${DRAFT_MODEL}" ]]; then
    echo "ERROR: DRAFT_MODEL directory not found: ${DRAFT_MODEL}" >&2
    echo "Train a draft first (and ensure it is saved), e.g.:" >&2
    echo "  SAVE_STRATEGY=epoch bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh" >&2
    exit 1
  fi
  DRAFT_MODEL="$(resolve_draft_model "${DRAFT_MODEL}")"
  if [[ ! -f "${DRAFT_MODEL}/config.json" ]]; then
    echo "ERROR: no config.json under DRAFT_MODEL=${DRAFT_MODEL}" >&2
    echo "  Point DRAFT_MODEL at a checkpoint dir, e.g.:" >&2
    echo "    DRAFT_MODEL=output/smolvlm_256m_hawk/checkpoint-30000" >&2
    exit 1
  fi

  PREPARE_ARGS=(
    --draft_model "${DRAFT_MODEL}"
  )
  if [[ -n "${DRAFT_MODEL_CONFIG_PATH}" && -f "${DRAFT_MODEL_CONFIG_PATH}" ]]; then
    PREPARE_ARGS+=(--draft_model_config_path "${DRAFT_MODEL_CONFIG_PATH}")
  fi
  if [[ "${MIRACLE_MODE}" == "1" ]]; then
    PREPARE_ARGS+=(--eagle_miracle_mode)
  fi
  python3 scripts/speculative/smolvlm/prepare_draft_config_for_vllm_eval.py \
    "${PREPARE_ARGS[@]}"

  EXTRA+=(
    --draft_model "${DRAFT_MODEL}"
    --use_eagle
    --num_spec_tokens "${NUM_SPEC_TOKENS}"
  )
  if [[ "${MIRACLE_MODE}" == "1" ]]; then
    EXTRA+=(--eagle_miracle_mode --miracle_gt_dir "${MIRACLE_GT_DIR}")
    if [[ -n "${MIRACLE_HS_DIR}" ]]; then
      EXTRA+=(--miracle_hs_dir "${MIRACLE_HS_DIR}")
    fi
    echo "MIRACLE_MODE=1 — oracle GT target-HS along target trajectory"
    echo "  draft_model=${DRAFT_MODEL}"
    echo "  miracle_gt_dir=${MIRACLE_GT_DIR}  (phase A cache by dataset name)"
    echo "  (phase A: GT load-or-generate, B: capture tape, C: timed eagle+inject)"
  fi
else
  echo "USE_EAGLE=0 — baseline eval (no draft / speculative decoding)"
fi

# Default output layout (only if OUTPUT_FILE unset):
#   draft/eval/<data>/results.jsonl
#   draft/eval/<data>_miracle/results.jsonl
DATA_NAME="$(dataset_folder_name "${DATASET}")"
if [[ "${MIRACLE_MODE}" == "1" ]]; then
  DATA_NAME="${DATA_NAME}_miracle"
fi
if [[ -z "${OUTPUT_FILE}" ]]; then
  if [[ "${USE_EAGLE}" == "1" ]]; then
    OUTPUT_FILE="${DRAFT_MODEL}/eval/${DATA_NAME}/results.jsonl"
  else
    OUTPUT_FILE="results/baseline/eval/${DATA_NAME}/results.jsonl"
  fi
fi
mkdir -p "$(dirname "${OUTPUT_FILE}")"

CMD=(
  python3 tools/vllm_offline_eagle3_vlm_batch.py
  --target_model "${TARGET_MODEL}"
  --dataset "${DATASET}"
  --num_prompts "${NUM_PROMPTS}"
  --temp "${TEMP}"
  --max_num_seqs "${MAX_NUM_SEQS}"
  --max_model_len "${MAX_MODEL_LEN}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --output_len "${OUTPUT_LEN}"
  --tp "${TP}"
  --output_file "${OUTPUT_FILE}"
)
CMD+=("${EXTRA[@]}")
if [[ ${#DEBUG_ARGS[@]} -gt 0 ]]; then
  CMD+=("${DEBUG_ARGS[@]}")
fi
echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Results: ${OUTPUT_FILE}"
