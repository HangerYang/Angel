#!/usr/bin/env bash
#
# ViSpec end-to-end pipeline (stages 1-3) for SmolVLM-256M (Idefics3 arch).
#
# Usage:
#   bash run_smolvlm.sh                 # all stages
#   bash run_smolvlm.sh 1 1.1 1.2       # data generation only
#   bash run_smolvlm.sh 3               # evaluation only
#
# Data:
#   * Stage 1.1: Aeala/ShareGPT_Vicuna_unfiltered (HF auto-download)
#   * Stage 1.2: LLaVA-Pretrain at LLAVA_DATA_PATH (default: /data/llava_datasets/...)
#   * Stage 3: HuggingFaceM4/COCO (HF auto-download)

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
RESULT_DIR="vispec_data/smolvlm/results"
LOG_DIR="vispec_data/smolvlm/logs"

# ViSpec default sample range (68000 examples)
START=0
END=67999
STAGE1_EPOCH_STATE="state_20"

NUM_Q=2
DEPTH=3
TOP_K=8
TOTAL_TOKEN=30
MTP_STEPS=1
MAX_LEN=4096
TEMPERATURE="0.0"

mkdir -p "${TEXT_DATA_ROOT}" "${MM_DATA_ROOT}" "${CKPT_STAGE1}" "${CKPT_STAGE2}" "${RESULT_DIR}" "${LOG_DIR}"

# ─── Stage selection ────────────────────────────────────────────────────────
STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
  STAGES=(1 2 3)
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

# ─── Stage 3: evaluation + speedup ────────────────────────────────────────────
if run_stage 3; then
  echo "=== Stage 3: evaluation (COCO caption, GPU ${GPU_IDS[0]}) ==="
  SPEC_DIR="${SPEC_DIR:-${CKPT_STAGE2}}"
  BASE_OUT="${RESULT_DIR}/coco_caption_test/smolvlm_baseline"
  SPEC_OUT="${RESULT_DIR}/coco_caption_test/smolvlm_spec"

  echo "--- baseline ---"
  python -m vispec.evaluation.gen_baseline_answer_coco_caption \
    --base-model-path="${BASE_MODEL}" \
    --model-id=test \
    --bench-name="${BASE_OUT}/" \
    --spec-model-path="${SPEC_DIR}" \
    --temperature="${TEMPERATURE}"

  echo "--- speculative (ViSpec) ---"
  python -m vispec.evaluation.gen_spec_answer_coco_caption \
    --base-model-path="${BASE_MODEL}" \
    --model-id=test \
    --bench-name="${SPEC_OUT}/" \
    --spec-model-path="${SPEC_DIR}" \
    --num-q="${NUM_Q}" --depth="${DEPTH}" --top-k="${TOP_K}" \
    --total-token="${TOTAL_TOKEN}" --use-ours=True \
    --temperature="${TEMPERATURE}"

  echo "--- speedup report ---"
  BASE_JSONL="${BASE_OUT}/test-temperature-${TEMPERATURE}.jsonl"
  SPEC_JSONL="${SPEC_OUT}/test-temperature-${TEMPERATURE}.jsonl"
  python - "$BASE_JSONL" "$SPEC_JSONL" <<'PY'
import json, sys
base_f, spec_f = sys.argv[1], sys.argv[2]

def toks_per_sec(path):
    tks, secs = [], []
    with open(path) as f:
        for line in f:
            c = json.loads(line)["choices"][0]
            tks.append(sum(c["new_tokens"]))
            secs.append(sum(c["wall_time"]))
    return sum(tks) / sum(secs), tks

def acc_len(path):
    vals = []
    with open(path) as f:
        for line in f:
            c = json.loads(line)["choices"][0]
            vals += c.get("acceptance_length", [])
    return sum(vals) / len(vals) if vals else float("nan")

b_tps, _ = toks_per_sec(base_f)
s_tps, _ = toks_per_sec(spec_f)
print(f"baseline    : {b_tps:6.2f} tok/s")
print(f"speculative : {s_tps:6.2f} tok/s")
print(f"mean accept : {acc_len(spec_f):.3f}")
print(f"SPEEDUP     : {s_tps / b_tps:.2f}x")
PY
fi

echo "All requested stages complete."
