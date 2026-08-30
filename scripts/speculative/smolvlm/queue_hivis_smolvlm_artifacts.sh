#!/usr/bin/env bash
# Waits for the Qwen2.5-VL-7B EAGLE3 data-gen run (which currently holds all
# 8 GPUs) to finish, then:
#   1. Runs a tiny single-GPU smoke test of the HiViS SmolVLM-256M-Instruct
#      two-stage recipe (generate -> stage1 -> stage2) against the
#      angelslim-smolvlm-eagle3-artifacts dataset, to catch config/data-format
#      bugs before committing GPU-hours.
#   2. If the smoke test succeeds, runs the real 8-GPU stage1(2ep)+stage2(1ep)
#      run. If it fails, this script exits (via set -e) WITHOUT starting the
#      real run.
set -euo pipefail

QWEN_DATAGEN_LOG=/home/hyang/Angel/dataset/qwen2_5_vl_7b_target_gen_full_run.log
QWEN_DATAGEN_PID=3569633

echo "[$(date --iso-8601=seconds)] Waiting for Qwen2.5-VL-7B data-gen (pid $QWEN_DATAGEN_PID) to finish..."
while true; do
  if grep -q "Full run finished." "$QWEN_DATAGEN_LOG" 2>/dev/null; then
    echo "[$(date --iso-8601=seconds)] Qwen data-gen log reports completion."
    break
  fi
  if ! kill -0 "$QWEN_DATAGEN_PID" 2>/dev/null; then
    echo "[$(date --iso-8601=seconds)] Qwen data-gen process no longer running (pid $QWEN_DATAGEN_PID gone)."
    break
  fi
  sleep 30
done

cd /home/hyang/Angel
DATA_FILE_COMMON=/home/hyang/Angel/dataset/angelslim-smolvlm-eagle3-artifacts/data/smolvlm256m/train_path_b64.jsonl

echo "[$(date --iso-8601=seconds)] === Smoke test: HiViS SmolVLM-256M-Instruct, 1 GPU, 40 samples, 1 epoch each stage ==="
GPUS="0" \
DATA_FILE="$DATA_FILE_COMMON" \
HIVIS_DATA_ROOT=/home/hyang/Angel/dataset/hivis_smolvlm_256m_generated_artifacts_smoke \
OUTPUT_ROOT=/home/hyang/Angel/output/hivis_official/smolvlm_256m_artifacts_smoke \
STAGE1_DIR=/home/hyang/Angel/output/hivis_official/smolvlm_256m_artifacts_smoke/stage1 \
STAGE2_DIR=/home/hyang/Angel/output/hivis_official/smolvlm_256m_artifacts_smoke/stage2 \
STAGE1_CKPT=/home/hyang/Angel/output/hivis_official/smolvlm_256m_artifacts_smoke/stage1/state_0 \
START=0 END=40 \
STAGE1_EPOCHS=1 STAGE2_EPOCHS=1 \
BS_STAGE1=2 BS_STAGE2=1 \
STAGE=all \
bash scripts/speculative/smolvlm/train_official_hivis_smolvlm.sh
echo "[$(date --iso-8601=seconds)] Smoke test passed."

echo "[$(date --iso-8601=seconds)] === Full run: HiViS SmolVLM-256M-Instruct, 8 GPUs, stage1(2ep)+stage2(1ep) ==="
GPUS="0 1 2 3 4 5 6 7" \
DATA_FILE="$DATA_FILE_COMMON" \
HIVIS_DATA_ROOT=/home/hyang/Angel/dataset/hivis_smolvlm_256m_generated_artifacts \
OUTPUT_ROOT=/home/hyang/Angel/output/hivis_official/smolvlm_256m_artifacts_stage1_2ep_stage2_1ep \
STAGE1_DIR=/home/hyang/Angel/output/hivis_official/smolvlm_256m_artifacts_stage1_2ep_stage2_1ep/stage1 \
STAGE2_DIR=/home/hyang/Angel/output/hivis_official/smolvlm_256m_artifacts_stage1_2ep_stage2_1ep/stage2 \
STAGE1_CKPT=/home/hyang/Angel/output/hivis_official/smolvlm_256m_artifacts_stage1_2ep_stage2_1ep/stage1/state_0 \
STAGE1_EPOCHS=2 STAGE2_EPOCHS=1 \
STAGE=all \
bash scripts/speculative/smolvlm/train_official_hivis_smolvlm.sh
echo "[$(date --iso-8601=seconds)] HiViS SmolVLM stage1+stage2 full run finished."
