#!/usr/bin/env bash
set -uo pipefail
cd /home/hyang/AngelSlim
source /home/hyang/miniconda3/etc/profile.d/conda.sh; conda activate angel
export PYTHONPATH=/home/hyang/AngelSlim
C=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3.json
run () {  # $1=name  $2=ckpt
  local d=my_angel/plan/runs/$1; mkdir -p "$d"
  echo "=== $1  $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=0 ANGELSLIM_EAGLE_LATENCY=1 \
  DATASET=opendatalab/OmniDocBench NUM_PROMPTS=20 OUTPUT_LEN=128 NUM_SPEC_TOKENS=4 \
  TEMP=0 MAX_NUM_SEQS=1 ALLOW_CUDA_GRAPHS=0 \
  DRAFT_MODEL="$2" DRAFT_MODEL_CONFIG_PATH=$C \
  OUTPUT_FILE="$d/results.json" \
  timeout 1800 bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh > "$d/run.log" 2>&1 \
    && echo "   ok" || echo "   FAILED (see $d/run.log)"
}
run 2b_1band_H   my_angel/plan/variants/2b_1band_H
run 2b_2band_2H  my_angel/plan/variants/2b_2band_2H
run 2b_3band_3H  my_angel/plan/variants/2b_3band_3H
run 5_eagle31    my_angel/plan/variants/5_eagle31
echo "=== all done $(date +%H:%M:%S)"
