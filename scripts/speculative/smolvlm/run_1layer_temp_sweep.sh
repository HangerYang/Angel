#!/usr/bin/env bash
# Sequential sweep: every 1-layer draft + the no-eagle target baseline,
# 8 benchmarks, temp 0 and temp 1. One vLLM job at a time on GPU 0 so the
# tok/s numbers stay comparable with the existing rerun tables.
#
#   bash scripts/speculative/smolvlm/run_1layer_temp_sweep.sh
#
# Cells that already have acceptance_metrics.json are skipped by the DP
# script (FORCE=1 to redo them), so the 16 existing temp0 core cells are
# reused and only the missing ones run.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

DATASETS_ALL="Lin-Chen/MMStar MMMU/MMMU opendatalab/OmniDocBench HuggingFaceH4/MATH-500 lmms-lab/textvqa lmms-lab/chartqa ai4math/mathvista lmms-lab/COCO-Caption"

CFG_DIR="angelslim/compressor/speculative/train/configs"

# name | draft checkpoint | train config | out root (temp appended)
RUNS=(
  "no_eagle_baseline|-|-|my_angel/no_eagle_baseline"
  "baseline_1layer|my_angel/eagle/baseline_1layer/checkpoint-66466|${CFG_DIR}/smolvlm-256m-eagle3.json|my_angel/eagle/baseline_1layer/rerun"
  "banded_mix_fc_3.1|my_angel/smolvlm-256m-eagle3-banded-mix-fc-3.1/checkpoint-66466|${CFG_DIR}/smolvlm-256m-eagle3-banded-mix-fc-3.1.json|my_angel/smolvlm-256m-eagle3-banded-mix-fc-3.1/rerun"
  "banded_mix_wide_3.1|my_angel/smolvlm-256m-eagle3-banded-mix-wide-3.1/checkpoint-66466|${CFG_DIR}/smolvlm-256m-eagle3-banded-mix-wide-3.1.json|my_angel/smolvlm-256m-eagle3-banded-mix-wide-3.1/rerun"
)

mkdir -p logs
echo "=== 1-layer temp sweep START $(date -Iseconds) ==="

for temp in 0 1; do
  for spec in "${RUNS[@]}"; do
    IFS='|' read -r name draft cfg out_base <<< "${spec}"
    out_root="${out_base}/temp${temp}"
    echo ""
    echo "########## ${name}  temp${temp}  -> ${out_root}  $(date -Iseconds)"

    if [[ "${name}" == "no_eagle_baseline" ]]; then
      CUDA_VISIBLE_DEVICES=0 USE_EAGLE=0 TEMP="${temp}" \
        RUN_NAME="${name}" OUT_ROOT="${out_root}" DATASETS="${DATASETS_ALL}" \
        bash scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh
    else
      CUDA_VISIBLE_DEVICES=0 TEMP="${temp}" \
        RUN_NAME="${name}" OUT_ROOT="${out_root}" DATASETS="${DATASETS_ALL}" \
        DRAFT_MODEL="${draft}" DRAFT_MODEL_CONFIG_PATH="${cfg}" \
        bash scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh
    fi
    echo "########## ${name} temp${temp} exit=$? $(date -Iseconds)"
  done
done

echo ""
echo "=== 1-layer temp sweep DONE $(date -Iseconds) ==="
