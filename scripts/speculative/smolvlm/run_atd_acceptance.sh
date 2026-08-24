#!/usr/bin/env bash
# Acceptance run under PROMPT_STYLE=answer_then_describe, on the VQA benchmarks
# whose raw outputs were too short for speculation to pay (8-20 output tokens).
# chartqa is excluded: no prompt moved it off ~140 tokens.
# Separate output root (rerun_atd) so the raw-prompt tables stay untouched.
#
#   bash scripts/speculative/smolvlm/run_atd_acceptance.sh
#   python my_angel/make_atd_results.py

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

DATASETS_ATD="${DATASETS_ATD:-Lin-Chen/MMStar MMMU/MMMU lmms-lab/textvqa ai4math/mathvista}"
CFG_DIR="angelslim/compressor/speculative/train/configs"
STYLE="${STYLE:-answer_then_describe}"
TEMP="${TEMP:-0}"
NP="${NP:-80}"

RUNS=(
  "no_eagle_baseline|-|-|my_angel/no_eagle_baseline/atd_temp${TEMP}"
  "baseline_1layer|my_angel/eagle/baseline_1layer/checkpoint-66466|${CFG_DIR}/smolvlm-256m-eagle3.json|my_angel/eagle/baseline_1layer/rerun_atd/temp${TEMP}"
  "banded_mix_fc_3.1|my_angel/smolvlm-256m-eagle3-banded-mix-fc-3.1/checkpoint-66466|${CFG_DIR}/smolvlm-256m-eagle3-banded-mix-fc-3.1.json|my_angel/smolvlm-256m-eagle3-banded-mix-fc-3.1/rerun_atd/temp${TEMP}"
  "banded_mix_wide_3.1|my_angel/smolvlm-256m-eagle3-banded-mix-wide-3.1/checkpoint-66466|${CFG_DIR}/smolvlm-256m-eagle3-banded-mix-wide-3.1.json|my_angel/smolvlm-256m-eagle3-banded-mix-wide-3.1/rerun_atd/temp${TEMP}"
)

mkdir -p logs
echo "=== ATD acceptance START $(date -Iseconds)  style=${STYLE} N=${NP} ==="

for spec in "${RUNS[@]}"; do
  IFS='|' read -r name draft cfg out_root <<< "${spec}"
  echo ""
  echo "########## ${name}  -> ${out_root}  $(date -Iseconds)"
  if [[ "${name}" == "no_eagle_baseline" ]]; then
    CUDA_VISIBLE_DEVICES=0 USE_EAGLE=0 TEMP="${TEMP}" NUM_PROMPTS="${NP}" PROMPT_STYLE="${STYLE}" \
      RUN_NAME="${name}" OUT_ROOT="${out_root}" DATASETS="${DATASETS_ATD}" \
      bash scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh
  else
    CUDA_VISIBLE_DEVICES=0 TEMP="${TEMP}" NUM_PROMPTS="${NP}" PROMPT_STYLE="${STYLE}" \
      RUN_NAME="${name}" OUT_ROOT="${out_root}" DATASETS="${DATASETS_ATD}" \
      DRAFT_MODEL="${draft}" DRAFT_MODEL_CONFIG_PATH="${cfg}" \
      bash scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh
  fi
  echo "########## ${name} exit=$? $(date -Iseconds)"
done

echo ""
echo "=== ATD acceptance DONE $(date -Iseconds) ==="
