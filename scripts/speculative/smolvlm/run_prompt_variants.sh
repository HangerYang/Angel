#!/usr/bin/env bash
# Follow-up to run_prompt_style_ab.sh: the papers' appended
# "Please answer with an explanation." did nothing on the 5 VQA benchmarks
# (SmolVLM-256M treats it as more question text). Try whole-prompt variants
# instead. Target-only, 10 prompts, temp 0 -- same protocol as the A/B.
#
#   bash scripts/speculative/smolvlm/run_prompt_variants.sh
#   python my_angel/make_prompt_style_ab.py

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

# Only the benchmarks where the paper prompt failed.
DATASETS_VQA="Lin-Chen/MMStar MMMU/MMMU lmms-lab/textvqa lmms-lab/chartqa ai4math/mathvista"
STYLES="${STYLES:-detail_prefix cot describe_first min_words}"
NP="${NP:-10}"
OUT_BASE="${OUT_BASE:-my_angel/prompt_style_ab}"

mkdir -p logs
echo "=== prompt-variant sweep START $(date -Iseconds)  N=${NP} ==="

for style in ${STYLES}; do
  echo ""
  echo "########## style=${style}  $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=0 USE_EAGLE=0 TEMP=0 NUM_PROMPTS="${NP}" \
    PROMPT_STYLE="${style}" RUN_NAME="prompt_style_${style}" \
    OUT_ROOT="${OUT_BASE}/${style}" DATASETS="${DATASETS_VQA}" \
    bash scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh
  echo "########## style=${style} exit=$? $(date -Iseconds)"
done

echo ""
echo "=== prompt-variant sweep DONE $(date -Iseconds) ==="
