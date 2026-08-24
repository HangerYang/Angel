#!/usr/bin/env bash
# A/B the generation prompts: raw dataset question vs the EAGLE-3 VLM papers'
# task prompts (--prompt_style verbose). Target-only (no draft), 10 prompts per
# benchmark -- output length is a property of the target, so the draft would
# only cost time here.
#
#   bash scripts/speculative/smolvlm/run_prompt_style_ab.sh
#
# Then summarise with:  python my_angel/make_prompt_style_ab.py

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

DATASETS_ALL="Lin-Chen/MMStar MMMU/MMMU opendatalab/OmniDocBench HuggingFaceH4/MATH-500 lmms-lab/textvqa lmms-lab/chartqa ai4math/mathvista lmms-lab/COCO-Caption"
NP="${NP:-10}"
OUT_BASE="${OUT_BASE:-my_angel/prompt_style_ab}"

mkdir -p logs
echo "=== prompt-style A/B START $(date -Iseconds)  N=${NP} ==="

for style in raw verbose; do
  echo ""
  echo "########## style=${style}  $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=0 USE_EAGLE=0 TEMP=0 NUM_PROMPTS="${NP}" \
    PROMPT_STYLE="${style}" RUN_NAME="prompt_style_${style}" \
    OUT_ROOT="${OUT_BASE}/${style}" DATASETS="${DATASETS_ALL}" \
    bash scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh
  echo "########## style=${style} exit=$? $(date -Iseconds)"
done

echo ""
echo "=== prompt-style A/B DONE $(date -Iseconds) ==="
