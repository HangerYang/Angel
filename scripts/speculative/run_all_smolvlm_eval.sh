#!/usr/bin/env bash
#
# Every drafter arm x every working benchmark on the SmolVLM-256M target,
# through HiViS's own measurement protocol.
#
#   PYTHON=/path/to/python \
#   HIVIS=/path/to/a/utility/checkout/HiViS \
#   ANGELSLIM_ROOT=/path/to/a/feature-vistoken-attn-prune/checkout \
#   bash run_all_smolvlm_eval.sh
#
# Serial: one cell at a time on one GPU with the rest idle, because tok/s is
# wall clock and a parallel sweep silently measures system load instead. tau is
# a count ratio and survives a parallel run; nothing else here does.
#
# Resumable: a cell whose .json already exists is skipped, so a killed run
# continues where it stopped. Delete the file to force a re-run.
#
# Protocol comes from run_angelslim_eval.py's defaults, which are HiViS's own
# (max_new_tokens 500, 3 untimed warm-up prompts, seed 42, --max_input_tokens
# 4000, inference_mode, its decode path). Do not override them for a table that
# is meant to be comparable with HiViS's published numbers.
set -u

PYTHON=${PYTHON:?set PYTHON to an interpreter with the HiViS dependencies}
HIVIS=${HIVIS:?set HIVIS to the HiViS directory of a utility checkout}
# Only the reducer arms need this: pool/short load the training-side reduction
# code by path. It lives on feature/vistoken-attn-prune, not on utility, so it
# is a SEPARATE checkout. Leave unset to skip those arms.
ANGELSLIM_ROOT=${ANGELSLIM_ROOT:-}
VISTOKEN=${VISTOKEN:-/home/hyang/Angel/output/vistoken}
QSAMPLER=${QSAMPLER:-/home/hyang/Angel/output/qs-e512-tbnoce-mqt64/final/qsampler.pt}
# A ViSpec draft trained for THIS target. ViSpec's published checkpoints are
# Qwen-sized and will not load here. Leave unset to skip the arm.
VISPEC_CKPT=${VISPEC_CKPT:-}
HIVIS_CKPT=${HIVIS_CKPT:-/home/hyang/tmp/hivis_way_smolvlm256m_stage1_20ep_export}

OUT=${OUT:-/home/hyang/Angel/my_angel/smolvlm_sweep}
GPU=${GPU:-0}
N=${N:-80}

# Eleven of the thirteen names supported_benchmarks() reports.
#   gqa       needs eval_data/llava_v1_5_mix665k/images/gqa/images, absent here
#   seedbench benchmark_data.py reads row["question"]; the jsonl field is "text"
# Four of the eleven read images from directories git does not track and are
# therefore NOT portable to another machine: mme, mmvet, textvqa (and gqa).
BENCHES=${BENCHES:-"textvqa ScienceQA vqav2 mme mmvet MathVista ChartQA mmmu DocVQA mmmu_history omnidocbench"}

# tag|draft_method|checkpoint|extra flags
# Ordered cheapest first so a short run still covers the important arms.
ARMS=${ARMS:-"
naive|angelslim_eagle3|$VISTOKEN/full_c4|--naive
full_c4|angelslim_eagle3|$VISTOKEN/full_c4|
noimg_c4|angelslim_eagle3|$VISTOKEN/noimg_c4|
full_c20|angelslim_eagle3|$VISTOKEN/full_c20|
hivis|hivis|$HIVIS_CKPT|
vispec|vispec|$VISPEC_CKPT|
pool64_c4|angelslim_eagle3|$VISTOKEN/pool64_c4|--draft_image_reduce pool --draft_image_factor 13
e64_c4|angelslim_eagle3|$VISTOKEN/e64_c4|--draft_image_reduce short --draft_image_edge 512
qs16_c4|angelslim_eagle3|$VISTOKEN/qs16_c4|--draft_image_reduce short --draft_image_edge 512
"}

mkdir -p "$OUT/logs"
cd "$HIVIS" || exit 1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

echo "=== SmolVLM-256M sweep :: GPU $GPU, n=$N :: $(date '+%F %T') ==="
echo "    out=$OUT"
[[ -z "$ANGELSLIM_ROOT" ]] && echo "    ANGELSLIM_ROOT unset -> skipping pool/e64/qs16"
[[ -z "$VISPEC_CKPT"    ]] && echo "    VISPEC_CKPT unset -> skipping vispec"

for bench in $BENCHES; do
  echo "--- $bench  $(date '+%T')"
  while IFS='|' read -r tag method ckpt extra; do
    [[ -z "${tag// }" ]] && continue
    json="$OUT/${bench}__${tag}.json"
    [[ -f "$json" ]] && { echo "    $tag  done"; continue; }
    [[ -z "$ckpt" ]] && { echo "    $tag  SKIP (no checkpoint configured)"; continue; }
    # A directory that exists but holds no weights is a run still training;
    # scoring it would silently measure the wrong model.
    if [[ -d "$ckpt" && ! -f "$ckpt/model.safetensors" ]]; then
      echo "    $tag  SKIP (no model.safetensors in $ckpt)"; continue
    fi
    root_flag=""
    if [[ "$extra" == *--draft_image_reduce* ]]; then
      [[ -z "$ANGELSLIM_ROOT" ]] && { echo "    $tag  SKIP (needs ANGELSLIM_ROOT)"; continue; }
      root_flag="--draft_image_root $ANGELSLIM_ROOT"
    fi
    # The qsampler arm's budget must match what its drafter was trained on;
    # both sides read the same two variables so they cannot drift apart.
    qs_env=()
    if [[ "$tag" == qs16* ]]; then
      [[ -f "$QSAMPLER" ]] || { echo "    $tag  SKIP (no qsampler at $QSAMPLER)"; continue; }
      qs_env=(VISTOKEN_QSAMPLER_CKPT="$QSAMPLER" VISTOKEN_QSAMPLER_N=16)
    fi

    echo "    $tag  $(date '+%T')"
    env "${qs_env[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" run_angelslim_eval.py \
      --draft_method "$method" --draft "$ckpt" \
      --dataset "$bench" --n "$N" \
      $extra $root_flag --out "$json" \
      > "$OUT/logs/${bench}__${tag}.log" 2>&1
    status=$?
    [[ $status -eq 0 ]] || echo "      FAILED exit=$status -- see $OUT/logs/${bench}__${tag}.log"
  done <<< "$ARMS"
done

echo "=== DONE $(date '+%F %T') ==="
echo "tau (EAGLE, accept_length+1) and mean_accept_length (HiViS) are both in"
echo "each JSON's metrics, as are tok_per_s / _macro / _macro_retok. Compare"
echo "arms only on prompts where every arm produced identical text -- they do"
echo "not always, and tok/s rises with output length."
