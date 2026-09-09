#!/usr/bin/env bash
# Target-side image-compression sweep for SmolVLM-256M on VLMEvalKit.
#
# Two families of arm, both driven by env vars read in vlmeval/vlm/smolvlm.py:
#
#   pixel     SMOLVLM_LONGEST_EDGE=<edge>    downsize before tiling (the original
#                                            three arms; edge 512 emits exactly one
#                                            tile, the <global-img> thumbnail)
#   select    SMOLVLM_TOKEN_SELECT=<spec>    encode at full resolution, then keep a
#                                            subset of the 832 rows
#
# Arm zero is `global_only`: a full-resolution pass keeping only the last 64 rows
# (the <global-img> tile). It MUST reproduce the img64 pixel arm to within ~1
# point. If it does not, the inputs_embeds surgery is broken and every selector
# number after it is garbage. Run it before trusting anything else.
set -uo pipefail

VEK=${VEK:-/home/hyang/VLMEvalKit}
PY=${PY:-/home/hyang/miniconda3/envs/eval/bin/python}

# transformers is PINNED. Measured on ChartQA_TEST, SmolVLM-256M, 832 rows:
#   4.54.0   Overall 55.44   (published 55.6)
#   4.57.6   Overall 38.88   (-16.6, both splits crushed, predictions simply wrong
#                             -- not a scoring or formatting artefact)
# Installed side-by-side rather than into the env:
#   pip install --target $TF_PREFIX transformers==4.54.0
TF_PREFIX=${TF_PREFIX:-$HOME/tmp/vek/tf454}
OUT=${OUT:-$HOME/tmp/vek/out}
GPUS=${GPUS:-0,1,2,3}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")

# The three that can rank methods. ScienceQA / MMStar / MMMU move 0-2 points at
# 64 rows and cannot separate anything -- run them only to confirm a winner.
DATASETS=${DATASETS:-"ChartQA_TEST TextVQA_VAL DocVQA_VAL"}

# tag:kind:value   kind = edge | select | none
ARMS=${ARMS:-"\
img64:edge:512 \
global_only:select:global_only \
stride64:select:stride:64 \
random64:select:random:64 \
norm64:select:norm:64 \
divprune64:select:divprune:64 \
cdpruner64:select:cdpruner:64 \
cdpruner16:select:cdpruner:16 \
cdpruner8:select:cdpruner:8 \
img832:none: \
img320:edge:1024"}

# Rows of the <global-img> tile are the pixel baseline in disguise: a selector
# free to pick them and doing so has only rediscovered img64. Excluded from the
# candidate pool by default; the per-arm stats file records the fraction anyway.
export SMOLVLM_GLOBAL_POLICY=${SMOLVLM_GLOBAL_POLICY:-exclude}
export SMOLVLM_SELECT_SEED=${SMOLVLM_SELECT_SEED:-0}

mkdir -p "$OUT/logs"

for data in $DATASETS; do
  # Decode this dataset's images once, single-process, before any torchrun.
  # Concurrent ranks racing on the same half-written JPEG cost 68 minutes of
  # NCCL timeout on DocVQA_VAL; see warm_images.py.
  if [ ! -f "$OUT/logs/.warm_$data" ]; then
    echo "warm  $data (single-process image extraction)"
    ( cd "$VEK" && PYTHONPATH="$TF_PREFIX:$VEK" \
      "$PY" /home/hyang/AngelSlim/my_angel/vlmevalkit/warm_images.py "$data" ) \
      > "$OUT/logs/warm__${data}.log" 2>&1 && touch "$OUT/logs/.warm_$data"
  fi

  for arm in $ARMS; do
    tag=${arm%%:*}; rest=${arm#*:}; kind=${rest%%:*}; value=${rest#*:}

    if compgen -G "$OUT/$tag/SmolVLM-256M/SmolVLM-256M_${data}_*acc.csv" > /dev/null; then
      echo "skip  $data / $tag (scored)"; continue
    fi

    unset SMOLVLM_LONGEST_EDGE SMOLVLM_TOKEN_SELECT
    case "$kind" in
      edge)   export SMOLVLM_LONGEST_EDGE="$value" ;;
      select) export SMOLVLM_TOKEN_SELECT="$value" ;;
      # Downscale first, then select: trades sharpness for coverage. At edge
      # 2048 a 64-row budget sees 8.3% of the crop rows; at 1024 it sees 25%.
      # Written both:<edge>:<method>:<budget>.
      both)   export SMOLVLM_LONGEST_EDGE="${value%%:*}"
              export SMOLVLM_TOKEN_SELECT="${value#*:}" ;;
      none)   ;;
      *) echo "bad arm spec: $arm" >&2; exit 2 ;;
    esac
    export SMOLVLM_SELECT_STATS="$OUT/$tag/select_stats.jsonl"
    mkdir -p "$OUT/$tag"

    # Judge policy. Without --judge, run.py routes every MCQ / Y-N dataset to
    # gpt-4o-mini (run.py:344-357), and LOCAL_LLM in .env then hijacks that to
    # the local Qwen3-235B -- silently changing the scoring convention for MCQ
    # sets relative to the earlier judge-free table. Keep MCQ deterministic and
    # spend the judge only where the benchmark genuinely needs free-form grading.
    case "$data" in
      MMVet|MathVista_MINI) judge=qwen3-4b ;;
      *)                    judge=exact_matching ;;
    esac

    echo "run   $data / $tag  (${kind}=${value:-default}, judge=$judge)"
    ( cd "$VEK" && PYTHONPATH="$TF_PREFIX:$VEK" CUDA_VISIBLE_DEVICES=$GPUS \
      timeout "${RUN_TIMEOUT:-45m}" \
      "$PY" -m torch.distributed.run \
        --nproc-per-node=$NPROC --master-port=$((29500 + RANDOM % 1000)) \
        run.py --model SmolVLM-256M --data "$data" --judge "$judge" \
        --work-dir "$OUT/$tag" --reuse ) \
      > "$OUT/logs/${data}__${tag}.log" 2>&1
    echo "      exit=$? $(ls $OUT/$tag/SmolVLM-256M/*${data}*acc.csv 2>/dev/null | head -1)"
  done
done
