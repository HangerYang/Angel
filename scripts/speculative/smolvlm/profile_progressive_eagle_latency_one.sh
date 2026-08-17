#!/usr/bin/env bash
# Profile one progressive EAGLE vLLM generation with coarse CUDA-synchronized stages.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

export PATH="${PYTHON_BIN_DIR:-/home/hyang/miniconda3/envs/angel/bin}:$PATH"

DRAFT_MODEL="${DRAFT_MODEL:-/home/hyang/AngelSlim/output/smolvlm_256m_eagle3_progressive_nccl/checkpoint-66466}"
DRAFT_MODEL_CONFIG_PATH="${DRAFT_MODEL_CONFIG_PATH:-angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive.json}"
DATASET="${DATASET:-lmms-lab/textvqa}"
OUT_ROOT="${OUT_ROOT:-${DRAFT_MODEL%/}/latency_profile_one}"
LOG_FILE="${LOG_FILE:-${OUT_ROOT}/profile.log}"

mkdir -p "${OUT_ROOT}"

ANGELSLIM_EAGLE_LATENCY=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
DRAFT_MODEL="${DRAFT_MODEL}" \
DRAFT_MODEL_CONFIG_PATH="${DRAFT_MODEL_CONFIG_PATH}" \
DATASET="${DATASET}" \
NUM_PROMPTS="${NUM_PROMPTS:-1}" \
OUTPUT_LEN="${OUTPUT_LEN:-64}" \
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" \
TEMP="${TEMP:-0}" \
OUTPUT_FILE="${OUT_ROOT}/results.json" \
ACCEPTANCE_METRICS_FILE="${OUT_ROOT}/acceptance_metrics.json" \
bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh 2>&1 | tee "${LOG_FILE}"

python3 - "${LOG_FILE}" <<'PY'
import json
import re
import sys
from collections import defaultdict

pat = re.compile(r"\[ANGELSLIM_EAGLE_LATENCY\] (\{.*\})")
events = []
in_generation = False
for line in open(sys.argv[1], errors="replace"):
    if "Starting generation" in line:
        in_generation = True
    m = pat.search(line)
    if not in_generation or not m:
        continue
    events.append(json.loads(m.group(1)))

by_name = defaultdict(list)
for event in events:
    by_name[event["event"]].append(float(event["ms"]))

print("\nLatency summary from actual generation only:", sys.argv[1])
print("| event | calls | total ms | mean ms |")
print("|---|---:|---:|---:|")
for name in sorted(by_name):
    vals = by_name[name]
    print(f"| {name} | {len(vals)} | {sum(vals):.3f} | {sum(vals)/len(vals):.3f} |")

print("\nFirst 40 raw timing events:")
for event in events[:40]:
    print(json.dumps(event, sort_keys=True))
PY
