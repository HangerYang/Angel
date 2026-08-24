#!/usr/bin/env bash
# Run true vLLM speculative acceptance eval across multiple benchmarks.
# This wraps eval_eagle3_vlm_batch.sh and summarizes acceptance_metrics.json.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

DRAFT_MODEL="${DRAFT_MODEL:-output/smolvlm_256m_eagle3_online}"
RUN_NAME="${RUN_NAME:-$(basename "${DRAFT_MODEL%/}")}"
OUT_ROOT="${OUT_ROOT:-${DRAFT_MODEL%/}/eval_acceptance}"
DATASETS="${DATASETS:-lmms-lab/textvqa MMMU/MMMU Lin-Chen/MMStar HuggingFaceH4/MATH-500}"

# Forwarded defaults. Override them exactly as in eval_eagle3_vlm_batch.sh.
NUM_PROMPTS="${NUM_PROMPTS:-80}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
TEMP="${TEMP:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

metrics_files=()
for DATASET_ONE in ${DATASETS}; do
  data_name="$(basename "${DATASET_ONE}")"
  data_name="${data_name%.jsonl}"
  data_name="${data_name%.json}"
  if [[ "${MIRACLE_MODE:-0}" =~ ^(1|true|yes|on)$ ]]; then
    data_name="${data_name}_miracle"
  fi

  out_dir="${OUT_ROOT}/${data_name}"
  mkdir -p "${out_dir}"
  metrics_file="${out_dir}/acceptance_metrics.json"
  metrics_files+=("${metrics_file}")

  echo "=== ${DATASET_ONE} -> ${metrics_file} ==="
  PYTHON_BIN="${PYTHON_BIN}" \
  DRAFT_MODEL="${DRAFT_MODEL}" \
  DATASET="${DATASET_ONE}" \
  NUM_PROMPTS="${NUM_PROMPTS}" \
  OUTPUT_LEN="${OUTPUT_LEN}" \
  NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  TEMP="${TEMP}" \
  OUTPUT_FILE="${out_dir}/results.json" \
  ACCEPTANCE_METRICS_FILE="${metrics_file}" \
    bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh

done

"${PYTHON_BIN}" - "${RUN_NAME}" "${metrics_files[@]}" <<'PYSUMMARY'
import json
import sys
from pathlib import Path

run_name = sys.argv[1]
paths = [Path(p) for p in sys.argv[2:]]
print()
print(f"| bench | {run_name} acceptance length | draft acceptance | drafts |")
print("|---|---:|---:|---:|")
for path in paths:
    m = json.loads(path.read_text())
    bench = Path(str(m.get("dataset", path.parent.name))).name
    acc_len = m.get("mean_acceptance_length")
    draft_rate = m.get("draft_acceptance_rate")
    drafts = m.get("num_drafts")
    acc_len_s = f"{acc_len:.4f}" if isinstance(acc_len, (int, float)) else "n/a"
    draft_rate_s = f"{100 * draft_rate:.2f}%" if isinstance(draft_rate, (int, float)) else "n/a"
    drafts_s = str(drafts) if drafts is not None else "n/a"
    print(f"| {bench} | {acc_len_s} | {draft_rate_s} | {drafts_s} |")
PYSUMMARY
