#!/usr/bin/env bash
# Run ALL layer-importance metrics (ce_grad + agreement + delta + info + image_attn).
#
# Example (large data elsewhere):
#   DATA_PATH=/data/mixed.jsonl OUTPUT_DIR=/data/layer_imp_out \
#     bash run_all.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"
run_metrics "all" "$@"
