#!/usr/bin/env bash
# CE activation gradients: masked ‖∂L/∂h_ℓ‖ under target next-token CE.
#
#   DATA_PATH=/data/large.jsonl bash run_ce_grad.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"
run_metrics "ce_grad" "$@"
