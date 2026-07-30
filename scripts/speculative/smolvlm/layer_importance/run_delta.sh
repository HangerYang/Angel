#!/usr/bin/env bash
# Embedding change: ‖h_ℓ − h_{ℓ−1}‖ / ‖h_{ℓ−1}‖ on loss positions.
#
#   DATA_PATH=/data/large.jsonl bash run_delta.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"
run_metrics "delta" "$@"
