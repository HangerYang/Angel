#!/usr/bin/env bash
# Information proxies: effective rank + per-dim variance of HS.
#
#   DATA_PATH=/data/large.jsonl bash run_info.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"
run_metrics "info" "$@"
