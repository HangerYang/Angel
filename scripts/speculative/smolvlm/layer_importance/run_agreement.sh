#!/usr/bin/env bash
# Agreement: final-norm+lm_head probe vs final logits (KL) and gold CE.
#
#   DATA_PATH=/data/large.jsonl bash run_agreement.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"
run_metrics "agreement" "$@"
