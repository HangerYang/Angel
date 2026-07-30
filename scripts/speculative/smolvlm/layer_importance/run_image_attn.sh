#!/usr/bin/env bash
# Image attention: attn mass from text/loss queries → image tokens.
# Text-only samples are skipped automatically (no crash).
#
#   DATA_PATH=/data/vl_mixed.jsonl bash run_image_attn.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"
run_metrics "image_attn" "$@"
