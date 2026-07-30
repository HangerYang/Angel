#!/usr/bin/env bash
# Backward-compatible alias → run_all.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_all.sh" "$@"
