#!/usr/bin/env bash
# Train both branch-change follow-ups, then run the atd eval grid on them.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
bash "${ROOT}/scripts/speculative/smolvlm/run_branch_change_r33k_train.sh"
bash "${ROOT}/scripts/speculative/smolvlm/run_branch_change_r33k_eval.sh"
echo "=== queue complete: $(date '+%F %T') ==="
