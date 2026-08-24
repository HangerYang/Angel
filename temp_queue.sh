#!/usr/bin/env bash
# Train both branch-change follow-ups, then run the atd eval grid on them.
set -euo pipefail
cd /home/hyang/AngelSlim
bash temp_run.sh
bash temp_eval_grid.sh
echo "=== queue complete: $(date '+%F %T') ==="
