#!/bin/bash
# Thin wrapper: official HiViS training for Qwen2.5-VL-3B specifically.
# All the actual logic (generate/stage1/stage2, env vars, defaults) lives in
# the unified scripts/speculative/train_official_hivis.sh, shared across all
# three supported targets -- see that script's header and
# scripts/speculative/README_train_official_hivis.md for the full option
# list. This wrapper exists only so existing callers (run_official_hivis_
# qwen3b_2ep_1ep.sh, ...) that `bash` this exact path keep working unchanged.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec env MODEL=qwen25vl_3b bash "$ROOT/scripts/speculative/train_official_hivis.sh"
