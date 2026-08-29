#!/bin/bash
# Thin wrapper: official HiViS training for SmolVLM-256M specifically.
# All the actual logic (generate/stage1/stage2, env vars, defaults) lives in
# the unified scripts/speculative/train_official_hivis.sh, shared across all
# three supported targets -- see that script's header and
# scripts/speculative/README_train_official_hivis.md for the full option
# list. This wrapper exists only so existing callers (run_official_hivis_
# smolvlm_2ep_1ep.sh, queue_hivis_smolvlm_artifacts.sh, ...) that `bash` this
# exact path keep working unchanged.
#
# Note for callers relying on the old default output paths: this script's
# own STAGE=generate no longer does two passes (generate_text/generate_mm
# are gone -- ge_data_smolvlm.py now does one pass, see common.py), and the
# default HIVIS_DATA_ROOT/OUTPUT_ROOT directory names changed from
# *_smolvlm_256m_* to *_smolvlm256m_* to match MODEL=smolvlm256m. Callers
# that set these explicitly (as the ones above do) are unaffected.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec env MODEL=smolvlm256m bash "$ROOT/scripts/speculative/train_official_hivis.sh"
