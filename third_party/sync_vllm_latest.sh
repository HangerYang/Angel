#!/usr/bin/env bash
# Universal: refresh local third_party/vllm to the latest *tracked* AngelSlim patches.
#
# Use this after every `git pull` of AngelSlim (any machine / any env):
#
#   bash third_party/sync_vllm_latest.sh
#   source third_party/env.sh
#
# What it does:
#   1. Ensures third_party/vllm exists at tag v0.25.0 (clones if missing)
#   2. Hard-resets tracked sources to clean v0.25.0 (drops stale hand-edits /
#      old patch application). Keeps untracked native .so overlays.
#   3. Re-applies every patch under third_party/patches/
#   4. Optional: LINK=1 also re-runs link_local_vllm.sh (.so overlay + .pth)
#
# First-time / new CUDA machine still needs install once:
#   bash third_party/install_local_vllm.sh            # CUDA 13.0
#   VLLM_CUDA=12.6 bash third_party/install_local_vllm.sh
#
# Do NOT hand-edit third_party/vllm for portable changes — edit .patch files.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SRC="${ROOT}/third_party/vllm"
VLLM_TAG="${VLLM_TAG:-v0.25.0}"
VLLM_REMOTE="${VLLM_REMOTE:-https://github.com/vllm-project/vllm.git}"

if [[ ! -f "${VLLM_SRC}/pyproject.toml" ]]; then
  echo "Cloning vLLM ${VLLM_TAG} → ${VLLM_SRC}"
  git clone --branch "${VLLM_TAG}" --depth 1 "${VLLM_REMOTE}" "${VLLM_SRC}"
fi

if [[ ! -d "${VLLM_SRC}/.git" ]]; then
  echo "ERROR: ${VLLM_SRC} is not a git checkout; cannot reset cleanly." >&2
  exit 1
fi

echo "=== sync_vllm_latest: reset ${VLLM_SRC} → ${VLLM_TAG} ==="
cd "${VLLM_SRC}"
# Make sure the tag exists locally (shallow clones may need fetch).
if ! git rev-parse -q --verify "refs/tags/${VLLM_TAG}" >/dev/null 2>&1 \
  && ! git rev-parse -q --verify "${VLLM_TAG}" >/dev/null 2>&1; then
  git fetch --depth 1 origin tag "${VLLM_TAG}" || git fetch --depth 1 origin "${VLLM_TAG}"
fi
git checkout -f "${VLLM_TAG}" >/dev/null
git reset --hard "${VLLM_TAG}"
# Drop previously patched *tracked* paths to clean tag; keep untracked .so.
# Patch-added files are untracked, so the reset above leaves them behind and the
# next `git apply` would fail with "already exists". Delete exactly the paths the
# patches create, read out of the patches themselves so this never goes stale.
while IFS= read -r added; do
  [[ -n "${added}" ]] && rm -f "${added}"
done < <(
  cat "${ROOT}"/third_party/patches/*.patch 2>/dev/null |
    awk '/^diff --git /{path=$4; sub(/^b\//, "", path); next}
         /^new file mode /{if (path != "") print path; path=""}'
)

echo "=== sync_vllm_latest: apply AngelSlim patches ==="
cd "${ROOT}"
bash "${ROOT}/third_party/apply_vllm_patches.sh"

if [[ "${LINK:-0}" == "1" ]]; then
  echo "=== sync_vllm_latest: LINK=1 → link_local_vllm.sh ==="
  bash "${ROOT}/third_party/link_local_vllm.sh"
fi

echo "=== sync_vllm_latest: OK ==="
echo "  Next: source third_party/env.sh"
echo "  Verify: python -c \"import vllm, os; print(vllm.__version__, os.path.realpath(vllm.__file__))\""
