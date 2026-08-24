#!/usr/bin/env bash
# Apply AngelSlim-tracked patches onto third_party/vllm (v0.25.0).
# Safe / idempotent: skips patches that are already applied.
#
# Prefer after git pull (resets stale tree + reapplies everything):
#   bash third_party/sync_vllm_latest.sh
#
# Apply-only:
#   bash third_party/apply_vllm_patches.sh
#
# Called from link_local_vllm.sh after the wheel .so overlay so a fresh
# clone+link on any server gets SmolVLM Eagle3 support.
#
# ORDER MATTERS. The patches form a dependency chain (each is generated against
# the tree the previous one produces), so they carry NN- numeric prefixes and
# are applied in sorted order. Never rename one out of its slot, and generate a
# new patch against the tree the highest-numbered one leaves behind.
#
# Only top-level *.patch files here target vLLM. Patches for other nested repos
# live in subdirectories (e.g. patches/hivis/) and are not scanned.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SRC="${ROOT}/third_party/vllm"
PATCH_DIR="${ROOT}/third_party/patches"

if [[ ! -f "${VLLM_SRC}/pyproject.toml" ]]; then
  echo "ERROR: missing vLLM checkout at ${VLLM_SRC}" >&2
  echo "  git clone --branch v0.25.0 --depth 1 https://github.com/vllm-project/vllm.git third_party/vllm" >&2
  exit 1
fi

shopt -s nullglob
patches=()
while IFS= read -r p; do
  patches+=("${p}")
done < <(printf '%s\n' "${PATCH_DIR}"/*.patch | LC_ALL=C sort)

if [[ ${#patches[@]} -eq 0 ]]; then
  echo "No patches in ${PATCH_DIR}"
  exit 0
fi

cd "${VLLM_SRC}"

# Drop empty placeholders so they neither apply nor mask a real gap.
kept=()
for patch in "${patches[@]}"; do
  if [[ -s "${patch}" ]]; then
    kept+=("${patch}")
  else
    echo "Skipping empty: $(basename "${patch}")"
  fi
done
patches=("${kept[@]}")
if [[ ${#patches[@]} -eq 0 ]]; then
  echo "No non-empty patches in ${PATCH_DIR}"
  exit 0
fi

# Idempotency for a chain is decided by the LAST patch, not per patch: once the
# chain is applied, every earlier patch's context has been rewritten by a later
# one, so an individual --reverse --check on it fails even though it is applied.
last="${patches[${#patches[@]} - 1]}"
if git apply --reverse --check "${last}" >/dev/null 2>&1; then
  echo "Already applied: all ${#patches[@]} AngelSlim vLLM patches."
  exit 0
fi

# Not fully applied, so the chain is about to be replayed from the tag. Tracked
# edits (a half-applied chain, or hand-edits) would collide. Untracked files are
# fine: link_local_vllm.sh drops native .so overlays here before calling us.
if [[ -n "$(git diff --name-only)" ]]; then
  echo "ERROR: ${VLLM_SRC} has modified tracked files but the patch chain is not" >&2
  echo "       fully applied, so it cannot be replayed onto this tree:" >&2
  git diff --name-only | sed 's/^/  /' >&2
  echo "  Reset to clean v0.25.0 and reapply everything:" >&2
  echo "    bash third_party/sync_vllm_latest.sh" >&2
  exit 1
fi

for patch in "${patches[@]}"; do
  name="$(basename "${patch}")"
  if git apply --check "${patch}" >/dev/null 2>&1; then
    echo "Applying: ${name}"
    git apply "${patch}"
  else
    echo "ERROR: cannot apply ${name}" >&2
    echo >&2
    git apply --check "${patch}" 2>&1 | sed 's/^/  /' >&2
    echo >&2
    echo "  The patches are an ordered chain; each expects the tree the" >&2
    echo "  previous one leaves behind. Reset and reapply from clean v0.25.0:" >&2
    echo "    bash third_party/sync_vllm_latest.sh" >&2
    echo "  If that still fails, ${VLLM_SRC} is not clean v0.25.0:" >&2
    echo "    git -C ${VLLM_SRC} log --oneline -1   # expect v0.25.0" >&2
    echo "    git -C ${VLLM_SRC} status --short" >&2
    exit 1
  fi
done

echo "AngelSlim vLLM patches OK."
