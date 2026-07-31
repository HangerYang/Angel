#!/usr/bin/env bash
# Apply AngelSlim-tracked patches onto third_party/vllm (v0.25.0).
# Safe / idempotent: skips patches that are already applied.
#
#   bash third_party/apply_vllm_patches.sh
#
# Called from link_local_vllm.sh after the wheel .so overlay so a fresh
# clone+link on any server gets SmolVLM Eagle3 support.
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
patches=("${PATCH_DIR}"/*.patch)
if [[ ${#patches[@]} -eq 0 ]]; then
  echo "No patches in ${PATCH_DIR}"
  exit 0
fi

cd "${VLLM_SRC}"
for patch in "${patches[@]}"; do
  name="$(basename "${patch}")"
  if git apply --reverse --check "${patch}" >/dev/null 2>&1; then
    echo "Already applied: ${name}"
    continue
  fi
  if git apply --check "${patch}" >/dev/null 2>&1; then
    echo "Applying: ${name}"
    git apply "${patch}"
  else
    echo "ERROR: cannot apply ${name} (wrong vLLM version or conflicting edits)" >&2
    echo "  Expected clean vLLM v0.25.0 under ${VLLM_SRC}" >&2
    exit 1
  fi
done

echo "AngelSlim vLLM patches OK."
