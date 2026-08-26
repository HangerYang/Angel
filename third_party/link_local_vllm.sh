#!/usr/bin/env bash
# Wire the *current* Python env to this repo's third_party/vllm checkout.
#
# Portable: no hardcoded machine paths. Derives repo root from this script's
# location, and site-packages from whatever `python` is on PATH (conda / uv / venv).
#
# Preferred one-shot (picks CUDA wheels / source build):
#   bash third_party/install_local_vllm.sh                  # CUDA 13.0 (default)
#   VLLM_CUDA=12.6 bash third_party/install_local_vllm.sh   # CUDA 12.6
#
# Manual:
#   1) Activate that env
#   2) Install a CUDA-matching vllm==0.25.0 wheel (or source build) on THIS machine
#   3) bash third_party/link_local_vllm.sh
#   4) source third_party/env.sh      # optional but recommended; also done via .pth
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/third_party/vllm"
PKG="${SRC}/vllm"

if [[ ! -f "${SRC}/pyproject.toml" ]]; then
  echo "ERROR: full vLLM source not found at ${SRC}" >&2
  echo "  Expected a clone of https://github.com/vllm-project/vllm @ v0.25.0" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not on PATH. Activate your env first." >&2
  exit 1
fi

# Resolve site-packages for the active interpreter (uv/conda/venv/system).
SITE="$(python - <<'PY'
import site, sys
paths = site.getsitepackages() if hasattr(site, "getsitepackages") else []
if not paths:
    # venv fallback
    import sysconfig
    paths = [sysconfig.get_path("purelib")]
print(paths[0])
PY
)"
INSTALLED="${SITE}/vllm"
PTH="${SITE}/angelslim-local-vllm.pth"

echo "python:    $(command -v python)"
echo "repo vllm: ${SRC}"
echo "site-pkg:  ${SITE}"

# Overlay native extensions from a real wheel install into the source tree.
# Required on each machine: .so files are ABI/CUDA-specific and not portable.
# SKIP_OVERLAY=1: keep .so already built in-tree (CUDA 12.6 source install).
if [[ "${SKIP_OVERLAY:-0}" == "1" ]]; then
  echo "SKIP_OVERLAY=1 — keeping in-tree .so under ${PKG}/"
  if [[ ! -e "${PKG}/_C_stable_libtorch.abi3.so" && ! -e "${PKG}/_C.abi3.so" ]]; then
    echo "ERROR: SKIP_OVERLAY=1 but no compiled .so in ${PKG}" >&2
    exit 1
  fi
elif [[ -d "${INSTALLED}" && ! -L "${INSTALLED}" ]]; then
  VER="$(python -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || true)"
  if [[ -n "${VER}" && "${VER}" != "0.25.0" ]]; then
    echo "WARNING: env has vllm==${VER}, expected 0.25.0" >&2
  fi
  echo "Overlaying wheel package -> ${PKG}/  (native .so for this machine)"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${INSTALLED}/" "${PKG}/"
  else
    cp -a "${INSTALLED}/." "${PKG}/"
  fi
elif [[ ! -e "${PKG}/_C_stable_libtorch.abi3.so" && ! -e "${PKG}/_C.abi3.so" ]]; then
  echo "ERROR: no site-packages/vllm to overlay, and source tree has no compiled .so." >&2
  echo "  Install a CUDA-matching vLLM in THIS env first, then re-run:" >&2
  echo "    bash third_party/install_local_vllm.sh                 # CUDA 13.0" >&2
  echo "    VLLM_CUDA=12.6 bash third_party/install_local_vllm.sh  # CUDA 12.6" >&2
  exit 1
else
  echo "No site-packages/vllm dir; keeping existing .so already in ${PKG}"
fi

# Remove env-owned package tree / editable stubs so imports use third_party only.
# Keep vllm-*.dist-info: vLLM's CUDA plugin calls importlib.metadata.version("vllm").
# Deleting it raises PackageNotFoundError, platform detection fails, and LLM()
# crashes with RuntimeError: Device string must not be empty.
if [[ -e "${INSTALLED}" || -L "${INSTALLED}" ]]; then
  echo "Removing ${INSTALLED}"
  rm -rf "${INSTALLED}"
fi
shopt -s nullglob
for stale in \
  "${SITE}"/__editable__.vllm*.pth \
  "${SITE}"/__editable___*vllm* \
  "${SITE}"/*vllm*.egg-link
do
  echo "Removing ${stale}"
  rm -rf "${stale}"
done
dists=("${SITE}"/vllm-*.dist-info)
shopt -u nullglob
if [[ ${#dists[@]} -eq 0 ]]; then
  DIST="${SITE}/vllm-0.25.0.dist-info"
  mkdir -p "${DIST}"
  cat > "${DIST}/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: vllm
Version: 0.25.0
EOF
  echo "Wrote stub ${DIST}  (importlib.metadata.version needs this for CUDA detection)"
else
  echo "Keeping ${dists[0]}  (needed for CUDA platform detection)"
fi

# AngelSlim-tracked patches (portable across servers). Must run AFTER rsync
# overlay, which restores stock vLLM sources from the wheel.
echo "Applying AngelSlim vLLM patches..."
bash "${ROOT}/third_party/apply_vllm_patches.sh"

# .pth path is absolute for *this* machine's checkout; regenerate on each server.
echo "${SRC}" > "${PTH}"
echo "Wrote ${PTH}"

# Also print the portable PYTHONPATH form (works even without .pth).
echo
echo "Optional (portable across shells):"
echo "  source ${ROOT}/third_party/env.sh"

python - <<'PY'
import os
from importlib.metadata import PackageNotFoundError, version

import vllm
from vllm.platforms import current_platform

path = os.path.realpath(vllm.__file__)
print(f"OK: vllm {vllm.__version__}")
print(f"    {path}")
if "third_party/vllm/vllm" not in path.replace("\\", "/"):
    raise SystemExit(f"import did not resolve to third_party checkout: {path}")
if "site-packages/vllm" in path.replace("\\", "/"):
    raise SystemExit(f"still importing from site-packages: {path}")
try:
    meta = version("vllm")
except PackageNotFoundError as e:
    raise SystemExit(
        "importlib.metadata cannot see vllm — keep site-packages/vllm-*.dist-info"
    ) from e
print(f"    importlib.metadata version: {meta}")
dev = current_platform.device_type
print(f"    platform device_type: {dev!r}")
if not dev:
    raise SystemExit(
        "empty device_type (CUDA detection failed). "
        "Usually means vllm-*.dist-info is missing."
    )
PY
