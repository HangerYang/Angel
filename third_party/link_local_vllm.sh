#!/usr/bin/env bash
# Wire the *current* Python env to this repo's third_party/vllm checkout.
#
# Portable: no hardcoded machine paths. Derives repo root from this script's
# location, and site-packages from whatever `python` is on PATH (conda / uv / venv).
#
# On every new server / env:
#   1) Activate that env
#   2) uv pip install vllm==0.25.0    # builds/fetches native .so for THIS machine
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
if [[ -d "${INSTALLED}" && ! -L "${INSTALLED}" ]]; then
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
  echo "  Install the wheel in THIS env first, then re-run:" >&2
  echo "    uv pip install vllm==0.25.0   # or: pip install vllm==0.25.0" >&2
  echo "    bash third_party/link_local_vllm.sh" >&2
  exit 1
else
  echo "No site-packages/vllm dir; keeping existing .so already in ${PKG}"
fi

# Remove env-owned package tree so imports cannot silently use a stale install.
if [[ -e "${INSTALLED}" || -L "${INSTALLED}" ]]; then
  echo "Removing ${INSTALLED}"
  rm -rf "${INSTALLED}"
fi

# .pth path is absolute for *this* machine's checkout; regenerate on each server.
echo "${SRC}" > "${PTH}"
echo "Wrote ${PTH}"

# Also print the portable PYTHONPATH form (works even without .pth).
echo
echo "Optional (portable across shells):"
echo "  source ${ROOT}/third_party/env.sh"

python - <<'PY'
import os, vllm
path = os.path.realpath(vllm.__file__)
print(f"OK: vllm {vllm.__version__}")
print(f"    {path}")
if "third_party/vllm/vllm" not in path.replace("\\", "/"):
    raise SystemExit(f"import did not resolve to third_party checkout: {path}")
if "site-packages/vllm" in path.replace("\\", "/"):
    raise SystemExit(f"still importing from site-packages: {path}")
PY
