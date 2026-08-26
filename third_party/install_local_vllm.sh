#!/usr/bin/env bash
# Install vLLM 0.25.0 for THIS machine's CUDA, then wire AngelSlim patches.
#
# Default = CUDA 13.0 (same as stock vLLM 0.25.0 PyPI wheel).
# CUDA 12.6 servers: no official +cu126 wheel for 0.25.0 → build from source
# against a cu126 PyTorch (needs CUDA toolkit / nvcc on that machine).
#
# Usage (from AngelSlim repo root, with the target env activated):
#   bash third_party/install_local_vllm.sh                  # CUDA 13.0
#   VLLM_CUDA=12.6 bash third_party/install_local_vllm.sh   # CUDA 12.6
#   VLLM_CUDA=12.9 bash third_party/install_local_vllm.sh   # prebuilt CUDA 12.x
#
# Optional:
#   PIP=uv|pip          package installer (default: uv if present, else pip)
#   VLLM_VERSION=0.25.0
#   SKIP_CLONE=1        do not clone if third_party/vllm missing (error instead)
#   SKIP_LINK=1         install wheel/source only; skip link_local_vllm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/third_party/vllm"
VLLM_VERSION="${VLLM_VERSION:-0.25.0}"
VLLM_CUDA="${VLLM_CUDA:-13.0}"

if command -v uv >/dev/null 2>&1 && [[ "${PIP:-}" != "pip" ]]; then
  PIP_CMD=(uv pip)
elif [[ "${PIP:-}" == "uv" ]]; then
  echo "ERROR: PIP=uv but uv not on PATH" >&2
  exit 1
else
  PIP_CMD=(python -m pip)
fi

_arch() {
  case "$(uname -m)" in
    x86_64|amd64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    *) echo "ERROR: unsupported arch $(uname -m)" >&2; exit 1 ;;
  esac
}

_ensure_clone() {
  if [[ -f "${SRC}/pyproject.toml" ]]; then
    return 0
  fi
  if [[ "${SKIP_CLONE:-0}" == "1" ]]; then
    echo "ERROR: missing ${SRC} (SKIP_CLONE=1)" >&2
    exit 1
  fi
  echo "Cloning vLLM v${VLLM_VERSION} -> ${SRC}"
  git clone --branch "v${VLLM_VERSION}" --depth 1 \
    https://github.com/vllm-project/vllm.git "${SRC}"
}

_pip_install() {
  echo "+ ${PIP_CMD[*]} install $*"
  "${PIP_CMD[@]}" install "$@"
}

_install_cu130() {
  # Stock PyPI wheel = CUDA 13.0 (filename has no +cu130 suffix).
  echo "[install] VLLM_CUDA=13.0  (default PyPI wheel + torch cu130)"
  if command -v uv >/dev/null 2>&1 && [[ "${PIP_CMD[0]}" == "uv" ]]; then
    _pip_install "vllm==${VLLM_VERSION}" --torch-backend=cu130
  else
    _pip_install torch torchvision \
      --index-url https://download.pytorch.org/whl/cu130
    _pip_install "vllm==${VLLM_VERSION}"
  fi
}

_install_cu129() {
  # Official prebuilt CUDA 12.x wheel for v0.25.0 (needs driver CUDA ≥ 12.9).
  local arch whl
  arch="$(_arch)"
  whl="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_${arch}.whl"
  echo "[install] VLLM_CUDA=12.9  (prebuilt +cu129)"
  echo "  wheel: ${whl}"
  if command -v uv >/dev/null 2>&1 && [[ "${PIP_CMD[0]}" == "uv" ]]; then
    _pip_install "${whl}" --torch-backend=cu129
  else
    _pip_install torch torchvision \
      --index-url https://download.pytorch.org/whl/cu129
    _pip_install "${whl}"
  fi
}

_install_cu126() {
  # No official v0.25.0+cu126 wheel. Build extensions against installed cu126 torch.
  echo "[install] VLLM_CUDA=12.6  (torch cu126 + build vLLM from ${SRC})"
  echo "  Requires: CUDA toolkit 12.6 (nvcc) on PATH / CUDA_HOME, matching GPU arch."
  if ! command -v nvcc >/dev/null 2>&1 && [[ -z "${CUDA_HOME:-}" ]]; then
    echo "WARNING: nvcc not found and CUDA_HOME unset — build will likely fail." >&2
    echo "  On the 12.6 server: install cuda-toolkit-12-6, then e.g." >&2
    echo "    export CUDA_HOME=/usr/local/cuda-12.6" >&2
    echo "    export PATH=\"\$CUDA_HOME/bin:\$PATH\"" >&2
  fi

  _pip_install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

  _ensure_clone
  (
    cd "${SRC}"
    # Allow reuse of the torch we just installed (strips pinned torch from reqs).
    python use_existing_torch.py
    _pip_install -r requirements/build/cuda.txt
    # Compile into the checkout; then link_local overlays/patches as usual.
    _pip_install --no-build-isolation --editable .
  )
}

echo "python: $(command -v python)"
echo "repo:   ${ROOT}"
echo "cuda:   ${VLLM_CUDA}"
echo "vllm:   ${VLLM_VERSION}"

case "${VLLM_CUDA}" in
  13.0|130|cu130)
    _ensure_clone
    _install_cu130
    ;;
  12.9|129|cu129)
    _ensure_clone
    _install_cu129
    ;;
  12.6|126|cu126)
    _install_cu126
    ;;
  *)
    echo "ERROR: unknown VLLM_CUDA=${VLLM_CUDA}" >&2
    echo "  Supported: 13.0 (default), 12.6 (source build), 12.9 (prebuilt +cu129)" >&2
    exit 1
    ;;
esac

if [[ "${SKIP_LINK:-0}" != "1" ]]; then
  # Source builds already compiled .so into third_party/vllm — do not rsync a
  # leftover CUDA 13 wheel over them.
  case "${VLLM_CUDA}" in
    12.6|126|cu126) export SKIP_OVERLAY=1 ;;
  esac
  bash "${ROOT}/third_party/link_local_vllm.sh"
  # shellcheck disable=SC1091
  source "${ROOT}/third_party/env.sh"
fi

python - <<'PY'
import os
from importlib.metadata import version

import torch
import vllm
from vllm.model_executor.models.idefics3 import Idefics3ForConditionalGeneration as C
from vllm.model_executor.models.interfaces import SupportsEagle3 as S
from vllm.platforms import current_platform

path = os.path.realpath(vllm.__file__)
print(f"OK: vllm {vllm.__version__}")
print(f"    {path}")
print(f"    torch {torch.__version__}  cuda {torch.version.cuda}")
if "third_party/vllm" not in path.replace("\\", "/"):
    raise SystemExit(f"import did not resolve to third_party checkout: {path}")
if S not in C.__mro__:
    raise SystemExit("SmolVLM Eagle3 patch missing — run bash third_party/apply_vllm_patches.sh")
meta = version("vllm")
dev = current_platform.device_type
print(f"    importlib.metadata version: {meta}")
print(f"    platform device_type: {dev!r}")
if not dev:
    raise SystemExit(
        "empty device_type (CUDA detection failed). "
        "Usually means vllm-*.dist-info is missing."
    )
print("SmolVLM Eagle3: OK")
PY

echo
echo "Done. Next shells:  source ${ROOT}/third_party/env.sh"
