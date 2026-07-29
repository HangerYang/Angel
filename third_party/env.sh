#!/usr/bin/env bash
# Portable env hook: put local third_party/vllm first on PYTHONPATH.
# Safe to source from any machine / any checkout path / any venv (conda, uv, …).
#
#   source third_party/env.sh
#
# Does NOT install packages. Pair with:
#   uv pip install vllm==0.25.0   # or pip, on THAT machine
#   bash third_party/link_local_vllm.sh
_ANGELSLIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_VLLM_SRC="${_ANGELSLIM_ROOT}/third_party/vllm"

if [[ ! -f "${_VLLM_SRC}/pyproject.toml" ]]; then
  echo "ERROR: missing full vLLM checkout at ${_VLLM_SRC}" >&2
  echo "  git clone --branch v0.25.0 --depth 1 https://github.com/vllm-project/vllm.git third_party/vllm" >&2
  return 1 2>/dev/null || exit 1
fi

# Prepend so local tree wins over any site-packages install.
case ":${PYTHONPATH:-}:" in
  *":${_VLLM_SRC}:"*) ;;
  *) export PYTHONPATH="${_VLLM_SRC}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

echo "[angelslim] PYTHONPATH includes ${_VLLM_SRC}"
unset _ANGELSLIM_ROOT _VLLM_SRC
