# AngelSlim patches for local vLLM (`v0.25.0`)

These patches live in **AngelSlim git** (portable across servers). They are **not** upstream vLLM commits.

Stock vLLM rejects Eagle3 for SmolVLM/Idefics3 (`Model does not support EAGLE3 interface`). Apply after cloning `third_party/vllm`.

## Patches

| Patch | Purpose |
|---|---|
| `vllm-v0.25.0-smolvlm-eagle3.patch` | `SupportsEagle3` on Idefics3/SmolVLM + `load_eagle_model` `text_model` wiring |
| `vllm-v0.25.0-eagle3-progressive-staged.patch` | Progressive staged + hawk aux injection for multi-layer Eagle3 draft (`llama_eagle3` + propose) |

## Apply

```bash
# Preferred one-shot (CUDA 13.0 default; use VLLM_CUDA=12.6 on older drivers):
bash third_party/install_local_vllm.sh

# Or only re-apply patches (also run by link_local_vllm.sh after .so overlay):
bash third_party/apply_vllm_patches.sh

# Manual:
cd third_party/vllm
git apply ../patches/vllm-v0.25.0-smolvlm-eagle3.patch
```

Idempotent: already-applied patches are skipped.

CUDA notes: patches are **CUDA-agnostic** (Python sources). Native `.so` must
still match the machine — choose `VLLM_CUDA=13.0` / `12.6` / `12.9` via
`third_party/install_local_vllm.sh` (see `third_party/README.md`).
