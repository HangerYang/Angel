# AngelSlim patches for local vLLM (`v0.25.0`)

These patches live in **AngelSlim git** (portable across servers). They are **not** upstream vLLM commits.

Stock vLLM rejects Eagle3 for SmolVLM/Idefics3 (`Model does not support EAGLE3 interface`). Apply after cloning `third_party/vllm`.

## Patches

| Patch | Purpose |
|---|---|
| `vllm-v0.25.0-smolvlm-eagle3.patch` | `SupportsEagle3` on Idefics3/SmolVLM + `load_eagle_model` `text_model` wiring |
| `vllm-v0.25.0-eagle3-progressive-staged.patch` | Progressive staged aux injection for multi-layer Eagle3 draft (`llama_eagle3` + propose) |

## Apply

```bash
# Preferred (also run by link_local_vllm.sh after .so overlay):
bash third_party/apply_vllm_patches.sh

# Manual:
cd third_party/vllm
git apply ../patches/vllm-v0.25.0-smolvlm-eagle3.patch
```

Idempotent: already-applied patches are skipped.
