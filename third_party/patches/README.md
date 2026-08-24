# AngelSlim patches for local vLLM (`v0.25.0`)

These patches live in **AngelSlim git** (portable across servers). They are **not** upstream vLLM commits.

Stock vLLM rejects Eagle3 for SmolVLM/Idefics3 (`Model does not support EAGLE3 interface`). Apply after cloning `third_party/vllm`.

## Patches

| Patch | Purpose |
|---|---|
| `vllm-v0.25.0-smolvlm-eagle3.patch` | `SupportsEagle3` on Idefics3/SmolVLM + `load_eagle_model` `text_model` wiring |
| `vllm-v0.25.0-eagle3-banded-mix-fc.patch` | `banded_mix_fc`: learned band mix in front of the stock fused_fc EAGLE 3.1 path |
| `vllm-v0.25.0-odyssey-rejection-sampler.patch` | Env-gated Odyssey rejection sampler hook (`ODYSSEY_BRANCH`), default vLLM path unchanged |

## Apply / always get the latest (universal)

**After every `git pull` of AngelSlim** (any machine), refresh local vLLM from
the tracked patches with one command:

```bash
bash third_party/sync_vllm_latest.sh
source third_party/env.sh
```

That script resets `third_party/vllm` to clean `v0.25.0`, then re-applies
**every** file under `third_party/patches/`. No per-file `git checkout` list.

First-time / new CUDA machine (once):

```bash
# CUDA 13.0 (default) — or VLLM_CUDA=12.6 / 12.9
bash third_party/install_local_vllm.sh
source third_party/env.sh
```

Other helpers:

```bash
bash third_party/apply_vllm_patches.sh   # apply only (idempotent skip-if-present)
LINK=1 bash third_party/sync_vllm_latest.sh  # also re-link .so / .pth
```

Do **not** hand-edit `third_party/vllm` for portable changes — edit `.patch` files.

CUDA notes: patches are **CUDA-agnostic** (Python sources). Native `.so` must
still match the machine — choose `VLLM_CUDA=13.0` / `12.6` / `12.9` via
`third_party/install_local_vllm.sh` (see `third_party/README.md`).
