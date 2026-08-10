# AngelSlim patches for local vLLM (`v0.25.0`)

These patches live in **AngelSlim git** (portable across servers). They are **not** upstream vLLM commits.

Stock vLLM rejects Eagle3 for SmolVLM/Idefics3 (`Model does not support EAGLE3 interface`). Apply after cloning `third_party/vllm`.

## Patches

| Patch | Purpose |
|---|---|
| `vllm-v0.25.0-smolvlm-eagle3.patch` | `SupportsEagle3` on Idefics3/SmolVLM + `load_eagle_model` `text_model` wiring |
| `vllm-v0.25.0-eagle3-progressive-staged.patch` | Progressive staged + hawk + **miracle** (oracle GT-HS tape for fused/progressive/hawk) |

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

Miracle eval (see `scripts/speculative/smolvlm/README.md` § Miracle mode).
Smoke (2 prompts):

```bash
MIRACLE_MODE=1 TEMP=0 NUM_PROMPTS=2 OUTPUT_LEN=32 \
  DATASET=dataset/smolvlm_256m_target_gen/data_0-36.jsonl \
  DRAFT_MODEL=output/smolvlm_256m_hawk/checkpoint-30000 \
  DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-hawk.json \
  MIRACLE_HS_DIR=results/miracle_smoke_hs \
  bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
```

CUDA notes: patches are **CUDA-agnostic** (Python sources). Native `.so` must
still match the machine — choose `VLLM_CUDA=13.0` / `12.6` / `12.9` via
`third_party/install_local_vllm.sh` (see `third_party/README.md`).
