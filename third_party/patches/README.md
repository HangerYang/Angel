# AngelSlim patches for local vLLM (`v0.25.0`)

These patches live in **AngelSlim git** (portable across servers). They are **not** upstream vLLM commits.

Stock vLLM rejects Eagle3 for SmolVLM/Idefics3 (`Model does not support EAGLE3 interface`). Apply after cloning `third_party/vllm`.

## Patches

| Patch | Purpose |
|---|---|
| `vllm-v0.25.0-smolvlm-eagle3.patch` | `SupportsEagle3` on Idefics3/SmolVLM + `load_eagle_model` `text_model` wiring |
| `vllm-v0.25.0-eagle3-progressive-staged.patch` | Progressive staged + hawk + **miracle** (oracle GT-HS tape for fused/progressive/hawk) |

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

### Re-apply progressive / miracle after pull

If an older progressive patch is already on the tree, reset then re-apply:

```bash
cd third_party/vllm
git checkout -- vllm/envs.py \
  vllm/model_executor/models/llama_eagle3.py \
  vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
rm -f vllm/model_executor/models/eagle_miracle.py
cd ../..
bash third_party/apply_vllm_patches.sh
source third_party/env.sh
```

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

Patch must include dense capture (`input_batch.positions`) and warmup-safe
miracle bind (`{index}-{hex}` req ids). Re-apply after every pull.
CUDA notes: patches are **CUDA-agnostic** (Python sources). Native `.so` must
still match the machine — choose `VLLM_CUDA=13.0` / `12.6` / `12.9` via
`third_party/install_local_vllm.sh` (see `third_party/README.md`).
