# AngelSlim patches for local vLLM (`v0.25.0`)

These patches live in **AngelSlim git** (portable across servers). They are **not** upstream vLLM commits.

Stock vLLM rejects Eagle3 for SmolVLM/Idefics3 (`Model does not support EAGLE3 interface`). Apply after cloning `third_party/vllm`.

## Patches

**The patches are an ordered chain, not a set.** Each is generated against the
tree the previous one leaves behind, so the `NN-` prefix *is* the apply order and
`apply_vllm_patches.sh` applies them in sorted order. A patch applied out of
order fails with `patch does not apply` — that is the symptom, not a version
mismatch. Never rename one out of its slot; generate a new patch against the tree
the highest-numbered one produces.

| Patch | Purpose |
|---|---|
| `10-vllm-v0.25.0-eagle3-progressive-staged.patch` | `SupportsEagle3` on Idefics3/SmolVLM + `load_eagle_model` `text_model` wiring; progressive staged + hawk + **miracle** (oracle GT-HS tape for fused/progressive/hawk); adds `vllm/angelslim_latency.py` and `eagle_miracle.py`. Supersedes the old standalone `vllm-v0.25.0-smolvlm-eagle3.patch` |
| `20-vllm-v0.25.0-eagle3-vistoken.patch` | **vistoken** visual row compression: each image tile's 64 draft rows collapse to k summaries, so the drafter routes over ~15 image rows instead of ~900. Auto-arms on `vistoken_compress` in the draft config; `VLLM_EAGLE_VISTOKEN=0` disables |
| `30-vllm-v0.25.0-eagle3-band-mix-qk-norm.patch` | `banded_mix_fc` (learned band mix in front of the stock fused_fc EAGLE-3.1 path), `banded_mix_wide` (no fusion FC; layer 0 takes `(1+B)H`), and `qk_norm` draft attention (per-head RMSNorm on Q/K pre-RoPE, loads `self_attn.q_norm/k_norm.weight`). These three interleave in `llama_eagle3.py` and no longer split into independent patches |
| `40-vllm-v0.25.0-angelslim-latency.patch` | `angelslim_time_block` around the target forward (`model_runner`) and the draft prefill (`speculator`) |
| `50-vllm-v0.25.0-odyssey-rejection-sampler.patch` | Odyssey: env-gated pure-torch resync branches in the rejection sampler. Inert unless `ODYSSEY_BRANCH` is set, so the default path stays on the stock Triton kernels |

Not part of the vLLM chain:

- HiViS is no longer patched from here. The vendored checkout at `HiViS/`
  carries its SmolVLM + AngelSlim-EAGLE3 changes in-tree; see
  `HiViS/README_ANGELSLIM.md`.
- `eagle_gist.py.txt` + `GIST_PATCH_NOTES.md` — oracle gist conditioning, applied
  by hand; see the notes. Not wired into the chain.

### Verifying the chain

From a pristine checkout the chain must apply with no fuzz and no rejects:

```bash
bash third_party/sync_vllm_latest.sh     # reset to v0.25.0, reapply all
bash third_party/apply_vllm_patches.sh   # second run must say "Already applied"
```

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
