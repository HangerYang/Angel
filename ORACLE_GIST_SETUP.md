# Oracle Gist Conditioning for EAGLE-3 — Setup Guide

This branch adds **oracle gist conditioning** to EAGLE-3 draft models: at train time, the draft learns from a vector summarizing the target's answer (an oracle upper bound); at vLLM eval time, you supply reference outputs and the oracle is live-encoded.

## What's included

### Train side (in angelslim/)
✓ **Fully committed**, ready to use immediately.

- **Gist encoder** ([gist_embedding.py](angelslim/compressor/speculative/train/data/gist_embedding.py)): live encoding at collate time, no caching
- **Three injection modes**:
  - `qkv`: gist as a third layer-0 attention stream, re-read every draft step (strongest signal)
  - `fc`: gist fused with target aux streams into the FC (seed only, decays over rollout)
  - `both`: both simultaneously (oracle ceiling)
- **Whole-mode only**: one constant vector per example (only `gist_mode="whole"` supported)
- **Three configs** ready to train:
  - `smolvlm-256m-eagle3-3.1-oracle-whole-gist.json` (qkv mode)
  - `smolvlm-256m-eagle3-3.1-oracle-whole-gist-fc.json` (fc mode)
  - `smolvlm-256m-eagle3-3.1-oracle-whole-gist-both.json` (both)

### vLLM side (in third_party/vllm/)
⚠️ **Hand-edits present** — must be re-applied after sync.

The vLLM changes live in `third_party/vllm/` which is vendored and synced. When you `bash third_party/sync_vllm_latest.sh`, it hard-resets the tree, losing these edits. **You must re-apply them** before running vLLM eval.

**Files modified**:
- `vllm/model_executor/models/eagle_gist.py` (new file)
- `vllm/model_executor/models/llama_eagle3.py` (extended)
- `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` (extended)

**Re-apply manually or with a script** (see below).

## One-time setup

```bash
# 1. Sync vLLM to clean state (required after any git pull)
bash third_party/sync_vllm_latest.sh

# 2. Install vLLM locally (or use editable install if already done)
# bash third_party/install_local_vllm.sh            # CUDA 13.0
# VLLM_CUDA=12.6 bash third_party/install_local_vllm.sh  # CUDA 12.6

# 3. Apply gist patches (see "Applying vLLM patches" below)
```

## Training an oracle-gist draft

All three modes train identically (~6.3 hours per 2 epochs on 4 H200s):

```bash
# Example: qkv mode
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-3.1-oracle-whole-gist.json \
TRAIN_DATA_PATH=dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl \
EVAL_DATA_PATH=dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl \
OUTPUT_DIR=output/smolvlm_256m_gist_qkv_2ep \
TRAIN_MODE=nccl NUM_TRAIN_EPOCHS=2 SAVE_STRATEGY=epoch EVAL_STRATEGY=epoch \
bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

(Same command works for `-oracle-whole-gist-fc.json` and `-oracle-whole-gist-both.json`.)

## Eval: acceptance (train side only)

Measure acceptance without needing vLLM (works for all modes):

```bash
DRAFT_MODEL=output/.../checkpoint-XXXX \
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-3.1-oracle-whole-gist.json \
NUM_SAMPLES=512 \
bash scripts/speculative/smolvlm/eval_gist_acceptance.sh
```

The output is a JSON file with `mean_accepted_length` and per-position acceptance rates — same oracle you trained with.

## Eval: vLLM decode with oracle (requires vLLM patches)

Generate on a reference dataset with live oracle encoding:

```bash
# First time: apply vLLM patches (see below)

# Then run vLLM with a reference file from a prior eval
GIST_REF=output/baseline-draft/eval/MMStar/results.jsonl \
DRAFT_MODEL=output/.../smolvlm_256m_gist_qkv_2ep/checkpoint-XXXX \
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-3.1-oracle-whole-gist.json \
DATASET=Lin-Chen/MMStar \
bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
```

The results log will show `Oracle gist first arm: X/Y requests matched (LIVE)` — confirm the oracle is actually running, not zeroed out.

## Applying vLLM patches

Use the checked-in idempotent applicator after every vLLM sync:

```bash
bash third_party/sync_vllm_latest.sh
python third_party/apply_gist_patches.py
```

The script copies `third_party/patches/eagle_gist.py.txt` into the vendored vLLM tree and applies anchored edits to `llama_eagle3.py` and `speculator.py`. The manual anchors are documented in [third_party/patches/GIST_PATCH_NOTES.md](third_party/patches/GIST_PATCH_NOTES.md) for debugging when upstream vLLM changes.

## Key concepts

### Oracle upper bound

The gist summarizes the target's **answer before generation**. Acceptance measured this way is an oracle upper bound — never a production speedup. It shows how much the draft *could* accept if it knew the future.

### One vector per request

Only `gist_mode="whole"` is supported: one constant vector per example. This is the only mode that can be computed once before generation and reused at every draft step.

The other mode (`gist_mode="remaining"`) would need refreshing every 7 tokens against text that doesn't exist yet — not implementable at decode time.

### Row matching

The reference file (results.jsonl) is indexed by `id`. Request N must correspond to reference row N. **Same dataset, same prompt order, or the oracle conditions on wrong answers silently.**

Watch for `(LIVE)` in the vLLM eval log to confirm the oracle is armed; a zero gist looks identical in the metrics.

### Conditioning strength

- **`qkv`**: gist re-read every draft step → signal doesn't decay (strongest)
- **`fc`**: gist baked into the seed → decays after first step (weakest)
- **`both`**: gist at both points (oracle ceiling, not servable unless vLLM is fully patched)

## What's next

1. **Train** one or more drafts using the configs above.
2. **Measure acceptance** on your target dataset using `eval_gist_acceptance.sh`.
3. **Compare** qkv vs fc vs a baseline (non-gist) draft to see which injection point wins.
4. **(Optional) Run vLLM decode** if you want to see oracle acceptance with live encoding, not just the theoretical number.

## Troubleshooting

- **vLLM patches lost after sync**: `sync_vllm_latest.sh` hard-resets the tree. Re-apply the patches, or create an automated script (see Option B).
- **"Oracle gist first arm: 0/1 requests matched (NO MATCH)"**: The request ID didn't parse or was out of bounds in the reference table. Ensure the reference file has rows for the indices you're generating.
- **"Unknown vLLM environment variable" warnings**: These are benign — `VLLM_EAGLE_GIST_*` are not in vLLM's registry, but they're read in the patched code.

## Reference

- Training script: [scripts/speculative/smolvlm/train_eagle3_vlm_online.sh](scripts/speculative/smolvlm/train_eagle3_vlm_online.sh)
- Acceptance eval: [scripts/speculative/smolvlm/eval_gist_acceptance.sh](scripts/speculative/smolvlm/eval_gist_acceptance.sh)
- vLLM eval (requires patches): [scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh](scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh)
- Config examples: [angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-3.1-oracle-whole-gist*.json](angelslim/compressor/speculative/train/configs/)
