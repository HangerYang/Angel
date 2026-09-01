# TODO — target-attention row pruning (vistoken-k1 + x64/m16)

Status: **trained and evaluated — no effect.** Written 2026-08-31; outcome
recorded 2026-09-01 in **`my_angel/VISUAL-COMPRESSOR.md`**. The prune beats plain
`vistoken-k1` by +0.5% tau at temp 0, inside run-to-run noise, and stays ~5% below
the no-compressor baseline. The `mode="random"` ablation was never run.

Config: `angelslim/compressor/speculative/train/configs/`
`smolvlm-256m-eagle3-vistoken-k1-attn-prune-x64-m16.json` — vistoken k=1 plus
`"vistoken_prune": {"group_size": 64, "keep_m": 16, "mode": "target_attn"}`.
Baseline to beat: the existing `vistoken-k1` run (same config minus the prune
block).

## Fixed before launch

`image_row_scores` read the captured projections as `q.view(1, seq_len, -1,
head_dim)[0]`, but `TargetQKCapture`'s hooks fire once for the whole batch, so
those tensors are `[B, S, H*d]`. At B>1 the element count still matched, so the
view did not raise — it reshaped to `[S, B*H, d]` and folded adjacent positions
into the head axis, and `prune_sample_image_rows` never received `b` at all.
Not a live miscompute (`PER_DEVICE_TRAIN_BATCH_SIZE` defaults to 1 and every
test built a batch of 1, which is why nothing caught it). Fixed in `0f6f9564`
with a B>1 regression test.

## Measured, and deliberately NOT changed

`my_angel/attn_prune_weighting_probe.py`, real SmolVLM-256M, 46 image samples /
626 tiles, one target forward each:

- **Weighting each query position by its image-attention mass: no effect, do
  not retry.** The softmax in `image_row_scores` is taken over image rows only,
  so every query position contributes a distribution summing to 1 — a position
  mid-way through text-only reasoning gets as much say as one actively reading
  the image. Weighting by the true image share
  (`exp(lse_image - lse_all)`, computable from the captured `k` with no extra
  forward) was implemented and measured: **84.0% of tiles keep an identical row
  set, mean Jaccard 0.981**. The reason is that the weights are already flat —
  the top 10% of query positions hold **0.097** of the mass against 0.100 for
  perfectly uniform. Query positions barely differ in how much they look at the
  image, so there is nothing for the weighting to exploit. Reverted.

## What the probe says about the signal itself

- **The attention signal is real but modest.** The top-16 rows of a tile hold
  **0.344** of that tile's score mass, against **0.250** for a flat
  distribution — 1.38x concentration. So the target does prefer some rows, but
  it is not ignoring the other 48.
- `Jaccard(target_attn, random)` is 0.144 against a chance value of 0.143, i.e.
  the attention-chosen set is simply a different subset, as expected. This is a
  sanity check, not evidence either way.

Read 0.344-vs-0.250 as the honest prior: there is something to select on, but
it is not a large margin, and none of this measures acceptance length. The one
run is what decides it.

## The run

```bash
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-vistoken-k1-attn-prune-x64-m16.json \
  SAVE_STRATEGY=steps SAVE_STEPS=5000 EVAL_STRATEGY=no \
  OUTPUT_DIR=my_angel/eagle/vistoken-k1-attn-prune-x64-m16 \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

`mode="random"` (same grouping and `keep_m`, random draw) is the ablation that
separates "pruning at all helps" from "this signal helps" — it needs a second
run, so it is not affordable now.
