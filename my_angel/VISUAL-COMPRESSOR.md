# Visual compression for the SmolVLM EAGLE-3 drafter

Status **2026-09-01**: two runs trained and evaluated, both a net loss. This file
is the record of what the method is and what is wrong with it.

---

## Background: what "a row" is

SmolVLM turns one image into tokens like this:

1. the image is split into **13 tiles** (sub-images)
2. each tile goes through the vision encoder + pixel shuffle → **64 tokens**,
   laid out as an **8×8 grid**, each token covering one patch of that tile
3. so one image ≈ **13 × 64 = 832 visual tokens** entering the LM

Measured on a real prompt: `hidden_states (1, 892, 5184)`, `tiles (13, 64)` —
832 of the 892 prompt tokens are image.

In the hidden-state tensor `[seq_len, 576]` each token occupies one row, which is
why the code and the tables below say "row" where they mean "visual token".

---

## The method

The EAGLE-3 drafter has **one** attention layer. The claim under test: that layer
cannot route usefully over 832 visual tokens, so compressing each tile's 64 tokens
down to `k` summaries should make routing tractable without losing what the drafter
needs.

`VisRowCompressor` (`angelslim/compressor/vistoken/row_compressor.py`, 56,457
params — 0.108% of the 52.0M draft) is a Q-Former-shaped module: `k` learned
queries cross-attend over a tile's 64 tokens and emit

```
w    = softmax(q · k_projᵀ / 8)     over the tile's 64 tokens, sums to 1
out  = Σᵢ wᵢ · tokenᵢ
```

**No value projection**, deliberately: the output is then a convex combination of
real target hidden states and stays inside the distribution `fc_norm` expects.

It runs on the **aux hidden states** (9 target layers concatenated, `[64, 9×576]`),
after the target forward and before the left shift, so one routing decision applies
to all 9 streams at once — the same image region reaches the drafter at every depth
the target read it at. The `k` summaries occupy fixed slots inside their own tile
(k=1 → slot 32), so each keeps a real target position and its RoPE angle.

---

## The two runs

Both are **`banded_mix_fc_3.1` plus the compressor** — same
`eagle_aux_injection_mode`, same 9 aux layers `[2,4,8,10,15,18,20,26,28]`, same 3
bands, same `fc_norm`/`norm_output`. The configs differ from the baseline only by
the added `vistoken_*` blocks. So both should dominate that baseline.

| run | what it adds | steps |
|---|---|---|
| `vistoken-k1` | each tile's 64 tokens → **1** (learned-query compression) | 66,466 (2 ep) |
| `vistoken-k1-attn-prune-x64-m16` | same, but first keep only the **top 16 of 64** by the target's own attention, then compress those | 66,466 (2 ep) |

---

## Results

vLLM V1 offline, `answer_then_describe`, N=80/benchmark, K=4, `max_num_seqs=1`,
`enforce_eager`, one job at a time on GPU 0. Full tables in `one-layer-results.md`.

| run | τ (temp 0) | tok/s | speedup |
|---|---:|---:|---:|
| `banded_mix_fc_3.1` *(baseline)* | **2.706** | **326.8** | 1.673x |
| `vistoken-k1` | 2.557 (−5.5%) | 302.5 (−7.4%) | 1.548x |
| `vistoken-k1-attn-prune-x64-m16` | 2.570 (−5.0%) | 301.9 (−7.6%) | 1.544x |
| `branch_change_top1_w01` *(current best)* | 2.815 (+4.0%) | 339.7 (+3.9%) | 1.738x |

The prune buys +0.5% over plain k1 — inside run-to-run noise.

**The loss is image-only, which is the clean control.** MATH-500 is the only
text-only benchmark, so the compressor is inert there:

| run | image-mean τ (7 benches) | MATH-500 τ (no image) |
|---|---:|---:|
| `banded_mix_fc_3.1` | 2.5882 | 3.5323 |
| `vistoken-k1` | 2.4252 (**−6.3%**) | 3.4767 (−1.6%) |
| `vistoken-k1-attn-prune-x64-m16` | 2.4365 (−5.9%) | 3.5083 (−0.7%) |

De-chaining the cumulative per-position rates
(`vllm_offline_eagle3_vlm_batch.py:1101-1104`) gives a **flat ~5% conditional
per-token loss** on image prompts, not a depth-growing pathology.

---

## Problem 1 — it is not compressing, it copies one token

`my_angel/vistoken_routing_probe.py`, real target, 46 image samples / 626 tiles:

| | ckpt-33233 (1 ep) | ckpt-66466 (2 ep) | reference |
|---|---:|---:|---|
| effective support `exp(H(w))` | **1.00** | **1.00** | 1 = hard pick, 64 = mean pool |
| top-1 routing weight | 0.9985 | **0.9987** | uniform = 0.0156 |
| raw logit range (max−min) | 64,373 | **173,616** | trained `temperature` = **8.0** |

The design wanted `w = [0.02, 0.01, 0.15, 0.14, …]` — a weighted blend. What it
trained is `w = [0, 0, …, 0.9987, …, 0]`. With no value projection that makes the
tile's "summary" a **verbatim copy of one of its 64 tokens**; the other 63 are
discarded. `vistoken-k1` is a learned **row sampler that throws away 98.4% of the
image**, not a compressor.

Losing only 6.3% acceptance for that is the surprising part.

---

## Problem 2 — the token it picks is the wrong one

`my_angel/vistoken_heatmap_probe.py` maps each pick back to its 8×8 slot.
Heatmaps: <https://claude.ai/code/artifact/52923aa0-37c4-40dc-8006-4443ae6c6bce>

| measurement | run 1 (of 64) | run 2 (of 16) | chance |
|---|---:|---:|---|
| top-1 routing weight | 0.9987 | 0.9992 | 0.0156 / 0.0625 |
| pick = target's most-attended slot | **0.6%** | 55.9% | 1.6% / 6.2% |
| pick = largest-magnitude slot | 2.6% | 28.4% | 1.6% / 6.2% |
| r(w, target attention) | +0.081 | +0.468 | 0 |
| r(w, slot magnitude) | +0.054 | +0.349 | 0 |

- **Run 1 agrees with the target less often than chance** (0.6% vs 1.6%). It has a
  positional habit instead: ~40% of picks land in the tile's **top row**,
  concentrated on three slots.
- **Run 2 collapsed onto a constant.** **84.8%** of all picks are slot 63 — the
  tile's last token, bottom-right corner. Adding the target-attention pre-filter
  made it worse, not better. Its 55.9% "agreement" is not semantic: both it and
  the target are pulled to the same corner.
- **The target barely discriminates either.** Its own attention over a tile runs
  1.3%–2.9% against 1.6% uniform, peaking at **1.86×** on slot 63 — an
  attention-sink artifact. A top-16 filter on that signal selects a border-biased
  set, not a salient one. This matches the pre-launch probe, which measured the
  top-16 rows holding 0.344 of a tile's score mass against 0.250 for flat.

---

## Problem 3 — why training produced this

The compressor is trained by the **drafter's next-token loss alone**. There is no
reconstruction or fidelity term anywhere. Three facts stack:

1. **The drafter barely uses visual tokens.** Training top-1 accuracy with all 832
   visual tokens is `0.6892`; with 13 it is `0.6895` — identical, and held-out
   agrees (`eval acc_0` 0.6414 vs 0.6407). So the gradient reaching the routing
   weights carries almost no information about which region matters.
2. **Nothing regularizes the routing.** `weight_decay = 0.0` (read from
   `training_args.bin`) and no entropy penalty, so `‖k_proj‖` grows without bound —
   its norm went 26.49 → 34.88 between epochs and the logit range grew **170%**,
   64,373 → 173,616. Sharpening always shaves a little loss; nothing pushes back.
3. **No value projection** means a one-hot `w` collapses the intended convex
   combination to a **corner of the simplex** — one original token.

So this is not a compressor that learned the wrong thing. It is a compressor whose
objective never constrained it, drifting into a fixed positional habit while the
loss curve looked healthy.

**It is the opposite of undertrained.** Nothing is saturated at 2 epochs — both
runs were still improving in the final decile, and compressor parameter norms grew
~30% between epochs — but the routing distribution had already collapsed and kept
collapsing. And the gap that matters is *widening*: `ploss_0` ratio vs the baseline
runs 1.281 → 1.565 monotonically across the 10 deciles of training, while top-1
accuracy stays exactly at parity. It preserves the argmax and destroys the
calibration.

---

## Problem 4 — the speed argument does not work either

Throughput went **down** 7.4%. The visual tokens only cost the drafter at
**prefill**, once per request, amortized over hundreds of decode steps; per decode
step the drafter processes one token regardless. Compression attacks a cost that is
not on the critical path and adds the compressor's own forward to every request.

**Row compression can only ever pay through better acceptance, never through lower
latency.** It is a representation-quality idea, not an efficiency idea.

---

## Problem 5 — one decode-path risk was flagged and never checked

`TODO-vistoken-k1.md` flagged it before launch. In the compact prefill,

```python
seq_len = int(self.input_buffers.positions[:num_tokens].max().item()) + 1
```
`third_party/vllm/.../autoregressive/speculator.py:651`

`seq_lens` stays the **full** prompt length (~900) while only `compact_n` (~100)
slots are written, so the draft's attention window spans image-position KV slots
this request never wrote. What is in them was never inspected.

Probably not the main cause — it would predict a depth-growing degradation and the
conditional per-position numbers show a flat one — but it is the last alternative
explanation, and closing it costs one instrumented eval run.

---

## What would have to change

| # | fix | why |
|---|---|---|
| 1 | **Constrain the routing geometry**: normalize the dot product (`q·k / (‖q‖‖k‖)` or a LayerNorm before `k_proj`), and/or weight-decay the compressor param group, and/or floor the entropy of `w`. | Without it every future vistoken run repeats the same collapse. The failure is in the objective's geometry, not in `k`. |
| 2 | **Give the compressor a signal that is actually about the image** — a reconstruction or target-fidelity term — since the draft loss demonstrably does not supply one (Problem 3, fact 1). | A one-hot pick that ignores the image is the *correct* optimum of the current objective. |
| 3 | **Then** revisit `num_queries: 4` / `16`. | k=1 gives the drafter 13 tokens for a whole image. With a one-hot router, raising `k` just samples `k` arbitrary tokens, so this is meaningless until 1 and 2 are done. |

Not worth GPU time as things stand: `query_mode: "mean"` and
`vistoken_prune: {mode: "random"}` both measure stages that are not the bottleneck.

**On "Plan 2"** (pretrain the compressor against the target for information
fidelity, *then* train end-to-end): the pretraining stage exists as
`tools/train_qsampler.py` on branch `mess`, with results in
`output/qsampler-n{1,4,8,16}` — target top-1 self-agreement 85.2% at 4x
compression, 79.4% at 64x. The end-to-end stage was never run and there is no
warm-start path for it. Fixes 1 and 2 are prerequisites either way: end-to-end
training under the current settings would erase a pretrained compressor within an
epoch.

---

## Reproducing

```bash
# train (4 GPUs, ~4h, 2 epochs, 66,466 steps)
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-vistoken-k1.json \
  SAVE_STRATEGY=epoch EVAL_STRATEGY=epoch \
  OUTPUT_DIR=my_angel/eagle/vistoken-k1 \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

# eval (identical harness to every other arm)
DATASETS_ATD="Lin-Chen/MMStar MMMU/MMMU opendatalab/OmniDocBench HuggingFaceH4/MATH-500 \
  lmms-lab/textvqa lmms-lab/chartqa ai4math/mathvista lmms-lab/COCO-Caption" \
  TEMP=0 bash scripts/speculative/smolvlm/run_atd_acceptance.sh
python my_angel/make_one_layer_results.py

# the two diagnostics
CUDA_VISIBLE_DEVICES=3 python my_angel/vistoken_routing_probe.py --n 48
CUDA_VISIBLE_DEVICES=3 python my_angel/vistoken_heatmap_probe.py --n 48
```

The eval log line confirming the compressor actually armed — present in every
`rerun_atd/temp{0,1}/_logs/<bench>.log` of both runs:

```
[speculator.py:536] vistoken: 1 rows/tile at slots (32,), 9 aux streams; compact draft prefill armed
```
