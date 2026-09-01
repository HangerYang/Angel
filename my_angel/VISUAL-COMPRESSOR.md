# Visual compression for the SmolVLM EAGLE-3 drafter

Status as of **2026-09-01**. Two runs trained, both evaluated, both a net loss.
This file is the record of what was built, what was measured, and what the
numbers actually say. It supersedes nothing — `TODO-vistoken-k1.md` (the design)
and `TODO-attn-prune.md` (the pre-launch probe) are still the source of truth for
*why* each piece is shaped the way it is.

---

## 1. The idea

SmolVLM-256M renders one image as **13–17 tiles × 64 rows ≈ 900 visual tokens**,
against ~70 text tokens. The EAGLE-3 drafter has **one** attention layer. The
claim under test: that single layer cannot route usefully over 900 image rows, so
compressing each tile's 64 rows to `k` summaries should make the routing tractable
without losing what the drafter needs.

Compression happens in **target-aux-hidden-state space**, after the target forward
and before the drafter's `fc_norm → band mix → fc`, so one routing decision is
applied to all 9 aux streams at once — the same image region reaches the drafter
at every depth the target read it at.

---

## 2. Key steps, in order

| # | Step | Where |
|---|---|---|
| 1 | **`VisRowCompressor`** — cross-attention over a tile's rows, learned queries, **no value projection** (every output row is a convex combination of real target hidden states, so it stays in `fc_norm`'s input distribution). 56,457 params at k=1 — **0.108%** of the 52.0M draft. | `angelslim/compressor/vistoken/row_compressor.py` |
| 2 | **Splice into the training sequence** immediately after the target forward and **before** the EAGLE left shift, while every row-aligned tensor is still 1:1 on absolute positions. One gather rebuilds ids / mask / positions / hidden states together. | `angelslim/compressor/vistoken/splice.py`, called from `online_eagle3_trainer.py` |
| 3 | **Fixed slot convention** — the `k` summaries occupy data-independent rows *inside their own tile* (k=1 → row 32). Fixed because vLLM builds the draft's slot mapping from `input_ids` alone, before the model runs, so the kept rows cannot depend on routing weights. A summary therefore keeps a real target position and the RoPE angle computed at it. | `VisRowCompressor.slot_offsets` |
| 4 | **vLLM decode support** — `third_party/patches/20-vllm-v0.25.0-eagle3-vistoken.patch`. Auto-arms whenever the draft config carries `vistoken_compress`; reuses HiViS's compact-prefill machinery so the KV cache is never shrunk, only partially written. Loads `vistoken.*` by importing AngelSlim, so training and decode run identical module code. | `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` |
| 5 | **Parity tests** — train and vLLM paths agree row-for-row; text rows byte-identical; `query_mode: mean` reproduces the uniform tile average; outputs inside the tile's convex hull; gradients reach every parameter. | `tests/test_vistoken_{splice,ttt,vllm_parity}.py` |
| 6 | **Run 1: `vistoken-k1`** — trained from scratch, 2 epochs, compressor + drafter jointly. | `my_angel/eagle/vistoken-k1` |
| 7 | **Target-attention row pruning** — a content-aware pre-filter in front of the compressor: rank each tile's 64 rows by the *target's own* q·k (mean over heads, over the 9 aux layers, over every loss-masked query position) and keep the top M=16. Captured by hooking `q_proj`/`k_proj` during the forward that already runs — no `output_attentions` (blocked under flash-attn-2), no second target forward. | `angelslim/compressor/vistoken/target_attn_prune.py` |
| 8 | **Pre-launch probe** — measured that weighting query positions by image-attention mass is a **no-op** (84.0% of tiles keep an identical row set, mean Jaccard 0.981), because the weights are already flat. Implemented, measured, reverted. Same probe put the honest prior on the prune signal: a tile's top-16 rows hold **0.344** of its score mass vs **0.250** flat — real, but modest. | `my_angel/attn_prune_weighting_probe.py`, commit `31cada4d` |
| 9 | **Run 2: `vistoken-k1-attn-prune-x64-m16`** — same as run 1 plus `vistoken_prune: {group_size: 64, keep_m: 16, mode: target_attn}`. | `my_angel/eagle/vistoken-k1-attn-prune-x64-m16` |
| 10 | **Full ATD eval of both**, 8 benchmarks × temp {0,1}, on the same harness as every other arm. | `my_angel/one-layer-results.md` |

---

## 3. What is in each run

**Both vistoken runs are `banded_mix_fc_3.1` plus the compressor.** Verified by
reading the configs: `smolvlm-256m-eagle3-vistoken-k1.json` and
`...-attn-prune-x64-m16.json` carry `eagle_aux_injection_mode: "banded_mix_fc"`,
the same 9 `aux_hidden_states_layer_ids` `[2,4,8,10,15,18,20,26,28]`, the same
3 `eagle_aux_layer_bands`, and `fc_norm: true`, `norm_output: true` — byte-identical
to the `banded_mix_fc_3.1` config apart from the added `vistoken_compress` (and
`vistoken_prune`) blocks.

| run | architecture | compressor | prune | steps |
|---|---|---|---|---|
| `banded_mix_fc_3.1` | 9 aux → 3 band mixes → 3H→H FC, EAGLE 3.1 | — | — | 66,466 (2 ep) |
| `vistoken_k1` | **same** | k=1/tile, learned query, shared routing, T=8.0 | — | 66,466 (2 ep) |
| `vistoken_k1_attn_prune_x64_m16` | **same** | **same** | target-attn top-16 of 64 | 66,466 (2 ep) |

So the user-level expectation is right: **vistoken ⊇ banded_mix_fc, therefore it
should be ≥ banded_mix_fc.** It is not.

---

## 4. Results

vLLM V1 offline, `answer_then_describe` prompt, N=80/benchmark, K=4,
`max_num_seqs=1`, `enforce_eager`, one job at a time on GPU 0. Full tables in
`my_angel/one-layer-results.md`.

### Headline — temp 0, mean over 8 benchmarks

| run | acceptance τ | tok/s | speedup vs target-only |
|---|---:|---:|---:|
| `banded_mix_fc_3.1` | **2.706** | **326.8** | **1.673x** |
| `vistoken_k1` | 2.557 (−5.5%) | 302.5 (−7.4%) | 1.548x |
| `vistoken_k1_attn_prune_x64_m16` | 2.570 (−5.0%) | 301.9 (−7.6%) | 1.544x |
| `branch_change_top1_w01` *(current best)* | 2.815 (+4.0%) | 339.7 (+3.9%) | 1.738x |

**The compressor costs acceptance and buys no speed.** It is a net loss on both axes.

### The loss is image-only — which is the clean control

MATH-500 is the only text-only benchmark (avg input 71.9 tokens, no image), so the
compressor is inert there. It is at parity. Every image benchmark loses.

| run | image-mean τ (7 benches) | MATH-500 τ (no image) |
|---|---:|---:|
| `banded_mix_fc_3.1` | 2.5882 | 3.5323 |
| `vistoken_k1` | 2.4252 (**−6.3%**) | 3.4767 (−1.6%) |
| `vistoken_k1_attn_prune_x64_m16` | 2.4365 (**−5.9%**) | 3.5083 (−0.7%) |

Same picture at temp 1: image-mean −3.0%, MATH-500 −2.9% — smaller and less
separated, because τ ≈ 1.4 everywhere at temp 1 leaves little headroom.

### The degradation is flat across draft depth, not compounding

`acceptance_rate_pos_i` is **cumulative** (`accepted_at_pos_i / num_drafts`,
`tools/vllm_offline_eagle3_vlm_batch.py:1101-1104`), so it must be de-chained
before it means anything. Temp 0, mean over the 7 image benchmarks:

| | pos 0 | pos 1 | pos 2 | pos 3 |
|---|---:|---:|---:|---:|
| `banded_mix_fc_3.1` cumulative | 0.6262 | 0.4203 | 0.3037 | 0.2380 |
| `vistoken_k1` cumulative | 0.5987 | 0.3723 | 0.2606 | 0.1935 |
| cumulative delta | −4.4% | −11.4% | −14.2% | −18.7% |
| **conditional** (`rate_i / rate_{i-1}`) delta | **−4.4%** | **−7.3%** | **−3.1%** | **−5.3%** |

The cumulative column looks like a depth-growing pathology; the conditional column
shows it is not. It is a **flat ~5% per-token quality loss on image prompts** that
simply compounds over the 4-step chain.

### Attention-guided pruning changed nothing

`vistoken_k1_attn_prune_x64_m16` beats plain `vistoken_k1` by +0.5% τ at temp 0 —
inside run-to-run noise, and still 5% below the no-compressor baseline. This is
consistent with the pre-launch probe's own prior (top-16 rows hold 0.344 of the
score mass vs 0.250 flat: real but modest). The `mode: "random"` ablation that
would separate "pruning at all" from "this signal" was never run.

---

## 5. Is it undertrained? — No, and the gap is *widening*

Answering the adaptor-saturation question directly, from
`checkpoint-66466/trainer_state.json` of each run.

### Nothing is saturated at 2 epochs — but neither is the baseline

Both runs were still improving at the same rate in the last 10% of training
(`acc_0`: banded +0.52pt, vistoken +0.59pt over the final decile). The compressor's
own parameters were still growing roughly linearly between epoch 1 and epoch 2:

| parameter | ckpt-33233 | ckpt-66466 | growth |
|---|---:|---:|---:|
| `vistoken.k_proj.0.weight` norm | 26.49 | 34.88 | +31.7% |
| `vistoken.queries` norm | 3.47 | 4.48 | +28.8% |
| `vistoken.tile_embed` norm | 10.49 | 13.44 | +28.2% |
| `vistoken.row_embed_delta` norm | 4.29 | 6.30 | +46.6% |

So the compressor is genuinely **far from converged**. That is the argument *for*
more epochs.

### But the argument against is stronger: top-1 is already at parity, and the distribution gap grows

Next-token top-1 accuracy is **identical** to the no-compressor baseline, on both
train and held-out eval:

| metric | `banded_mix_fc_3.1` | `vistoken_k1` | Δ |
|---|---:|---:|---:|
| train `acc_0`, final decile | 0.6892 | 0.6895 | **+0.04%** |
| eval `acc_0` @ epoch 1 | 0.6414 | 0.6407 | **−0.11%** |
| eval `acc_3` @ epoch 1 | 0.6058 | 0.6035 | −0.38% |

The *distribution*, however, is much worse — and getting relatively worse every
decile. `ploss_0` (soft CE against the teacher), ratio vistoken / banded:

| decile of training | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ploss_0` ratio | 1.281 | 1.332 | 1.379 | 1.429 | 1.475 | 1.506 | 1.524 | 1.526 | 1.548 | **1.565** |

Monotonic, no sign of turning over. Held-out agrees: `eval_loss` 2.068 vs 3.220 at
epoch 2; `eval/ploss_0` 0.674 vs 0.988 at epoch 1 (**+47%**, and since 50.1% of the
132,943 training samples carry an image — 66,571 of them, counted from
`dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl` — the image-only gap is
roughly double that).

**Conclusion: more training will not close this.** The compressor preserves the
argmax and destroys the calibration, and the calibration gap is diverging, not
converging. A third epoch buys a slightly better compressor of a kind that is
already known not to help.

### The routing barely learned anything

The most concrete symptom, read straight out of `checkpoint-66466/model.safetensors`:

- **`ref_mix` is still essentially uniform after 2 epochs.** The softmax over the
  9 aux streams that selects the routing reference sits at
  `L2=0.120 L4=0.116 L8=0.119 L10=0.117 L15=0.122 L18=0.119 L20=0.110 L26=0.092 L28=0.084`
  against 0.111 for perfectly uniform. The compressor never picked a depth to
  route on. (The attn-prune run differentiates a little more — `L20=0.167` — which
  is the one place pruning visibly changed the learned solution.)
- `tile_embed` rows 17–31 are **exactly zero** (15 of 32 never received a
  gradient), confirming no prompt in the training set exceeded 17 tiles.
- `queries` norm 4.48 vs mean `tile_embed` norm 1.67 — the query content dominates
  the tile-position term, i.e. tiles are still weakly distinguished.

---

## 6. The unresolved risk in the decode path

`TODO-vistoken-k1.md` flagged one thing to check on the first real run, and **it
was never checked.** In the compact prefill,

```python
seq_len = int(self.input_buffers.positions[:num_tokens].max().item()) + 1
```
`speculator.py:651`

`seq_lens` stays the **full** prompt length (~1000) while only `compact_n` (~100)
slots are actually written. The draft's attention window therefore still spans the
~900 image-position KV slots that this request never wrote. What is in them was
never inspected.

Two reasons this is **probably not** the main explanation, but is still worth
closing:

- it would predict a *depth-growing* degradation (more decode steps, more reliance
  on cache state); the conditional per-position numbers in §4 show a flat one;
- MATH-500 (no image → no dropped rows → no holes) is at parity, which is what a
  cache-hole bug would also predict, so it does not discriminate.

Verifying it costs one instrumented run and would remove the last alternative to
"the compressor genuinely loses information".

---

## 7. Why the speed argument does not work either

Even at neutral acceptance, compression would not have paid. Throughput went
**down** 7.4%. The image rows only cost the drafter in **prefill**, which happens
once per request and is amortized over hundreds of decode steps; per decode step
the drafter processes one row regardless. Compression attacks a cost that is not
on the critical path, and adds the compressor's own forward to every request.

This means **row compression can only ever pay through better acceptance, never
through lower latency** — which reframes the whole line of work: it is a
representation-quality idea, not an efficiency idea.

---

## 8. What "plan 2" would be

There is **no Plan 2 / Idea 2 recorded anywhere in this repo** — grep for
`idea 2|plan 2|second idea` across `feature/vistoken-attn-prune`,
`feature/visual-compression`, `main` and `feature/branch-distillation` returns only
the `# TODO — visual row compression (Idea 1)` heading. The design doc's forward
plan is an ablation list, not a second idea:

| candidate | what it settles | cost |
|---|---|---|
| **Offline routing-entropy probe** on the trained k=1 checkpoint | Whether the learned query is doing anything at all, or has collapsed to a mean pool. `ref_mix` being uniform is already suggestive. | ~30 min GPU, no training |
| `query_mode: "mean"` at k=1 | The null baseline. If plain averaging matches the learned compressor, the learned-query design is dead. | 1 training run |
| `num_queries: 4` / `16` | Whether k=1 is simply too aggressive — 13–17 rows for a whole image. The τ curve should peak, not saturate. | 1–2 runs |
| `vistoken_prune: {mode: "random"}` | Separates "pruning at all helps" from "the target-attention signal helps". | 1 run |
| Instrument `seq_lens` / cache holes (§6) | Closes the last decode-path alternative explanation. | 1 eval run |

**Recommendation.** Run the offline routing-entropy probe *first* — it is the only
item that costs no training, and combined with the uniform `ref_mix` it can kill or
rescue the whole line in an afternoon. If the routing has collapsed to a mean pool,
`num_queries: 4/16` is the only ablation worth the GPU time; if it has not, then
the compressor is making a real but wrong choice and `query_mode: "mean"` is the
comparison that matters.

Set against the alternative use of the same GPUs: `branch_change_top1_w01`
(**+4.0% τ, +3.9% tok/s** over `banded_mix_fc_3.1`) is a working direction with
several unexplored knobs, while visual compression is at **−5.5% τ** after two full
runs. That asymmetry, not any single number above, is the argument for treating
vistoken as a probe to close out rather than a line to extend.

---

## 9. Reproducing

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
```

The eval log line that confirms the compressor actually armed:

```
[speculator.py:536] vistoken: 1 rows/tile at slots (32,), 9 aux streams; compact draft prefill armed
```

Present in every `rerun_atd/temp{0,1}/_logs/<bench>.log` of both vistoken runs —
checked, the numbers above are with compression on.
