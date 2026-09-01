# Visual compression for the SmolVLM EAGLE-3 drafter

Status as of **2026-09-01**. Two runs trained, both evaluated, both a net loss —
and §6 explains why: the learned routing collapsed to a one-hot pick, so what
trained was a row *sampler*, not a compressor.
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
| 11 | **Routing-quality probe** — rebuild the trained compressor from a checkpoint, run the real target on 46 image samples / 626 tiles, and measure what the learned query actually does. This is what found the failure in §6. | `my_angel/vistoken_routing_probe.py` |

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
already known not to help — and §6 shows the routing distribution is collapsing
*further* every epoch, so more training actively makes the compressor worse at
the one thing it exists to do.

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

## 6. The failure: it is not compressing, it is hard-selecting one row of 64

`my_angel/vistoken_routing_probe.py`, real SmolVLM-256M target, **46 image
samples / 626 tiles**, both checkpoints of `vistoken-k1`.

### The routing softmax is completely saturated

| | ckpt-33233 (1 ep) | ckpt-66466 (2 ep) | reference |
|---|---:|---:|---|
| effective support `exp(H(w))` | **1.00** | **1.00** | 1 = hard pick, 64 = mean pool |
| top-1 routing weight | **0.9985** | **0.9987** | uniform = 0.0156 |
| raw logit range (max−min) | 64,373 | **173,616** | trained `temperature` = **8.0** |

The design intended a *convex combination* of a tile's 64 rows. The optimizer went
to a **corner of the simplex** instead. Since `VisRowCompressor` deliberately has
**no value projection**, a one-hot `w` means the "summary" of a tile is a
**verbatim copy of one of its 64 rows** — the other 63 contribute nothing.

So `vistoken-k1` is not a compressor. It is a learned **row sampler that keeps
1 row and drops 98.4% of the image**. That it costs only 6.3% acceptance is
almost impressive; that it never beat the baseline is not surprising.

### Root cause: nothing bounds the logit scale

`temperature = 8.0` is `sqrt(key_dim=64)` — correct for unit-scale keys. But
`weight_decay = 0.0` (read from `checkpoint-66466/training_args.bin`) and no
entropy penalty, so `k_proj` is free to grow without limit, and it does: its norm
went 26.49 → 34.88 between epochs (§5) and the resulting logit range **grew 170%,
64,373 → 173,616**. Divided by 8, the softmax sees logits ~21,700 wide. The loss
can always be reduced a little by committing harder, and nothing pushes back.

**This is the precise sense in which the adaptor's training quality is bad, and
it is the opposite of undertrained.** The model is still learning (§5) while the
routing distribution has already collapsed and is collapsing further every epoch.

### Un-saturating it would not rescue it — the query direction carries no signal

Re-softmaxing the *same trained logits* at higher temperature, against the
target's own attention over the same rows (ckpt-66466):

| temperature | `exp(H)` | Pearson r(w, target score) | top-16 Jaccard |
|---:|---:|---:|---:|
| 8 (trained) | 1.00 | 0.0813 | 0.2218 |
| 32 | 1.01 | 0.0815 | 0.2244 |
| 128 | 1.04 | 0.0822 | 0.2387 |
| 512 | 1.16 | 0.0844 | 0.2620 |
| 2048 | 1.88 | 0.0965 | 0.2606 |
| | | | *chance = 0.1429* |

At 256x the trained temperature the correlation moves from 0.081 to 0.097. **The
learned query direction is essentially uninformative about what the target
reads.** Fixing the temperature alone would turn a hard pick of an arbitrary row
into a soft average weighted by an arbitrary direction.

### Is the picked row at least sensible?

- **It is content-dependent, not a fixed habit.** The same tile index picks the
  same row only **20.6%** of the time across different images; 57 of the 64 row
  slots get used (effective 24.6).
- **But it disagrees with the target.** Top-16 Jaccard against the target's own
  attention ranking is **0.222** against a chance value of 0.143, and r = 0.081.
  So it is choosing on *something*, and that something is nearly orthogonal to
  what the target model actually reads out of the tile.
- The selected row is **cos 0.820** from the plain tile mean, i.e. it is
  meaningfully different from the mean — it is not accidentally reproducing the
  averaging null.

Both checkpoints tell the same story, and epoch 2 is *worse* than epoch 1 on every
axis that matters: more collapsed (effective distinct rows 29.0 → 24.6), further
from the mean (cos 0.832 → 0.820), logit range up 170%.

Reproduce:

```bash
CUDA_VISIBLE_DEVICES=3 python my_angel/vistoken_routing_probe.py --n 48
```

---

## 7. The unresolved risk in the decode path

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

## 8. Why the speed argument does not work either

Even at neutral acceptance, compression would not have paid. Throughput went
**down** 7.4%. The image rows only cost the drafter in **prefill**, which happens
once per request and is amortized over hundreds of decode steps; per decode step
the drafter processes one row regardless. Compression attacks a cost that is not
on the critical path, and adds the compressor's own forward to every request.

This means **row compression can only ever pay through better acceptance, never
through lower latency** — which reframes the whole line of work: it is a
representation-quality idea, not an efficiency idea.

---

## 9. Plan 2, and what is worth running next

Plan 2 is a **two-stage** plan:

> **Stage A.** Train the compressor against the target model alone — maximize the
> information the compressed rows carry, such that the **target's own output is
> unaffected**.
> **Stage B.** *Then* train end-to-end with the drafter, starting from that.

### Stage A exists and ran. Stage B never ran, and there is no code for it.

Stage A is `tools/train_qsampler.py` on branch `mess`. It freezes the vision
tower, connector, text model and `lm_head`; compresses each tile's 64 connector
rows to N; **scatters them back into the `<image>` slots and runs the frozen
target LM on them** (`run_text_branch`); and trains on
`KL(target_full ‖ target_compressed)` over the text positions plus a hidden
cosine term. That is exactly "不影响 target output". Results are on disk in
`output/qsampler-n{1,4,8,16}` and `logs/qsampler_n*.log`, 2 epochs each:

| N per tile | compression | eval KL | **target top-1 agreement** | hidden cos |
|---:|---:|---:|---:|---:|
| 1 | 64x | 0.4043 | **79.4%** | 0.9439 |
| 4 | 16x | 0.2740 | 82.8% | 0.9632 |
| 8 | 8x | 0.2201 | 84.3% | 0.9726 |
| 16 | 4x | 0.1930 | **85.2%** | 0.9763 |

Read the third column as *how often the target, fed compressed visual tokens,
still predicts its own original next token*. Even at 4x it disagrees with itself
**15%** of the time; at the 64x ratio `vistoken-k1` uses, **20%**. Eval flattens
after ~step 1000, so more epochs buy nothing. **64x sits at the expensive end of a
still-falling curve** — an independent argument for `num_queries: 4/16` over k=1.

Stage B was never attempted. `row_compressor.py:9` records the decision:
qsampler is *"(dropped): that compressed the connector output in LM embedding
space against a target-invariance objective"*, and `TODO-vistoken-k1.md` adds
**"not even used as an init"**. There is no code path that loads a trained
QSampler (or any pretrained compressor) into the EAGLE trainer —
`_build_vistoken_compressor` (`llama_eagle3.py:1038`) constructs a fresh
`VisRowCompressor` from config and nothing else.

### Why Stage A and Stage B do not currently connect

The two stages live in **different spaces**, and it is structural, not incidental:

| | QSampler (Stage A) | VisRowCompressor (what the drafter uses) |
|---|---|---|
| operates on | connector output, **before** the target LM | 9 aux hidden states, **after** the target LM |
| shape | `[64, 576]` → `[N, 576]` | `[64, 9×576]` → `[k, 9×576]` |
| fed **into** the target? | yes — that is what makes Stage A definable | **no** — these are outputs of the target |

Stage A's objective only exists where the compressed tensor is something the
target *reads*. Aux hidden states are something the target *emits*, so
"不影响 target output" has no meaning there — there is no second forward to
compare against. A QSampler also cannot be transplanted into the vistoken slot:
wrong space, wrong width.

So Plan 2 needs one of these, and it is a real decision:

| | how Stage A becomes definable | cost / catch |
|---|---|---|
| **A. Two target forwards** | Target runs once on the full image (the real one, being verified) and once on QSampler-compressed embeddings; the drafter's aux HS come from the second. Stage A pretraining is then *exactly* the right init and Stage B is natural. | One extra target prefill per request. Must be measured against the drafting cost it saves — §8 says that budget is thin. |
| **B. Keep aux-HS compression, replace Stage A's objective** | "Unaffected target output" is not available, but *information fidelity* is: train the compressor so a frozen readout reconstructs the tile's rows, or so the target's remaining layers produce the same final hidden state. Then Stage B end-to-end. | Keeps the current architecture and the fixed test-time setup. Needs a new pretraining objective written. |
| **C. Let the target itself run on compressed tokens** | Trivially definable. | **Ruled out** — the fixed test-time setup is that the target reads the full image; this would be a train/test mismatch. |

### The finding in §6 blocks Plan 2 either way

Whichever of A or B is chosen, Stage A produces a compressor with a *spread-out*
routing distribution — that is what "carries the most information" means. Stage B
then trains it end-to-end **under the same objective and the same optimizer
settings that produced the one-hot collapse in §6**: `weight_decay=0.0`, no
entropy penalty, logit range growing 170% per epoch.

Stage B as currently configured would destroy Stage A's result within an epoch.
**Fixing the routing geometry is a prerequisite for Plan 2, not an alternative to
it.**

### The rest of the forward plan

The design doc's remaining items were an ablation list. §6 changes what is worth
running from it.

### Ruled out by §6, do not spend GPU time on these

- **`query_mode: "mean"` as a null.** It was the right experiment *before* the
  probe. Now we know the learned routing is a one-hot pick with r = 0.08 against
  the target's attention, so "does averaging beat this?" is no longer an open
  question worth a 4-hour run — it is the same question as "is an arbitrary
  single row worse than the tile mean?", and the answer only matters if the
  learned path is repaired first.
- **`vistoken_prune: {mode: "random"}`.** The prune feeds a selector that then
  hard-picks one row anyway. Pruning 64 → 16 before a one-hot pick explains the
  observed +0.5% τ (noise). The ablation would measure a stage that is not the
  bottleneck.

### The two things actually worth doing

| # | what | why | cost |
|---|---|---|---|
| 1 | **Repair the routing and retrain k=1**: normalize the dot product (`q·k / (‖q‖‖k‖)`, or a LayerNorm before `k_proj`), and/or put weight decay on the compressor group, and/or add an entropy floor on `w`. `weight_decay=0.0` today, so nothing bounds `‖k_proj‖`. | Without this, every other vistoken run repeats the same collapse — the failure is in the objective's geometry, not in `k`. | 1 training run |
| 2 | **`num_queries: 4` / `16` after the repair** | k=1 gives the drafter 13–17 rows for a whole image. Even a *correct* summarizer may not fit an image into that — the qsampler sweep above measures the same curve in a different space and 64x sits at its expensive end. Run it only after (1); with a one-hot router, raising k just samples k arbitrary rows. | 1–2 runs |

| 3 | **Plan 2 Stage B, once (1) is in.** Pick option A or B above, write the Stage-A objective for the chosen space, pretrain, then end-to-end. | This is the plan the whole line was supposed to follow; it has never been executable because Stage B has no warm-start path and, until (1), would erase Stage A anyway. | 1 pretrain + 1 end-to-end run |

`seq_lens` / cache-hole instrumentation (§7) stays worth one cheap eval run
whenever a vistoken checkpoint is next evaluated, since it is the only remaining
alternative explanation for the decode gap.

### The honest framing

The probe converts "visual compression does not work" into something sharper:
**it was never tested.** What trained was a row *sampler*, not a compressor. That
is a real bug with a known fix, and it is the one argument for spending another
run here.

Against that: `branch_change_top1_w01` is at **+4.0% τ, +3.9% tok/s** over
`banded_mix_fc_3.1` with several unexplored knobs, while vistoken is at **−5.5% τ**
after two full runs and needs a third just to reach a fair test. If only one run is
affordable, item (1) is the one that buys information — it either produces a real
compressor or retires the idea on evidence rather than on a bug. Item (1) is also
the cheapest thing that makes Plan 2 worth starting: pretraining a compressor for
information fidelity and then handing it to an optimizer that collapses it to a
one-hot pick wastes both runs.

---

## 10. Reproducing

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
