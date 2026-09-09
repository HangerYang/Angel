# SmolVLM-256M visual-token compression — VLMEvalKit handoff

Written 2026-09-08. Everything below was measured on this machine; numbers are
copy-pasteable into a paper table. The last section is the **plan that has not
been run yet** — start there.

---

## 0. TL;DR

At a 64-token budget on SmolVLM-256M, **every published training-free token
selector loses to simply shrinking the image**, on all 9 benchmarks. The reason
is architectural, not a bug: Idefics3 *upscales* every image to `longest_edge`
before tiling, so the extra 768 crop rows are interpolation artefacts with no
independent information in them. Selecting among them cannot beat resampling.

So the training-free line is closed. The open line is a **trained compressor over a
frozen target** — the target model is never fine-tuned, which rules out most of
the published literature (§6, Related work). One such compressor is already
trained here and has never been benchmarked.

---

## 1. Where things live

| what | path |
|---|---|
| VLMEvalKit clone (patched) | `/home/hyang/VLMEvalKit` |
| driver / report / method code | `AngelSlim/my_angel/vlmevalkit/` |
| results tree | `~/tmp/vek/out/<arm>/SmolVLM-256M/` |
| pinned transformers 4.54.0 | `~/tmp/vek/tf454` |
| port-verification script | `~/tmp/vek/verify_ports.py` |
| Q-Former module | `AngelSlim/angelslim/compressor/vistoken/qsampler.py` |
| Q-Former checkpoints | `AngelSlim/output/qsampler-n{1,4,8,16}/final/qsampler.pt` |

Branch `feature/vistoken-attn-prune`.

`my_angel/vlmevalkit/` is everything that must travel to another machine:

| file | lines | why it is needed |
|---|---:|---|
| `smolvlm_select.py` | 521 | **the actual method code** — all selectors and the `inputs_embeds` surgery. Copy it to `<VLMEvalKit>/vlmeval/vlm/smolvlm_select.py`. Nothing else reproduces this. |
| `smolvlm.py.patch` | 59 | `git apply` onto `vlmeval/vlm/smolvlm.py`. Also transcribed in §3 if it conflicts. |
| `run_all.sh` | 111 | encodes the transformers pin, the warm-up ordering, the judge policy and the 45m timeout — i.e. §2's four pitfalls |
| `warm_images.py` | 41 | the single-process image decode that prevents the NCCL hang |
| `report.py` | 80 | per-benchmark score-file quirks (acc.csv / score.csv / OCRBench JSON / MME ÷28 / MCQ fractions) |
| `smoke_arms.py` | 106 | 96-cell smoke matrix through the real `generate_inner`; run after any port |

VLMEvalKit itself is a public repo — clone it fresh, apply the patch, drop in
`smolvlm_select.py`. The results tree `~/tmp/vek/out/` does **not** need to move;
every number is already in §5.

### Reproducing the environment elsewhere

```bash
pip install --target $TF_PREFIX transformers==4.54.0     # PIN. see §2
PYTHONPATH="$TF_PREFIX:$VEK"                             # both, in this order
```
`vlmeval` must resolve to `/home/hyang/VLMEvalKit`, not to any other checkout —
on this box the `eval` conda env has a second `vlmeval` pointing at
`visual-latent-CoT/VLM`, which is why `PYTHONPATH` is set explicitly.
Missing deps installed by hand: 7 packages, `pip install` on first ImportError.

---

## 2. Hard-won pitfalls (each of these cost real time)

1. **transformers is pinned to 4.54.0.** 4.57.6 scores ChartQA 38.88 instead of
   55.44 at 832 rows — a −16.6 regression, wrong predictions, not a scoring
   artefact. Do not upgrade.
2. **Warm the image cache single-process before any torchrun.** Concurrent
   ranks racing on a half-written JPEG raised `UnidentifiedImageError` on rank 0
   while the others hung in NCCL: 68 minutes lost on DocVQA_VAL. Use
   `warm_images.py`, and keep `timeout 45m` on every run.
3. **Judge policy.** Without `--judge`, `run.py:344-357` routes MCQ/Y-N sets to
   gpt-4o-mini, and `LOCAL_LLM` in `.env` then hijacks that to a local
   Qwen3-235B — silently changing the scoring convention. All numbers here use
   `--judge exact_matching` (no judge model anywhere).
4. **Do not `pgrep -f` a pattern that matches your own launcher's command line.**
   Cost 15 minutes of deadlock. Chain scripts instead of polling.
5. `vlmeval/vlm/smolvlm.py` line 1 is `import os.path as osp`, which binds only
   `osp` — add a bare `import os` if you re-apply the patch by hand.
6. Multi-image rows (MMMU has 47) break any "the global tile is the last 64
   rows" assumption. Locate `<global-img>` by token id **49152** instead.

---

## 3. The two patches in `vlmeval/vlm/smolvlm.py`

```python
import os                      # ADDED (line 1 only binds `osp`)
import os.path as osp
...
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

In `SmolVLM.__init__`, right after `AutoProcessor.from_pretrained`:
```python
edge = os.environ.get("SMOLVLM_LONGEST_EDGE")
if edge:
    self.processor.image_processor.size = {"longest_edge": int(edge)}
    warnings.warn(f"image processor longest_edge overridden to {edge}")

from .smolvlm_select import build_selector
self.row_selector = build_selector()
```

In `generate_inner`, before the normal generate:
```python
if self.row_selector is not None:
    from .smolvlm_select import generate_selected
    return generate_selected(self.model, self.processor, inputs,
                             self.row_selector, self.kwargs).strip()
```

`vlmeval/vlm/smolvlm_select.py` (~450 lines, new) holds the selectors and the
`inputs_embeds` surgery. Env vars:

| var | meaning |
|---|---|
| `SMOLVLM_TOKEN_SELECT` | `method:budget`, or `global_only`, `fastv@K:budget`, `hybrid:budget` |
| `SMOLVLM_LONGEST_EDGE` | pixel arm; `N*512` |
| `SMOLVLM_GLOBAL_POLICY` | `exclude` (default) / `include` / `only` |
| `SMOLVLM_SELECT_SEED`, `SMOLVLM_SELECT_STATS`, `SMOLVLM_HYBRID_SPLIT` | — |

Surgery notes: HF `generate` returns only new tokens when given `inputs_embeds`;
you must pass `attention_mask` yourself; positions re-densify after the splice.
Stats are flushed via `atexit.register` (an explicit `dump()` was never reached).

---

## 4. Tile arithmetic — read this before designing anything

SmolVLM-256M = SigLIP-base-patch16-**512** + pixel shuffle `scale_factor=4`.
512/16 = 32 patches per side → 1024 patches → 1024/16 = **64 tokens per tile,
always**. Tile *count* varies; per-tile token count never does.

`preprocessor_config.json`: `size={longest_edge: 2048}`,
`max_image_size={longest_edge: 512}`, `do_image_splitting=true`.

**`size` is a target, not a cap.** `_resize_output_size_rescale_to_max_len:71`
does `width = max_len` with no `min()` against the original, so a 64×64 image is
upscaled 32× to 2048×2048. Token count therefore depends only on **aspect ratio**:

| input | resized to | grid | tiles | tokens |
|---|---|---|---|---|
| 64×64 … 336×336 | 2048×2048 | 4×4 | 17 | **1088** |
| 1600×1067, 3840×2160 | 2048×~1200 | 4×3 | 13 | **832** |
| 2048×512 | 2048×512 | 4×1 | 5 | 320 |

A 64×64 thumbnail costs *more* visual tokens than a 4K photo.

### The resolution knob

The model card (README:277-278) documents `size={"longest_edge": N*512}`:

| N | edge | grid | tiles | tokens |
|---|---|---|---|---|
| 1 | 512 | 1×1 | 1 | **64** |
| 2 | 1024 | 2×2 | 4+1 | 320 |
| 3 | 1536 | 3×3 | 9+1 | 640 |
| 4 (default) | 2048 | 4×3 | 12+1 | 832 |

Only these steps exist — no 192, no 256. (Qwen2.5-VL is continuous by contrast:
patch 14, merge 2, one token per 28×28 px, range 4..16384, no 64-token floor and
no global thumbnail tile.)

**N=1 is exactly `do_image_splitting=False`.** `split_image:415` guards on
`height > 512 or width > 512`; at N=1 it fails, `num_splits = 0,0`, and only the
image itself is appended. Both paths end at a stretched 512×512 square
(`preprocess:786-796` vs `resize_for_vision_encoder`), differing by one
interpolation hop. This is a *documented* setting, not OOD — and the degradation
is smooth and task-shaped (ChartQA −26.6, SciQA −1.4), which is a resolution
signature, not a distribution break. Prefer the phrase "officially documented
resolution setting" over "in-distribution" in writing.

`<global-img>` is the **last** tile: the whole picture resized to 512². Tiles are
encoded independently along the batch dim, so it is *not* a CLS-like aggregator.
SmolVLM's vision tower has **no CLS token** (`embeddings` has only
`patch_embedding` + `position_embedding`) — this rules out VisionZip and
PruMerge+ outright.

---

## 5. Results

All: SmolVLM-256M-Instruct, transformers 4.54.0, `--judge exact_matching`.
MMMU is validation split only. MME = (perception+reasoning)/28. OCRBench ÷10.

### 5.1 The main table (9 benchmarks, 64-token budget)

| arm | ChartQA | OCRBench | MMMU | MMStar | MME* | SciQA | AI2D | TextVQA | POPE | **mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 832 tok, N=4 (ceiling) | 55.44 | 52.60 | 30.00 | 34.93 | 44.04 | 71.72 | 47.28 | 49.94 | 76.85 | **51.42** |
| 320 tok, N=2 | 52.56 | · | · | · | · | · | · | · | · | — |
| 64 tok, **N=1 pixel** | 28.84 | 35.60 | 31.33 | 32.33 | 36.32 | 70.34 | 45.73 | 38.75 | 68.14 | **43.04** |
| 64 tok, `<global-img>` surgery | 30.40 | 33.60 | 25.89 | 34.33 | 40.00 | 70.43 | 44.24 | 37.78 | 69.44 | 42.90 |
| 64 tok, DivPrune @ N=2 | 26.52 | 9.80 | 29.33 | 28.73 | 35.11 | 57.08 | 42.52 | 31.42 | 50.37 | 34.54 |
| 64 tok, DivPrune @ N=4 | 19.24 | 5.70 | 26.56 | 27.27 | 28.45 | 54.32 | 37.60 | 24.43 | 29.99 | 28.17 |

DivPrune loses to the pixel baseline on **9/9**. FastV is worse still (192
tokens: ChartQA 13.44, POPE 37.25, TextVQA 18.45).

`global_only` vs `img64`: 5 wins to 4, mean differs by 0.14 — **they are the
same arm**. Both feed the LM 64 rows of the whole image; the residual is the
resize chain (N=1 does one `resize(512)`; the surgery upscales to 2048, tiles,
then resizes back). `global_only` is `img64` at 13× the vision-encoder cost —
demote it to a footnote.

### 5.2 ChartQA budget ladder (why the trend matters)

| budget | select from 832 (N=4) | select from 320 (N=2) | pure pixel |
|---:|---:|---:|---:|
| 32 | 12.48 | 18.20 | — |
| 64 | 19.24 | 26.52 | **28.84** |
| 128 | 26.12 | 35.36 | — |
| 192 | 31.12 | 40.24 | — |
| 320 | — | — | **52.56** |
| 832 | — | — | 55.44 |

Middle column climbs toward the right column at every budget and never reaches
it. Less upscaling → better selection; the limit of that trend is N=1, where
there is nothing left to select. **Selection has no viable window above 64 tokens
on this architecture:** small N gives a clean pool with no slack, large N gives
slack full of interpolation noise.

### 5.3 Controls (ChartQA, 64 from 768 crop rows at N=4)

| | ChartQA |
|---|---:|
| DivPrune | 19.24 |
| random, seeds 0/1/2 | 12.56 / 13.84 / 12.60 |
| stride | 11.88 |
| norm | 24.52 |
| CDPruner | 20.44 |

DivPrune beats random by ~6 points with seed variance 1.3 — the method works,
the *pool* is what is broken. (norm/stride/random are internal controls only;
they never go in the paper.)

### 5.4 Below 64 tokens — the one regime not yet closed

`g_dp32` (DivPrune, 64 clean global rows → 32): **ChartQA 24.92** vs 30.40 for
all 64. `g_dp16`, `g_pool16`, `g_pool32` were launched and killed; no data.

This regime is different in kind: **32/16/8 tokens cannot be expressed by
resizing at all**, because 1 tile = 64 is a hard floor. Below 64 there is no
pixel baseline to lose to, so every §5.1 conclusion is void here. **`g_dp32` has
no random control** — that is the single most important missing cell if this
line is pursued.

---

## 6. NEXT — the Q-Former plan (nothing here has been run)

### Framing

- **Ceiling** = N=4, 832 tokens, mean **51.42**.
- **Base** = N=1, 64 tokens, mean **43.04**.
- **Goal**: train a Q-Former that emits ~64 tokens and recovers the 8.38-point
  gap — i.e. beat 43.04 and approach 51.42 at the same token budget.

This is a strictly better-posed target than the training-free work: the baseline
to beat is a *number*, not another method.

### The asset that already exists

`angelslim/compressor/vistoken/qsampler.py` — `N` learned queries cross-attend
over one tile's 64 keys (learned 8×8 2D positional embedding on the keys),
pre-norm blocks, output in LM embedding space as a drop-in for
`Idefics3Connector`. `init_mode="mean_pool"` starts block 0 as uniform average
pooling; `out_scale`/`calibrate_output_scale` matches the connector's per-token
RMS (~5.16, ~40× the text embedding scale — get this wrong and the LM reads a
blank image).

Trained (2 epochs, 4 GPUs, ~21 min each, conda env `angel`), on
`dataset/smolvlm_256m_target_gen_mixed_70k70k`, loss = KL to the target +
0.3 × cosine on layer 26:

| N | eval KL | top1_agree |
|---|---:|---:|
| 1 | 0.405 | 0.795 |
| 4 | 0.274 | 0.828 |
| 8 | 0.220 | 0.843 |
| 16 | 0.193 | 0.852 |

Eval flattens after ~step 1000; more than 2 epochs buys nothing at this size.

**These have never been evaluated on any benchmark.** They were built for the
EAGLE3 drafter and shelved as "not what EAGLE3 consumes" — but for feeding the
**target** model directly, the training objective (target-invariance) is exactly
right. Measuring them is nearly free.

### It is per-tile — this drives the whole design

`qsampler.py:161`: `[T, 64, H] → [T, N, H]`, shared weights, run per tile. So
output length = `tiles × N`, and the `<row_i_col_j>` grid markers stay valid.

| | 13 tiles (N=4) | 5 tiles (N=2) | 1 tile (N=1) |
|---|---:|---:|---:|
| n16 | 208 | 80 | **16** |
| n8 | 104 | 40 | **8** |
| n4 | 52 | 20 | 4 |

Consequences:
- **A ~64-token arm needs `num_queries≈5` at N=4** (13×5 = 65) or
  `num_queries=13` at N=2 (5×13 = 65). Neither is trained yet; both reuse the
  existing code unchanged.
- **A fixed 64 output regardless of aspect ratio needs a *global* Q-Former**
  (queries attend over all tiles at once). That breaks the per-tile marker
  structure and needs positional encoding over the full tile grid. Worth it only
  if variable length turns out to matter.
- The already-trained n16 at **N=1 gives exactly 16 tokens** — the 16/8 budget
  wanted for the downstream hand-off-to-another-model goal. And a 1-tile input
  is the same distribution as the `<global-img>` tile the sampler already saw in
  training, so this is not OOD for it.

### Suggested order

1. **Free measurement first.** Wire `QSampler` into `smolvlm_select.py` as a
   selector — image rows in `inputs_embeds` *are* connector output, so reshape to
   `[T,64,H]`, run the sampler, splice back (~30 lines). Evaluate the existing
   n16/n8/n4 checkpoints at N=1 and N=2 on ChartQA + TextVQA. ~40 min.
   This tells you whether KL 0.193 translates into benchmark accuracy at all
   before spending any training time.
2. **Train the ~64-token arms**: `num_queries=5` @ N=4, and `num_queries=13`
   @ N=2 (the "compress the image first, then train" variant). Compare both to
   43.04, ceiling 51.42.
3. **Controls, always.** `g_random{16,32}` (2-3 seeds) and `g_pool{16,32}`.
   Without them a Q-Former win is unattributable.
4. Only then consider the global (non-per-tile) variant.

### Related work — trained compressors under a FROZEN target

**Standing constraint: the target model is frozen. Only the compressor is
trained.** Most of the visual-compression literature does not respect this — it
trains the projector *and* the LLM together, so the reported numbers are not
comparable to anything measured here. The list below is split by whether a
method survives that constraint.

Checked 2026-09-08. arXiv ids only where confirmed from the source page.

#### A. Directly compatible — frozen backbone, compressor-only training

- **BLIP-2 Q-Former** — the precedent for the whole setting: a resampler trained
  against a **frozen** image encoder and a **frozen** LLM. QSampler is this idea,
  per-tile, in the LM's embedding space. Cite it for the setting, not the numbers.
- **VisionSelector** — arXiv **2510.16598**. Explicitly plug-and-play: the
  pretrained MLLM backbone stays **frozen**, only the selector trains — 16.87M
  parameters, trained at a fixed 20% retention budget, evaluated across
  architectures (e.g. LLaVA-OneVision-1.5-8B). **The closest published match to
  our setup and the natural baseline to reproduce**: end-to-end learnable
  selection, so it is the trained answer to the same question DivPrune answers
  training-free.
- **EvoComp** — arXiv **2604.17087**. A lightweight compressor between the
  alignment module and the LLM, emitting retention probabilities over visual
  tokens; stated to require **no fine-tuning of the vision encoder, the
  alignment module, or the LLM**. Same slot in the pipeline as QSampler.
- *(ours)* **QSampler** — §6. Frozen SmolVLM, trained to target-invariance
  (KL + layer-26 cosine). Already satisfies the constraint by construction.

The existence of A is the point: with a frozen target, a trained compressor
still reaches top1_agree 0.852 at N=16 on our own data, and BLIP-2 shows a
frozen LLM will accept a learned token distribution at all. The setting is not
exotic.

#### B. Architecture reusable, recipe is not

These are drop-in projector designs. The **module** can be trained with the
target frozen; the **papers** tune the LLM as well, so do not quote their
accuracy as a baseline — reproduce them in our setting or omit the number.

- **Honeybee** (C-Abstractor / D-Abstractor) — arXiv **2312.06742**, CVPR 2024.
  Argues a plain Q-Former destroys *locality*, and restores it with convolution
  or deformable attention while keeping the token budget freely settable. **The
  sharpest architectural critique of what we have** — QSampler's learned 8×8 key
  positional embedding is a weaker version of the same fix. C-Abstractor is the
  ablation to add.
- **DeCo** — arXiv **2405.20985**. Decouples token compression from semantic
  abstraction, and argues the Q-Former conflates the two and **loses to plain
  adaptive pooling** at equal budget. Read before believing any Q-Former win:
  it predicts our `g_pool` control is competitive, and those cells are empty.
- **TokenPacker** — coarse-to-fine projector (downsample → point-to-region
  cross-attention → cross-layer fusion). (id not verified)

#### C. Ruled out — they require training the target

Listed so nobody re-derives them. Their *diagnoses* may still be usable; their
methods are not.

- **LLaVA-Mini** — arXiv **2501.03895**, ICLR 2025. 1 vision token vs 576. But
  "modality pre-fusion" inserts LLM blocks ahead of the backbone and the whole
  model is trained. **Unusable as a method here.** Its finding is still worth
  citing and is cheap to test on SmolVLM: vision tokens matter mainly in the
  *early* LLM layers, where they fuse into text.
- **Victor** — learnable *register* tokens; visual tokens are dropped after a
  few layers and reasoning continues on ~8 registers (>96% of VQA accuracy).
  The registers are processed by the LLM, so the LLM must be trained to use
  them. Out. (id not verified)
- **MQT (Matryoshka Query Transformer)** — NeurIPS 2024. Trained jointly with
  LLaVA, so the method is out — **but the training trick transfers unchanged**:
  each step uses only the first *m* of *M* latent queries, *m* random, giving one
  model that serves any budget at inference. Applied to QSampler this collapses
  the whole n1/n4/n8/n16 sweep into a single run and yields 16/8/4 for free. It
  is a sampling change in the training loop, nothing more.

#### Index / unread

- Survey: *A Survey of Token Compression for Efficient MLLMs* — arXiv
  **2507.20198** (TMLR 2026). Lists:
  `github.com/cokeshao/Awesome-Multimodal-Token-Compression`,
  `github.com/daixiangzi/Awesome-Token-Compress`.
- *Are We Using the Right Benchmark: An Evaluation Framework for Visual Token
  Compression* — arXiv **2510.07143**. Read this before finalising the table;
  it is about exactly the measurement question §5 ran into.
- *IPCV* — arXiv **2512.18747**. *LaCo* — arXiv **2507.02279**. *Vision
  Remember* — arXiv **2506.03928**. *Learning Compact Vision Tokens* — arXiv
  **2506.07138**.

**How this changes the plan:** the baseline set becomes
`{VisionSelector, C-Abstractor, adaptive pooling}` at matched budget — the first
is the only published method that already runs under our constraint, the second
is the architectural challenger, the third is DeCo's warning. Train with
MQT-style query dropout so one run covers 64/16/8/4.

### Failure mode to design against

`VisRowCompressor` (the earlier attempt, `my_angel/VISUAL-COMPRESSOR.md`,
commits 1d9d939 / b36a671) **collapsed into a one-hot row picker**: routing
softmax effective support `exp(H(w)) = 1.00`, top-1 weight 0.9987, raw logit
range 173,616 against `temperature=8.0`. Cause: `weight_decay=0` and no entropy
penalty, so `k_proj` grew unboundedly. It trained a row *sampler*, not a
compressor, and was a net loss (temp-0 8-benchmark mean τ 2.557 vs 2.706).

`train_qsampler.py` already uses `weight_decay=0.05`, which is the main guard.
**Log `exp(H(attention))` per block during training anyway** — if it approaches
1.0 the Q-Former has degenerated into the selection methods that §5 already
proved lose.

### Constraints to respect (user's, standing)

- **No loading extra models.** This killed CDPruner, whose relevance term needs
  a CLIP/SigLIP text tower.
- **Only published methods go in the paper.** norm/stride/random/pool are
  controls, never deliverables.
- The method must eventually run on **Qwen2.5-VL** and **Qwen3-VL-4B**. Qwen has
  no upscaling floor and no global tile, so §5's "just shrink the image" result
  very likely **reverses** there — its crop tokens carry real information. That
  cross-model control is close to mandatory for a submission.
- Verify any ported method against the official implementation before trusting a
  number. DivPrune and CDPruner were checked with `~/tmp/vek/verify_ports.py` and
  produce **identical index sets** (64/64, 64/64, 128/128) on identical tensors.

---

## 7. Still open / not done

- **Hybrid** (`_sel_hybrid`: thumbnail + DivPrune) is implemented and
  smoke-tested, never run.
- **Qwen2.5-VL-7B** replication: token counts measured, selector port to Qwen's
  grid not started (~40 min).
- Missing cells: `img320` has only ChartQA; `g_dp16` / `g_pool16` / `g_random*`
  at the sub-64 budget.
- Not yet committed at the time of writing.
