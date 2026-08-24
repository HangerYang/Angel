# Draft-speed study — plan

Goal: measure **drafting speed**, not quality. Acceptance length and output text do
not matter. Every variant is **1 layer**, run through a pipeline close enough to the
real eval that the numbers translate. Each component added on top of plain eagle3
gets its own timed delta, and we also mark what could be pruned out of plain eagle3.

---

## Original plan

1. First, run eagle3. plain eagle3. with OmniDocBench. Target model goes first to
   generate the first token and hidden states, then eagle3 continues to generate for
   128 tokens (max, for now).

2. Next, banded_mix_uninit — basically similar to eagle3, but instead of taking 3
   fixed hidden layers, each layer is a mix of a few layers. Here, try 3, 6, 9.
   Compare the speed. Still a 1-layer eagle3, so a hypothetical situation.

3. Then, HiViS style, more or less. The target model still uses the full img tokens,
   but now the draft model takes in only the text tokens, concatenated with the hs,
   eagle3 style.

4. Now, add a Q-Former that takes 4/8/16 img tokens to 1, calculate the time.

5. eagle 3.1 style.

---

## Resolved spec

| | decision |
|---|---|
| backend | **vLLM, eager** — as close to the real eval as possible. Fall back to torch only where vLLM can't express the architecture |
| dataset | **OmniDocBench** (887 in / 418 out avg) |
| prompts | **20** |
| draft budget | **32 cycles x K=4 = 128 drafted tokens**, constant for every variant |
| K | always 4, regardless of acceptance |
| target | in the loop each cycle (draft re-seeds from its hidden state); target time excluded |
| verification | ignored — mock run, sequence always advances 4/cycle |
| draft depth | **1 layer** everywhere |
| weights | **untrained / random init is fine** — speed is determined by shapes and FLOPs, not values. This is what makes #2, #4, #5 possible without training runs |
| metrics | ms/cycle, ms/drafted-token, delta vs plain eagle3, plus prune analysis |

### Why no sampler patch is needed

"Always K=4" cannot be forced in vLLM without patching the rejection sampler — but it
does not need to be. vLLM always *proposes* K tokens per cycle no matter how many are
accepted, so **per-cycle draft cost is acceptance-independent**. Measure per-cycle
draft time in a real vLLM run and multiply by 32.

### vLLM support matrix

| experiment | vLLM today |
|---|---|
| 1. plain eagle3 | native |
| 2. banded mix 3/6/9 -> 3H | native draft; mixing done on target hidden states (see below) |
| 2b. H vs 2H vs 3H, no mixing | native (just fewer aux streams) |
| 5. eagle 3.1 | native — `fc_norm` + `norm_output` |
| 3. HiViS text-only draft prefill | needs a proposer patch; prototype in torch first |
| 4. Q-Former | new module + config plumbing; prototype in torch first |

---

## Per-experiment notes

### 1. Plain eagle3 (reference)
1-layer, `fused_fc`, aux from 3 target layers. Everything else is measured as a delta
against this.

### 2. Banded mix — 3 / 6 / 9 layers into 3 bands

Eagle3's FC takes **3 aux streams**, concatenates to 3H, and fuses to H. **Those 3
streams are the bands.** Each band is a mix of several target layers, so the FC input
is always 3H no matter how many source layers feed it:

| variant | source layers | layers per band | FC input |
|---|---|---|---|
| 3 | 3 | 1 (= plain eagle3) | 3H |
| 6 | 6 | 2 | 3H |
| 9 | 9 | 3 | 3H |

Mixing within a band is a learned weighted sum over that band's target hidden states
(`[k, S, H] -> [S, H]`), untrained here since only speed matters. The measured delta
vs #1 is therefore the **cost of mixing**, with FC width held constant.

Footnote: the existing `progressive_banded_mix` mode cannot express this at 1 layer —
it validates one band per *draft* layer (`llama_eagle3.py:691,701`,
`num_layers = config.num_hidden_layers`), so a 1-layer draft admits only 1 band. That
mode isn't needed; plain `fused_fc` with 3 banded streams is the right construction.

**2b — same knob, other axis.** #2 fixes the band count at 3 and varies how many
source layers feed each band. 2b fixes the layers-per-band at 1 and varies the **band
count**, which is what sets FC width:

| variant | bands | layers per band | FC input |
|---|---|---|---|
| H | 1 | 1 | H |
| 2H | 2 | 1 | 2H |
| 3H | 3 | 1 | 3H (= plain eagle3) |

So both are band-count x layers-per-band, and the two together separate **FC input
width** (2b) from **mixing cost** (#2). The `3H` cell of 2b and the `3` cell of #2 are
both plain eagle3 — they must reproduce #1, which is the harness self-check.

### 3. HiViS style
Target unchanged (full image tokens). The **draft** prefills text positions only;
image positions are dropped from its input and KV. Measured against #1, where the
draft prefills all ~887 positions.

Note: ~95% of a SmolVLM prompt is image tokens (measured 832/878 on textvqa), so this
cuts the draft's KV by roughly 19x. Expect a large effect, and note this measures the
**speed ceiling** HiViS is chasing — the draft here was trained with image tokens, so
its outputs would be degraded. Irrelevant for a timing study.

### 4. Q-Former
Minimal, sized for SmolVLM-256M — one cross-attention block, no BLIP-2 stack, no
self-attention among queries:

```
learned queries [N, 576]
  -> LN -> MHA(Q=queries, KV=image hidden states, 9 heads x 64) -> +residual
  -> LN -> MLP(576 -> 1536 -> 576) -> +residual
```

- `N = 832 / ratio` -> **208 / 104 / 52** queries for 4x / 8x / 16x
- ~2.6M params, ~100M MACs at N=208
- **Runs once, at prefill, on image positions only.** Decode steps read the compressed
  image KV; nothing re-runs. Where a later tensor needs shape alignment, fill with
  dummy values — only timing matters
- Compresses the concatenated `[832, 3H]` -> `[N, 3H]`
- Time the block with and without the MLP (~40% of its cost) as a free ablation

**Token embeddings are free here.** SmolVLM's image tokens all share one id (49190),
so their embeddings are already N copies of a single placeholder vector. Compressing
832 -> 208 is just 208 copies of that same vector — no learned embedding, no semantic
choice.

The point of this experiment: the Q-Former is **paid once at prefill but saves on all
128 draft steps**, cutting the draft's KV from ~880 to ~258 (4x) or ~102 (16x). The
question is whether the per-step attention saving repays it.

### 5. Eagle 3.1
`fc_norm: true` + `norm_output: true` — the only two fields separating
`smolvlm-256m-eagle3.json` from `smolvlm-256m-eagle3-3.1.json`.

Confirmed these **do** apply in `fused_fc` mode at 1 layer (`llama_eagle3.py:824`
builds one RMSNorm per aux stream and logs "EAGLE 3.1 enabled"). Note `fc_norm` is
*ignored* in `progressive_staged` without per-layer FC — not an issue here since
everything is 1-layer `fused_fc`.

---

## Deliverables

1. One table: ms/cycle and ms/drafted-token for every variant, delta vs #1
2. Per-component cost breakdown — what each addition costs on top of plain eagle3
3. Prune analysis for plain eagle3 itself: per-module profile of the 1-layer draft
   step. Expect the 32k-vocab `lm_head` matmul and the `d2t` remap to dominate

## Open / deferred

- Image-token counts for OmniDocBench not yet measured (only textvqa: 832/878 = 95%).
  #3 and #4 scale directly with this, so measure before interpreting them
- #3 and #4 land in torch first; port to vLLM only if the result justifies it
- COCO-Caption (908 in / 505 out) is the only image benchmark with more output than
  OmniDocBench, if a longer decode phase is ever wanted



I want to see, disregard of acceptancel length, at the same verification round number, whats the speed comparision. for each run, I wnat a report, telling me, during each drafting round,
  starting from prefilling stage, then first token, then second token, whats the time, and whats each component's time (standing out), averaged by all the examples.