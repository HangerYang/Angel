# Odyssey results — `g8_t0_main` (raw prompts)

> **Superseded for utility conclusions by [`RESULTS_ATD.md`](RESULTS_ATD.md).**
> This sweep used raw prompts, where MATH-500 has zero target utility and
> MMStar answers are ~10 tokens — too short for post-rejection salvage to
> compound. The `answer_then_describe` sweep is the one to read for whether a
> branch preserves utility. What still stands here: the §3 over-salvage control
> (text-only, so it can only run on MATH-500) and the §5 finding that `block`
> is a no-op at temp 0.

**Setup.** SmolVLM-256M-Instruct target + `my_angel/eagle/baseline_1layer/checkpoint-66466`
(1-layer EAGLE3 draft). γ = 8, temp 0 (greedy), `max_num_seqs=1`, eager, single
H100. 40 prompts × 256 max tokens on MATH-500 (text-only, long generations) and
MMStar (image, short multiple-choice). Natural EOS. 3052 verification rounds on
the baseline branch.

Raw: `results/g8_t0_main/report.json`, `results/g8_t0_main/utility.json`.

---

## 1. Throughput and acceptance

| branch | MATH-500 accept_len | MATH-500 tok/s | MMStar accept_len | MMStar tok/s |
|---|---:|---:|---:|---:|
| `baseline` | 2.405 | 240.9 | 2.265 | 26.7 |
| `block` | 2.405 | 253.0 | 2.265 | 26.6 |
| `recompute` (online) | 2.405 | 245.0 | 2.265 | 26.6 |
| `stale_corr` | **4.375** | **417.8** | 2.575 | 25.2 |
| `stale_skip` | **6.229** | **567.2** | 3.090 | 51.9 |

Per-round telemetry (pooled):

| branch | mean τ | reject rate | mean salvage | total salvaged |
|---|---:|---:|---:|---:|
| `baseline` | 2.375 | 97.6% | 0.000 | 0 |
| `recompute` | 2.399 | 97.5% | 0.000 | 0 |
| `stale_corr` | 4.267 | 80.9% | 1.045 | 1773 |
| `stale_skip` | 5.776 | 53.2% | 1.276 | 1141 |

## 2. Utility is preserved. Token-identity is not.

**This is the section that answers "is it worth it".** Answers are extracted
(option letter for MMStar, `\boxed{}` / last number for MATH-500) and scored
against ground truth, and separately against the baseline's own answer.

### MMStar — the only benchmark here with measurable utility

| branch | accuracy | Δ vs baseline | **utility agreement** | no-answer | exact match | prefix agreement |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.150 | +0.000 | 1.000 | 0.000 | — | — |
| `block` | 0.150 | +0.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| `recompute` | 0.150 | +0.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| `stale_corr` | 0.150 | **+0.000** | **1.000** | 0.000 | 0.525 | 0.851 |
| `stale_skip` | 0.150 | **+0.000** | **0.975** | 0.025 | 0.000 | 0.399 |

The branches did fire on MMStar — `stale_corr` salvaged 89 tokens over 154
rejections (0.58/rejection), `stale_skip` 106 over 214. The generated *strings*
diverged heavily (`stale_skip` matches the baseline string on 0 of 40 prompts,
agreeing on only 40% of the character prefix). **The extracted answer was
unchanged on 100% / 97.5% of prompts, and accuracy was identical.**

So on this benchmark stale-reuse buys salvage at no utility cost. Token-level
drift and utility drift came apart completely.

### MATH-500 — no utility signal at all

| branch | accuracy | degenerate-loop rate | no-answer | utility agreement |
|---|---:|---:|---:|---:|
| `baseline` | **0.000** | 0.150 | 0.000 | 1.000 |
| `stale_corr` | 0.000 | 0.125 | 0.200 | 0.050 |
| `stale_skip` | 0.000 | 0.200 | 0.425 | 0.025 |

SmolVLM-256M scores **0/40 on MATH-500** and degenerates into repeated-line
loops on 15% of prompts. Sample baseline output:

> The point $(0,3)$ is in the polar coordinate $(r,0)$.\
> The point $(0,3)$ is in the rectangular coordinate $(0,3)$.\
> *(repeats)*

There is no utility to preserve or destroy here, so the throughput numbers in
§1 for MATH-500 — the largest speedups in the whole sweep — are measured on
degenerate text. The high acceptance is partly *because* the output is
repetitive and therefore easy to draft.

> **Correction to an earlier claim.** I previously wrote that stale-reuse
> "destroys the output", citing 0.000 exact match and 0.116 prefix agreement on
> MATH-500. Both of those are exact-match metrics — prefix agreement is longest
> common character prefix over baseline length, i.e. a graded exact match, not a
> semantic one. They measure token reproduction, not usefulness. On utility the
> claim does not hold: MMStar shows zero utility loss, and MATH-500 has no
> utility either way.

## 3. The stale tail is still mostly not real

The branch-3 control re-scored 1200 rejection events against fresh target
distributions (36/37 request prefixes reconstructed and validated).

| branch | mean stale salvage | mean **fresh** salvage | over-salvage rate | exact agreement |
|---|---:|---:|---:|---:|
| `baseline` | 0.000 | 0.416 | 0.0% | 81.0% |
| `stale_corr` | 1.018 | 0.404 | **37.5%** | 48.9% |
| `stale_skip` | 1.316 | 0.428 | **44.2%** | 42.5% |

`stale_corr` claims 1.02 salvaged tokens per rejection where a real recompute
salvages 0.40 — over-claiming ~2.5×, accepting tokens the target would have
rejected in 37.5% of events.

This is a *mechanical* result and stands independently of §2: the stale
distribution really is accepting tokens the target would not. What §2 adds is
that on MMStar those off-path tokens did not change the answer. Both can be
true — the model reaches the same conclusion by a different wording.

**Genuine headroom.** The `baseline` row says a fresh recompute would salvage
0.42 tokens per rejection that baseline currently discards, in 19.0% of
rejections. Collecting it honestly costs one extra target forward: 3.94 ms
against a 0.33 ms draft forward (measured `target_model_forward` /
`draft_model_forward` spans). Paying 3.94 ms for 0.42 tokens when the round
already yields 2.4 is a loss. Branch 3 stays a control, not a candidate.

## 4. Entropy gating buys nothing here

Over-salvage rate (stale accepted more than fresh would) vs. entropy at the
rejected position — the quantity a gate would have to predict:

| entropy bin (nats) | `stale_corr` n | over-salvage | `stale_skip` n | over-salvage |
|---|---:|---:|---:|---:|
| [0, 1) | 469 | 0.324 | 241 | 0.461 |
| [1, 2) | 323 | 0.440 | 202 | 0.431 |
| [2, 3) | 142 | 0.415 | 163 | 0.411 |
| [3, 4) | 37 | 0.378 | 97 | 0.464 |
| [4, 5) | 11 | 0.182 | 22 | 0.455 |

Flat between 32% and 46% across the range, no usable trend. Raw salvage yield
vs. entropy (`entropy_scatter` in the report) is likewise non-monotonic. **An
entropy threshold cannot separate safe from unsafe stale-reuse on this pair.**

Caveat: this gates on *token* divergence from the control. Given §2, the
question worth gating on may be utility divergence instead — which this sweep
cannot answer, because utility divergence was zero on the one benchmark that
had utility.

## 5. Block verification is a no-op at temp 0

`block` is byte-identical to `baseline` — same acceptance length to 3 decimals,
1.000 exact match, same accuracy. That is correct, not a bug:
`_rejection_kernel` guards it with `if USE_BLOCK_VERIFICATION and not
is_greedy`. Joint-suffix verification only differs when sampling stochastically.

**Branch 4 has not actually been tested.** It needs temp > 0.

## 6. What this sweep cannot tell you

- **A benchmark where the target is both useful and verbose.** This is the
  binding limitation. MATH-500 gives long generations but 0.000 accuracy;
  MMStar gives real (if weak) utility but ~10-token answers, so there is little
  room for salvage to compound. The utility-preservation result in §2 is
  therefore encouraging but under-powered — it is 40 prompts, and MMStar
  accuracy of 0.150 is below 4-way chance, so the accuracy column carries
  little signal on its own. The *agreement* column (1.000 / 0.975) is the
  stronger evidence.
- **Distributional exactness (TV / KL over repeated samples).** Needs temp > 0
  and many samples per prompt. Everything here is temp 0.
- **Branch 4 at temp > 0**, per §5.
- **γ = 16.** Not run. At 0.33 ms/draft-step, 16 sequential draft passes cost
  5.2 ms, exceeding one 3.94 ms target pass, so the round would need acceptance
  > ~4.3 to beat γ=4. Measured baseline acceptance is 2.4.

## 7. Reproduce

```bash
TAG=g8_t0_main GAMMA=8 TEMP=0 NUM_PROMPTS=40 OUTPUT_LEN=256 GPU=0 \
  bash Odyssey/scripts/run_branch_sweep.sh

PYTHONPATH=Odyssey python3 -m odyssey.utility \
  --results_root Odyssey/results/g8_t0_main \
  --out Odyssey/results/g8_t0_main/utility.json
```

Next runs, in priority order:

1. A long-generation benchmark the target can actually do (GSM8K, HumanEval, or
   a captioning/VQA set with longer answers) — without one, §2 stays
   under-powered.
2. The same sweep at `TEMP=1`, which is the only way to get §5 and the
   distributional check.
