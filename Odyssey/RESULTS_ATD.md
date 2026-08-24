# Odyssey results — `g8_t0_atd` (answer_then_describe)

**This supersedes `RESULTS.md` for any conclusion about utility.** The raw-prompt
sweep could not answer the question: MATH-500 had zero target utility, and
MMStar's ~10-token answers left no room for salvage to compound.

**Setup.** Same target/draft/γ/temp as `g8_t0_main`
(SmolVLM-256M + `baseline_1layer/checkpoint-66466`, γ=8, temp 0, `max_num_seqs=1`,
eager). `PROMPT_STYLE=answer_then_describe` — the question plus *"Then describe
the image in detail to justify your answer."* 40 prompts × 512 max tokens on
MMStar, textvqa, mathvista. ~1250-1475 verification rounds per benchmark.

ATD is the right regime for this experiment for two reasons: it lengthens output
~9× (MMStar 8.8 → 80.1 tokens) so post-rejection salvage has room to compound,
and it keeps the answer at the **front**, so answer extraction still works while
the long justification tail is exposed to drift.

Raw: `results/g8_t0_atd/utility.json`.

---

## 1. Utility

| bench | branch | accuracy | Δ | utility agreement | no-answer | **mean cosine** | cos p10 | frac < 0.8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MMStar | `baseline` | 0.175 | — | 1.000 | 0.150 | 1.000 | 1.000 | 0.000 |
| MMStar | `block` | 0.175 | +0.000 | 1.000 | 0.150 | 1.000 | 1.000 | 0.000 |
| MMStar | `recompute` | 0.175 | +0.000 | 1.000 | 0.150 | 1.000 | 1.000 | 0.000 |
| MMStar | `stale_corr` | 0.200 | **+0.025** | 0.900 | 0.150 | 0.758 | 0.428 | 0.475 |
| MMStar | `stale_skip` | 0.100 | **−0.075** | 0.525 | 0.550 | 0.450 | 0.262 | 0.950 |
| mathvista | `baseline` | 0.075 | — | 1.000 | 0.700 | 1.000 | 1.000 | 0.000 |
| mathvista | `stale_corr` | 0.075 | **+0.000** | 0.550 | 0.575 | 0.672 | 0.133 | 0.575 |
| mathvista | `stale_skip` | 0.025 | **−0.050** | 0.325 | 0.400 | 0.399 | 0.149 | 1.000 |
| textvqa | `baseline` | n/a | — | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| textvqa | `stale_corr` | n/a | n/a | 1.000 | 1.000 | 0.810 | 0.562 | 0.475 |
| textvqa | `stale_skip` | n/a | n/a | 1.000 | 1.000 | 0.362 | 0.017 | 0.950 |

textvqa accuracy is `n/a`: the extractor is multiple-choice and textvqa is
free-form, so `no_answer_rate` is 1.000 and the agreement column is vacuous
there. Only the embedding columns carry signal for textvqa.

## 2. Acceptance and output length

| branch | MMStar accept | out tok | mathvista accept | out tok | textvqa accept | out tok |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` / `block` / `recompute` | 2.373 | 80.1 | 2.554 | 80.2 | 2.424 | 89.2 |
| `stale_corr` | 2.600 | **39.0** | 4.236 | **130.1** | 4.704 | **137.5** |
| `stale_skip` | 2.984 | **24.5** | 5.519 | 130.4 | 5.897 | **239.2** |

**The branches destabilise generation length in both directions** — MMStar output
halves (80 → 39 → 24 tokens), textvqa nearly triples for `stale_skip`
(89 → 239). Truncation is why `stale_skip`'s MMStar `no_answer_rate` jumps to
0.550: it is emitting EOS early, not answering worse per-token. This also makes
the throughput comparison not like-for-like, so tok/s is omitted here.

## 3. Read

**`stale_corr` roughly holds accuracy but does not hold the output.**
Accuracy is +0.025 on MMStar and +0.000 on mathvista — within noise at n=40,
not a gain. But utility agreement falls to 0.900 / 0.550, and cosine to
0.758 / 0.672 with ~50% of prompts below 0.8. So it is flipping individual
answers in both directions and substantially rewriting the justification, while
the aggregate accuracy happens to land in the same place. That is a different
risk profile from "preserves utility", and the raw-MMStar run (agreement 1.000)
did not show it because there was almost no output to diverge.

**`stale_skip` is genuinely damaging.** −0.075 and −0.050 accuracy, agreement
0.525 / 0.325, cosine 0.450 / 0.399 / 0.362 with 95-100% of prompts below 0.8.
On textvqa its p10 cosine is 0.017 — effectively unrelated text. Discard it.

**`block` and `recompute` remain bit-identical to `baseline`** on every metric,
as expected (block is a no-op at temp 0; recompute is baseline online).

### The three metrics disagree, and that is the point

For `stale_corr` on MMStar:

| metric | value | what it says |
|---|---:|---|
| exact match (raw run) | 0.525 | half the strings differ |
| accuracy delta | +0.025 | no utility loss |
| utility agreement | 0.900 | 10% of answers flipped |
| mean cosine | 0.758 | the text moved a lot |
| frac cosine < 0.8 | 0.475 | ...on half the prompts |

Accuracy alone would have said "free win". Exact match alone would have said
"destroyed". The embedding + agreement pair is what shows the real shape: the
answer usually survives, the justification usually does not, and 10% of the time
the answer flips too.

## 4. Caveats

- **n=40 per benchmark**, and baseline accuracy is low in absolute terms
  (MMStar 0.175, mathvista 0.075). Accuracy deltas of ±0.05-0.075 are 2-3
  prompts. Treat the accuracy column as weak; the agreement and cosine columns
  are computed over the same 40 pairs but are far less quantised.
- **Wall-clock is contaminated.** A `train_eagle3_online.py` job of yours
  occupied all 4 GPUs partway through the sweep, so the `stale_skip`,
  `recompute`, and `stale_corr/mathvista` runs were re-run at
  `GPU_MEMORY_UTILIZATION=0.25` alongside it. Acceptance length, salvage,
  utility and cosine are unaffected by contention; throughput is, which is the
  second reason tok/s is omitted from §2.
- **No branch-3 control on these benchmarks.** The offline re-scorer is
  text-only by construction (see README) and all three ATD benchmarks are image
  benchmarks. The over-salvage numbers in `RESULTS.md` §3 stand, but they come
  from MATH-500 under raw prompting.
- **Embedding model** is `all-MiniLM-L6-v2`, 384-dim. Fine for relative
  comparison across branches; not a semantic-equivalence oracle.

## 5. Reproduce

```bash
TAG=g8_t0_atd GAMMA=8 TEMP=0 NUM_PROMPTS=40 OUTPUT_LEN=512 GPU=0 \
  PROMPT_STYLE=answer_then_describe \
  DATASETS="Lin-Chen/MMStar lmms-lab/textvqa ai4math/mathvista" \
  bash Odyssey/scripts/run_branch_sweep.sh

PYTHONPATH=Odyssey python3 -m odyssey.utility \
  --results_root Odyssey/results/g8_t0_atd \
  --out Odyssey/results/g8_t0_atd/utility.json --embed
```

Still open: `TEMP=1` (the only way to test `block` at all, and the only way to
get a distributional check), and a free-form answer extractor so textvqa
contributes more than cosine.
