# answer_then_describe: does lengthening the output recover the speedup?

Generated 2026-08-20T19:46:24-07:00.

**Population**: N=80 prompts per benchmark, temp 0, `output_len=1024`,
`max_num_seqs=1`, K=4, `enforce_eager`, gpu-mem 0.8, one vLLM job at a time on GPU 0.

**Prompt styles**: `raw` (dataset question verbatim, image placeholders stripped as in
the raw harness) vs `answer_then_describe`
(`"Answer this question: {q} Then describe the image in detail to justify your answer."`).

**Arms**: `target-only` = no draft (`USE_EAGLE=0`); `with-draft` = EAGLE-3 speculative
decoding with the named draft. Note the two arms are **not** bit-identical at temp 0 in
this build — on long outputs the speculative path diverges mid-sequence (COCO-Caption
63/80 samples, chartqa 13/80; MMStar and MATH-500 0/80) — so a length taken from the
target-only arm is not interchangeable with one taken from a draft arm.

Benchmarks are the four whose raw output was too short for speculation to pay and that
`answer_then_describe` lengthened. chartqa is excluded: no prompt style moved it.

## Output length (target-only arm)

| benchmark | N | arm | raw out tok | ATD out tok | length ratio |
|---|---:|---|---:|---:|---:|
| MMStar | 80 | target-only | 8.8 | 87.0 | **9.85x** |
| MMMU | 80 | target-only | 45.0 | 297.7 | **6.61x** |
| textvqa | 80 | target-only | 17.8 | 114.1 | **6.40x** |
| mathvista | 80 | target-only | 11.4 | 90.3 | **7.92x** |

## Throughput and speedup (with-draft arm)

Speedup is against `no_eagle_baseline` measured under the *same* prompt style, so the
raw and ATD columns are each internally consistent.

| benchmark | N | draft | layers | raw tok/s | ATD tok/s | raw speedup | ATD speedup | ATD accept |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MMStar | 80 | `baseline_1layer` | 1 | 52.6 | 254.3 | 1.010x | **1.408x** | 2.373 |
| MMStar | 80 | `banded_mix_fc_3.1` | 1 | 51.1 | 279.5 | 0.981x | **1.548x** | 2.626 |
| MMStar | 80 | `banded_mix_wide_3.1` | 1 | 53.0 | 259.5 | 1.018x | **1.437x** | 2.525 |
| MMMU | 80 | `baseline_1layer` | 1 | 206.5 | 324.5 | 1.359x | **1.628x** | 2.484 |
| MMMU | 80 | `banded_mix_fc_3.1` | 1 | 214.2 | 332.4 | 1.410x | **1.667x** | 2.656 |
| MMMU | 80 | `banded_mix_wide_3.1` | 1 | 214.5 | 324.9 | 1.412x | **1.630x** | 2.637 |
| textvqa | 80 | `baseline_1layer` | 1 | 120.4 | 292.0 | 1.063x | **1.568x** | 2.515 |
| textvqa | 80 | `banded_mix_fc_3.1` | 1 | 121.9 | 295.3 | 1.077x | **1.586x** | 2.700 |
| textvqa | 80 | `banded_mix_wide_3.1` | 1 | 121.1 | 302.4 | 1.070x | **1.624x** | 2.752 |
| mathvista | 80 | `baseline_1layer` | 1 | 65.6 | 283.9 | 0.992x | **1.575x** | 2.531 |
| mathvista | 80 | `banded_mix_fc_3.1` | 1 | 65.6 | 268.5 | 0.993x | **1.489x** | 2.576 |
| mathvista | 80 | `banded_mix_wide_3.1` | 1 | 65.4 | 282.3 | 0.990x | **1.566x** | 2.742 |

## Baseline (target-only) throughput

| benchmark | N | raw tok/s | ATD tok/s |
|---|---:|---:|---:|
| MMStar | 80 | 52.1 | 180.6 |
| MMMU | 80 | 151.9 | 199.4 |
| textvqa | 80 | 113.2 | 186.2 |
| mathvista | 80 | 66.1 | 180.3 |

Produced by `scripts/speculative/smolvlm/run_atd_acceptance.sh`;
regenerate with `python my_angel/make_atd_results.py`.
