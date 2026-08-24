# Rerun results — temp 0, GPU 0 sequential

Generated 2026-08-20T19:46:18-07:00.

All cells: one vLLM job at a time on GPU 0, nothing else on the box — so
timing is comparable across runs.

| | |
|---|---|
| population | N=80 prompts per benchmark |
| prompt style | `raw` (dataset question verbatim; image placeholders stripped as in the raw harness) |
| temperature | 0 |
| arms | `no_eagle_baseline` = target-only; every other row = with-draft (EAGLE-3, K=4) |
| decode | `output_len=1024`, `max_num_seqs=1`, `enforce_eager`, tp 1 |
| target | HuggingFaceTB/SmolVLM-256M-Instruct |

Rows are grouped by **draft depth** (`num_hidden_layers`, read from each
run's checkpoint config). Depth is the main cost lever: a 3-layer draft
pays ~3x the per-step decode cost of a 1-layer one, so acceptance is only
comparable within a group — across groups, read the throughput table.

## Acceptance length

| run | MMStar | MMMU | OmniDocBench | MATH-500 | mean |
|---|---|---|---|---|---|
| **1-layer drafts** | | | | | |
| `baseline_1layer` | 2.048 | 2.358 | 2.234 | 3.070 | **2.427** |
| `banded_mix_fc_3.1` | 2.211 | 2.630 | 2.653 | 3.532 | **2.757** |
| `banded_mix_wide_3.1` | 2.208 | 2.625 | 2.730 | 3.379 | **2.736** |
| **3-layer drafts** | | | | | |
| `baseline_staged` | 2.039 | 2.536 | 2.471 | 3.445 | **2.623** |
| `branch_distill_w01` | 2.045 | 2.493 | 2.686 | 3.523 | **2.687** |
| `attn_match_img_w01_L1_15_23` | 2.208 | 2.493 | 2.498 | 3.404 | **2.651** |
| `banded_mix_uninit` | 2.124 | 2.633 | 2.599 | 3.504 | **2.715** |
| `per_layer_fc` | 2.263 | 2.643 | 2.516 | 3.523 | **2.736** |
| `per_layer_fc_feedback` | 1.954 | 2.725 | 2.562 | 3.604 | **2.711** |
| `per_layer_fc_feedback_earlyexit` | 2.248 | 2.632 | 2.636 | 3.418 | **2.734** |
| `banded_mix_per_layer_fc_feedback` | 2.270 | 2.734 | 2.680 | 3.560 | **2.811** |

## Throughput (tok/s)

| run | MMStar | MMMU | OmniDocBench | MATH-500 | mean |
|---|---|---|---|---|---|
| **1-layer drafts** | | | | | |
| `baseline_1layer` | 52.6 | 206.5 | 307.2 | 416.9 | **245.8** |
| `banded_mix_fc_3.1` | 51.1 | 214.2 | 339.4 | 477.5 | **270.5** |
| `banded_mix_wide_3.1` | 53.0 | 214.5 | 343.7 | 456.0 | **266.8** |
| **3-layer drafts** | | | | | |
| `baseline_staged` | 52.9 | 187.7 | 272.8 | 401.3 | **228.7** |
| `branch_distill_w01` | 51.8 | 189.6 | 295.6 | 411.2 | **237.1** |
| `attn_match_img_w01_L1_15_23` | 52.5 | 186.3 | 265.6 | 389.6 | **223.5** |
| `banded_mix_uninit` | 51.1 | 189.9 | 281.5 | 392.5 | **228.7** |
| `per_layer_fc` | 52.3 | 190.2 | 274.3 | 402.7 | **229.9** |
| `per_layer_fc_feedback` | 53.0 | 190.5 | 270.2 | 388.2 | **225.5** |
| `per_layer_fc_feedback_earlyexit` | 53.5 | 184.7 | 272.1 | 368.1 | **219.6** |
| `banded_mix_per_layer_fc_feedback` | 52.6 | 187.4 | 279.0 | 387.8 | **226.7** |

## Speedup vs non-speculative target-only baseline

| run | MMStar | MMMU | OmniDocBench | MATH-500 | mean |
|---|---|---|---|---|---|
| `no_eagle_baseline` (tok/s) | 52.1 | 151.9 | 205.8 | 210.0 | |
| **1-layer drafts** | | | | | |
| `baseline_1layer` | 1.010x | 1.359x | 1.493x | 1.985x | **1.462x** |
| `banded_mix_fc_3.1` | 0.981x | 1.410x | 1.649x | 2.274x | **1.579x** |
| `banded_mix_wide_3.1` | 1.018x | 1.412x | 1.670x | 2.171x | **1.568x** |
| **3-layer drafts** | | | | | |
| `baseline_staged` | 1.016x | 1.236x | 1.326x | 1.911x | **1.372x** |
| `branch_distill_w01` | 0.996x | 1.248x | 1.436x | 1.958x | **1.410x** |
| `attn_match_img_w01_L1_15_23` | 1.008x | 1.226x | 1.291x | 1.855x | **1.345x** |
| `banded_mix_uninit` | 0.981x | 1.250x | 1.368x | 1.869x | **1.367x** |
| `per_layer_fc` | 1.005x | 1.252x | 1.333x | 1.918x | **1.377x** |
| `per_layer_fc_feedback` | 1.017x | 1.254x | 1.313x | 1.849x | **1.358x** |
| `per_layer_fc_feedback_earlyexit` | 1.028x | 1.216x | 1.322x | 1.753x | **1.330x** |
| `banded_mix_per_layer_fc_feedback` | 1.010x | 1.234x | 1.356x | 1.846x | **1.361x** |

## Prompt / output sizes

N=80, temp 0. Taken from whichever run carries the cell; the speculative path is
not bit-exact against target-only in this build, so lengths can differ by ~1-3%
between arms on long outputs (chartqa 88.4 target-only vs 84.9-86.0 with a draft).

| dataset | N | avg input tok | avg output tok |
|---|---:|---|---|
| MMStar | 80 | 957.0 | 8.8 |
| MMMU | 80 | 1018.4 | 45.0 |
| OmniDocBench | 80 | 887.0 | 418.3 |
| MATH-500 | 80 | 71.9 | 566.6 |

## Extended benchmarks (runs that have them)

| run | textvqa | chartqa | mathvista | COCO-Caption |
|---|---|---|---|---|
| `baseline_1layer` accept | 2.028 | 2.012 | 1.672 | 2.331 |
| `baseline_1layer` tok/s | 120.4 | 228.0 | 65.6 | 323.7 |
| `banded_mix_per_layer_fc_feedback` accept | 2.197 | 2.274 | 1.802 | 2.604 |
| `banded_mix_per_layer_fc_feedback` tok/s | 111.2 | 192.5 | 59.9 | 274.7 |
| `banded_mix_fc_3.1` accept | 2.077 | 2.098 | 1.783 | 2.437 |
| `banded_mix_fc_3.1` tok/s | 121.9 | 235.3 | 65.6 | 324.3 |
| `banded_mix_wide_3.1` accept | 2.085 | 2.083 | 1.747 | 2.411 |
| `banded_mix_wide_3.1` tok/s | 121.1 | 227.8 | 65.4 | 314.4 |

| dataset | avg input tok | avg output tok |
|---|---|---|
| textvqa | 960.6 | 17.1 |
| chartqa | 983.2 | 84.9 |
| mathvista | 980.1 | 11.4 |
| COCO-Caption | 908.4 | 495.1 |

> `banded_mix_per_layer_fc_feedback` was evaluated in a separate session under `eval/plain/`, not in the sequential GPU-0 rerun, and used `gpu_memory_utilization 0.9` where the rerun used `0.8`. Every other flag matches (`max_num_seqs 1`, `output_len 1024`, K=4, `enforce_eager`, temp 0, N=80) and its `avg_input_tokens` agree exactly with the rerun's, so acceptance is directly comparable; treat its tok/s as indicative rather than head-to-head.

Cells found: 60 (11 runs x 4 core benchmarks = 44 expected, plus extended-benchmark cells).
