# One-layer drafts — 8 benchmarks, GPU 0 sequential

Generated 2026-09-01T00:12:44-07:00.

Every 1-layer draft plus the non-speculative target, over 8 benchmarks.
One vLLM job at a time on GPU 0 with nothing else on the box, so tok/s is
comparable across every cell.

| | |
|---|---|
| population | N=80 prompts per benchmark |
| prompt | long prompt (`answer_then_describe`): question unchanged, model answers then justifies |
| temperature | 0 and 1 |
| arms | `no_eagle_baseline` = target-only; every other row = with-draft (EAGLE-3, K=4) |
| decode | `output_len=1024`, `max_num_seqs=1`, `enforce_eager`, gpu-mem 0.8, tp 1 |
| target | HuggingFaceTB/SmolVLM-256M-Instruct |
| drafts | `checkpoint-66466` of each run (1 draft layer) |

Produced by `scripts/speculative/smolvlm/run_atd_acceptance.sh`:

```bash
DATASETS_ATD="Lin-Chen/MMStar MMMU/MMMU opendatalab/OmniDocBench HuggingFaceH4/MATH-500 \
  lmms-lab/textvqa lmms-lab/chartqa ai4math/mathvista lmms-lab/COCO-Caption" \
  TEMP=0 bash scripts/speculative/smolvlm/run_atd_acceptance.sh   # then TEMP=1
python my_angel/make_one_layer_results.py
```

**Engine / how this differs from the default eval.** Same harness every arm uses:
`run_atd_acceptance.sh` -> `scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh`
-> `tools/vllm_offline_eagle3_vlm_batch.py` -> vLLM V1 offline `LLM(...)` from
`third_party/vllm` (v0.25.0) with `speculative_config={method: eagle3, ...}` and
`enforce_eager`. The wrapper overrides exactly three things, none of them the engine:

| | default suite | this sweep |
|---|---|---|
| prompt style | `raw` (`eval_acceptance_suite_dp.sh:114`) | `answer_then_describe` |
| benchmarks | 10 others: ChartQA, VQAv2, GQA, ScienceQA, MME, SEED-Bench, MMVet, MMBench, … (`:38-48`) | the 8 below |
| GPUs | round-robins one job per GPU | pinned `CUDA_VISIBLE_DEVICES=0`, serial, so tok/s is comparable |

`raw` left MMStar / textvqa / mathvista at 9-18 output tokens — too short for
speculation to pay — which is why this sweep, not the default one, is the record.

| run | what it is |
|---|---|
| `no_eagle_baseline` | target only, no speculative decoding |
| `baseline_1layer` | stock EAGLE-3: 3 aux layers, 3H→H fusion FC, 2H layer 0 |
| `banded_mix_fc_3.1` | 9 aux layers → 3 learned band mixes → 3H→H FC, EAGLE 3.1 |
| `banded_mix_wide_3.1` | same 3 band mixes, **no** FC: 4H layer 0 `[emb｜band0｜band1｜seed]` |
| `vistoken_k1` | `banded_mix_fc_3.1` + learned-query row compression: each tile's 64 visual rows → k=1 summary |
| `vistoken_k1_attn_prune_x64_m16` | `vistoken_k1` + target-attention row pruning ahead of the compressor: score each 64-row tile by the target's own q·k and keep the top M=16 rows |
| `branch_distill_top1_w01` | `banded_mix_fc_3.1` + branch-aware distillation on the full teacher distribution at the forked token (draft top-1 vs teacher top-3, w=0.1, 1 step) |
| `branch_ratio_t02_top1_w01` | branch fork gated by a teacher-probability *ratio* threshold (t=0.2) instead of top-3 membership |
| `branch_change_top1_w01` | `banded_mix_fc_3.1` + branch-change distillation: where the draft's top-1 is not the teacher's but sits in the teacher's top-3, fork one draft step onto that token and match the teacher's centered-logit *delta* (w=0.1, 1 step). **The default.** |
| `branch_change_top1_w01_steps2` | same, but the branch fires at two rollout steps (`branch_distill_steps=2`) |
| `branch_change_top1_w03` | same, w=0.3 |
| `branch_change_deltaweight_w01` | same, w=0.1, but the branch loss is weighted per token by the teacher delta magnitude |
| `branch_change_top2_curr_r33k` | top-2 fork with a curriculum, resumed from `checkpoint-33233` of `banded_mix_fc_3.1` |
| `branch_change_top2_curr_synth_r33k` | as above plus synthetic branch tokens |

## Prompt

Every benchmark uses the long prompt: the model answers first and then justifies,
so outputs are long enough for speculative decoding to pay. The question text
itself is unchanged. How each benchmark is treated:

| benchmark | treatment |
|---|---|
| MMStar | wrapper |
| MMMU | wrapper |
| OmniDocBench | prompt replaced (OCR) |
| MATH-500 | **unmodified** — reruns the raw prompt |
| textvqa | wrapper |
| chartqa | wrapper |
| mathvista | wrapper |
| COCO-Caption | prompt replaced (caption) |

## Results — temp 0

### Acceptance length

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_1layer` | 2.373 | 2.484 | 2.594 | 3.070 | 2.515 | 2.043 | 2.531 | 2.454 | **2.508** |
| `banded_mix_fc_3.1` | 2.626 | 2.656 | 2.809 | 3.532 | 2.700 | 2.214 | 2.576 | 2.535 | **2.706** |
| `banded_mix_wide_3.1` | 2.525 | 2.637 | 2.813 | 3.379 | 2.752 | 2.176 | 2.742 | 2.536 | **2.695** |
| `vistoken_k1` | 2.363 | 2.439 | 2.707 | 3.477 | 2.429 | 2.236 | 2.567 | 2.236 | **2.557** |
| `vistoken_k1_attn_prune_x64_m16` | 2.377 | 2.520 | 2.704 | 3.508 | 2.580 | 2.225 | 2.421 | 2.228 | **2.570** |
| `branch_distill_top1_w01` | 2.503 | 2.673 | 2.799 | 3.488 | 2.716 | 2.205 | 2.694 | 2.480 | **2.695** |
| `branch_ratio_t02_top1_w01` | 2.545 | 2.687 | 2.814 | 3.508 | 2.632 | 2.181 | 2.642 | 2.516 | **2.691** |
| `branch_change_top1_w01` | 2.590 | 2.867 | 2.977 | 3.605 | 2.774 | 2.217 | 2.939 | 2.551 | **2.815** |
| `branch_change_top1_w01_steps2` | 2.599 | 2.876 | 2.901 | 3.637 | 2.764 | 2.217 | 2.918 | 2.528 | **2.805** |
| `branch_change_top1_w03` | 2.557 | 2.848 | 2.914 | 3.570 | 2.757 | 2.213 | 2.906 | 2.544 | **2.789** |
| `branch_change_deltaweight_w01` | 2.575 | 2.831 | 2.876 | 3.626 | 2.699 | 2.216 | 2.788 | 2.543 | **2.769** |
| `branch_change_top2_curr_r33k` | 2.620 | 2.755 | 2.884 | 3.577 | 2.676 | 2.221 | 2.840 | 2.540 | **2.764** |
| `branch_change_top2_curr_synth_r33k` | 2.631 | 2.762 | 2.890 | 3.581 | 2.699 | 2.230 | 2.786 | 2.534 | **2.764** |

### Throughput (tok/s)

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `no_eagle_baseline` | 180.6 | 199.4 | 207.4 | 207.9 | 186.2 | 185.5 | 180.3 | 205.7 | **194.1** |
| `baseline_1layer` | 254.3 | 324.5 | 361.2 | 424.9 | 292.0 | 246.3 | 283.9 | 334.1 | **315.2** |
| `banded_mix_fc_3.1` | 279.5 | 332.4 | 370.9 | 475.2 | 295.3 | 257.4 | 268.5 | 335.2 | **326.8** |
| `banded_mix_wide_3.1` | 259.5 | 324.9 | 367.4 | 449.1 | 302.4 | 252.1 | 282.3 | 330.2 | **321.0** |
| `vistoken_k1` | 240.9 | 300.9 | 344.2 | 473.3 | 265.1 | 245.2 | 260.7 | 289.9 | **302.5** |
| `vistoken_k1_attn_prune_x64_m16` | 242.5 | 308.7 | 344.5 | 467.8 | 274.0 | 241.2 | 242.8 | 293.7 | **301.9** |
| `branch_distill_top1_w01` | 258.1 | 338.6 | 373.0 | 470.7 | 305.4 | 253.7 | 285.1 | 331.4 | **327.0** |
| `branch_ratio_t02_top1_w01` | 264.3 | 336.8 | 372.4 | 468.8 | 292.1 | 260.3 | 274.6 | 333.1 | **325.3** |
| `branch_change_top1_w01` | 266.7 | 359.6 | 391.1 | 497.3 | 307.1 | 261.2 | 299.2 | 335.1 | **339.7** |
| `branch_change_top1_w01_steps2` | 275.2 | 361.8 | 372.6 | 483.1 | 306.8 | 260.4 | 291.2 | 328.6 | **335.0** |
| `branch_change_top1_w03` | 270.4 | 359.6 | 377.5 | 493.2 | 307.7 | 254.4 | 303.0 | 335.9 | **337.7** |
| `branch_change_deltaweight_w01` | 270.4 | 345.8 | 371.3 | 475.9 | 304.0 | 261.8 | 276.7 | 338.7 | **330.6** |
| `branch_change_top2_curr_r33k` | 269.6 | 344.2 | 374.0 | 481.3 | 297.7 | 257.6 | 298.5 | 338.0 | **332.6** |
| `branch_change_top2_curr_synth_r33k` | 276.7 | 345.3 | 381.3 | 475.8 | 295.8 | 260.5 | 285.4 | 342.0 | **332.9** |

### Speedup vs `no_eagle_baseline`

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_1layer` | 1.408x | 1.628x | 1.742x | 2.044x | 1.568x | 1.328x | 1.575x | 1.624x | **1.615x** |
| `banded_mix_fc_3.1` | 1.548x | 1.667x | 1.789x | 2.286x | 1.586x | 1.388x | 1.489x | 1.629x | **1.673x** |
| `banded_mix_wide_3.1` | 1.437x | 1.630x | 1.772x | 2.160x | 1.624x | 1.359x | 1.566x | 1.605x | **1.644x** |
| `vistoken_k1` | 1.334x | 1.509x | 1.660x | 2.277x | 1.424x | 1.322x | 1.446x | 1.409x | **1.548x** |
| `vistoken_k1_attn_prune_x64_m16` | 1.343x | 1.548x | 1.662x | 2.250x | 1.472x | 1.301x | 1.347x | 1.428x | **1.544x** |
| `branch_distill_top1_w01` | 1.429x | 1.698x | 1.799x | 2.264x | 1.640x | 1.368x | 1.582x | 1.611x | **1.674x** |
| `branch_ratio_t02_top1_w01` | 1.463x | 1.689x | 1.796x | 2.255x | 1.569x | 1.404x | 1.523x | 1.619x | **1.665x** |
| `branch_change_top1_w01` | 1.477x | 1.804x | 1.886x | 2.392x | 1.650x | 1.408x | 1.660x | 1.629x | **1.738x** |
| `branch_change_top1_w01_steps2` | 1.524x | 1.814x | 1.797x | 2.324x | 1.648x | 1.404x | 1.615x | 1.597x | **1.715x** |
| `branch_change_top1_w03` | 1.497x | 1.804x | 1.821x | 2.372x | 1.653x | 1.372x | 1.681x | 1.633x | **1.729x** |
| `branch_change_deltaweight_w01` | 1.497x | 1.734x | 1.791x | 2.289x | 1.633x | 1.411x | 1.535x | 1.646x | **1.692x** |
| `branch_change_top2_curr_r33k` | 1.493x | 1.726x | 1.804x | 2.315x | 1.599x | 1.389x | 1.656x | 1.643x | **1.703x** |
| `branch_change_top2_curr_synth_r33k` | 1.532x | 1.732x | 1.839x | 2.289x | 1.589x | 1.405x | 1.583x | 1.662x | **1.704x** |

## Results — temp 1

### Acceptance length

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_1layer` | 1.478 | 1.335 | 1.302 | 1.490 | 1.379 | 1.245 | 1.496 | 1.424 | **1.394** |
| `banded_mix_fc_3.1` | 1.553 | 1.368 | 1.296 | 1.630 | 1.430 | 1.245 | 1.481 | 1.433 | **1.429** |
| `banded_mix_wide_3.1` | 1.554 | 1.356 | 1.308 | 1.594 | 1.390 | 1.240 | 1.562 | 1.429 | **1.429** |
| `vistoken_k1` | 1.504 | 1.313 | 1.263 | 1.583 | 1.353 | 1.259 | 1.457 | 1.364 | **1.387** |
| `vistoken_k1_attn_prune_x64_m16` | 1.519 | 1.322 | 1.267 | 1.615 | 1.352 | 1.234 | 1.502 | 1.374 | **1.398** |
| `branch_distill_top1_w01` | 1.474 | 1.430 | 1.355 | 1.579 | 1.403 | 1.247 | 1.552 | 1.431 | **1.434** |
| `branch_ratio_t02_top1_w01` | 1.527 | 1.385 | 1.309 | 1.596 | 1.401 | 1.227 | 1.598 | 1.422 | **1.433** |
| `branch_change_top1_w01` | 1.604 | 1.371 | 1.327 | 1.616 | 1.450 | 1.261 | 1.525 | 1.435 | **1.449** |
| `branch_change_top1_w01_steps2` | 1.564 | 1.356 | 1.348 | 1.640 | 1.419 | 1.241 | 1.514 | 1.432 | **1.439** |
| `branch_change_top1_w03` | 1.490 | 1.403 | 1.358 | 1.622 | 1.397 | 1.240 | 1.527 | 1.438 | **1.434** |
| `branch_change_deltaweight_w01` | 1.490 | 1.379 | 1.313 | 1.601 | 1.395 | 1.261 | 1.532 | 1.436 | **1.426** |
| `branch_change_top2_curr_r33k` | 1.534 | 1.378 | 1.401 | 1.633 | 1.413 | 1.265 | 1.507 | 1.437 | **1.446** |
| `branch_change_top2_curr_synth_r33k` | 1.488 | 1.371 | 1.335 | 1.581 | 1.395 | 1.232 | 1.537 | 1.443 | **1.422** |

### Throughput (tok/s)

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `no_eagle_baseline` | 171.5 | 197.3 | 200.8 | 208.2 | 195.0 | 189.5 | 172.4 | 205.1 | **192.5** |
| `baseline_1layer` | 163.7 | 178.7 | 176.8 | 209.5 | 174.8 | 160.0 | 179.6 | 196.3 | **179.9** |
| `banded_mix_fc_3.1` | 156.1 | 175.3 | 174.5 | 221.5 | 178.4 | 156.3 | 167.9 | 195.9 | **178.2** |
| `banded_mix_wide_3.1` | 156.6 | 173.8 | 170.5 | 210.4 | 167.2 | 155.2 | 178.3 | 192.1 | **175.5** |
| `vistoken_k1` | 137.9 | 156.9 | 157.9 | 214.8 | 160.8 | 145.0 | 165.2 | 174.9 | **164.2** |
| `vistoken_k1_attn_prune_x64_m16` | 133.3 | 160.6 | 156.8 | 222.0 | 160.6 | 143.3 | 158.3 | 179.2 | **164.3** |
| `branch_distill_top1_w01` | 151.3 | 184.3 | 179.6 | 216.8 | 173.8 | 158.3 | 168.8 | 191.1 | **178.0** |
| `branch_ratio_t02_top1_w01` | 166.3 | 175.9 | 171.1 | 221.4 | 173.0 | 152.3 | 182.0 | 193.5 | **179.4** |
| `branch_change_top1_w01` | 155.5 | 181.3 | 174.2 | 217.9 | 182.8 | 157.4 | 170.7 | 193.3 | **179.1** |
| `branch_change_top1_w01_steps2` | 166.6 | 161.7 | 176.4 | 222.9 | 173.0 | 155.9 | 174.3 | 188.0 | **177.3** |
| `branch_change_top1_w03` | 163.3 | 180.6 | 183.0 | 218.6 | 172.0 | 156.7 | 173.0 | 190.5 | **179.7** |
| `branch_change_deltaweight_w01` | 159.9 | 174.5 | 174.0 | 214.5 | 169.6 | 152.6 | 165.9 | 196.6 | **176.0** |
| `branch_change_top2_curr_r33k` | 163.8 | 171.4 | 183.9 | 218.4 | 171.9 | 161.1 | 164.9 | 191.2 | **178.3** |
| `branch_change_top2_curr_synth_r33k` | 164.8 | 179.6 | 176.4 | 213.4 | 169.3 | 156.2 | 168.9 | 193.4 | **177.7** |

### Speedup vs `no_eagle_baseline`

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_1layer` | 0.954x | 0.906x | 0.880x | 1.007x | 0.896x | 0.844x | 1.042x | 0.957x | **0.936x** |
| `banded_mix_fc_3.1` | 0.910x | 0.888x | 0.869x | 1.064x | 0.915x | 0.825x | 0.974x | 0.955x | **0.925x** |
| `banded_mix_wide_3.1` | 0.913x | 0.881x | 0.849x | 1.011x | 0.858x | 0.819x | 1.034x | 0.937x | **0.913x** |
| `vistoken_k1` | 0.804x | 0.795x | 0.786x | 1.032x | 0.825x | 0.765x | 0.958x | 0.853x | **0.852x** |
| `vistoken_k1_attn_prune_x64_m16` | 0.777x | 0.814x | 0.781x | 1.066x | 0.824x | 0.756x | 0.918x | 0.874x | **0.851x** |
| `branch_distill_top1_w01` | 0.882x | 0.934x | 0.894x | 1.041x | 0.891x | 0.836x | 0.979x | 0.932x | **0.924x** |
| `branch_ratio_t02_top1_w01` | 0.969x | 0.892x | 0.852x | 1.064x | 0.887x | 0.804x | 1.056x | 0.943x | **0.933x** |
| `branch_change_top1_w01` | 0.907x | 0.919x | 0.868x | 1.047x | 0.937x | 0.830x | 0.990x | 0.942x | **0.930x** |
| `branch_change_top1_w01_steps2` | 0.972x | 0.819x | 0.878x | 1.071x | 0.887x | 0.823x | 1.011x | 0.916x | **0.922x** |
| `branch_change_top1_w03` | 0.952x | 0.915x | 0.911x | 1.050x | 0.882x | 0.827x | 1.003x | 0.929x | **0.934x** |
| `branch_change_deltaweight_w01` | 0.932x | 0.884x | 0.866x | 1.030x | 0.870x | 0.806x | 0.962x | 0.959x | **0.914x** |
| `branch_change_top2_curr_r33k` | 0.955x | 0.869x | 0.915x | 1.049x | 0.882x | 0.850x | 0.956x | 0.932x | **0.926x** |
| `branch_change_top2_curr_synth_r33k` | 0.961x | 0.910x | 0.878x | 1.025x | 0.868x | 0.824x | 0.979x | 0.943x | **0.924x** |

## Prompt / output sizes

Target-only arm (`no_eagle_baseline`), long prompt, N=80. The with-draft arms
differ slightly on long outputs because the speculative path is not bit-exact
here; do not mix the two.

| dataset | N | temp | avg input tok | avg output tok |
|---|---:|---:|---:|---:|
| MMStar | 80 | 0 | 972.0 | 87.0 |
| MMMU | 80 | 0 | 1033.9 | 297.7 |
| OmniDocBench | 80 | 0 | 903.0 | 592.0 |
| MATH-500 | 80 | 0 | 71.9 | 566.6 |
| textvqa | 80 | 0 | 975.6 | 114.1 |
| chartqa | 80 | 0 | 998.2 | 124.5 |
| mathvista | 80 | 0 | 995.1 | 90.3 |
| COCO-Caption | 80 | 0 | 902.4 | 605.2 |
| MMStar | 80 | 1 | 972.0 | 65.9 |
| MMMU | 80 | 1 | 1033.9 | 171.7 |
| OmniDocBench | 80 | 1 | 903.0 | 281.6 |
| MATH-500 | 80 | 1 | 71.9 | 244.6 |
| textvqa | 80 | 1 | 975.6 | 158.8 |
| chartqa | 80 | 1 | 998.2 | 124.8 |
| mathvista | 80 | 1 | 995.1 | 74.2 |
| COCO-Caption | 80 | 1 | 902.4 | 590.4 |

### Why the long prompt (target-only, temp 0)

The short prompt this sweep replaced left several benchmarks at 9-45 output
tokens — too short for speculation to pay. Superseded numbers, kept as the
reason for the change:

| benchmark | short-prompt out tok | long-prompt out tok | ratio |
|---|---:|---:|---:|
| MMStar | 8.8 | 87.0 | **9.85x** |
| MMMU | 45.0 | 297.7 | **6.61x** |
| OmniDocBench | 416.3 | 592.0 | **1.42x** |
| MATH-500 | 566.6 | 566.6 | **1.00x** |
| textvqa | 17.8 | 114.1 | **6.40x** |
| chartqa | 88.4 | 124.5 | **1.41x** |
| mathvista | 11.4 | 90.3 | **7.92x** |
| COCO-Caption | 484.9 | 605.2 | **1.25x** |
