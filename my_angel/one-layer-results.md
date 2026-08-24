# One-layer drafts — 8 benchmarks, GPU 0 sequential

Generated 2026-08-23T22:49:03-07:00.

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
| `branch_distill_top1_w01` | `banded_mix_fc_3.1` + branch-aware distillation (draft top-1 vs teacher top-3, w=0.1, 1 step) |
| `vistoken_k1` | `banded_mix_fc_3.1` + learned-query row compression: each tile's 64 visual rows → k=1 summary |
| `branch_change_top1_w01` | `banded_mix_fc_3.1` + branch-change distillation: where the draft's top-1 is not the teacher's but sits in the teacher's top-3, fork one draft step onto that token and match the teacher's centered-logit *delta* (w=0.1, 1 step). The default. |

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
| `branch_ratio_t02_top1_w01` | 2.545 | 2.687 | 2.814 | 3.508 | 2.632 | 2.181 | 2.642 | 2.516 | **2.691** |
| `branch_change_top1_w01` | 2.590 | **2.867** | **2.977** | **3.605** | **2.774** | 2.217 | **2.939** | **2.551** | **2.815** |
| *`branch_change_top2_curr_r33k`* | 2.620 | 2.755 | 2.884 | 3.577 | 2.676 | 2.221 | 2.840 | 2.540 | **2.764** |
| *`branch_change_top2_curr_synth_r33k`* | **2.631** | 2.762 | 2.890 | 3.581 | 2.699 | **2.230** | 2.786 | 2.534 | **2.764** |

### Throughput (tok/s)

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `no_eagle_baseline` | 180.6 | 199.4 | 207.4 | 207.9 | 186.2 | 185.5 | 180.3 | 205.7 | **194.1** |
| `baseline_1layer` | 254.3 | 324.5 | 361.2 | 424.9 | 292.0 | 246.3 | 283.9 | 334.1 | **315.2** |
| `banded_mix_fc_3.1` | **279.5** | 332.4 | 370.9 | 475.2 | 295.3 | 257.4 | 268.5 | 335.2 | **326.8** |
| `branch_ratio_t02_top1_w01` | 264.3 | 336.8 | 372.4 | 468.8 | 292.1 | 260.3 | 274.6 | 333.1 | **325.3** |
| `branch_change_top1_w01` | 266.7 | **359.6** | **391.1** | **497.3** | **307.1** | **261.2** | **299.2** | 335.1 | **339.7** |
| *`branch_change_top2_curr_r33k`* | 269.6 | 344.2 | 374.0 | 481.3 | 297.7 | 257.6 | 298.5 | 338.0 | **332.6** |
| *`branch_change_top2_curr_synth_r33k`* | 276.7 | 345.3 | 381.3 | 475.8 | 295.8 | 260.5 | 285.4 | **342.0** | **332.9** |

### Speedup vs `no_eagle_baseline`

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_1layer` | 1.408x | 1.628x | 1.742x | 2.044x | 1.568x | 1.328x | 1.575x | 1.624x | **1.615x** |
| `banded_mix_fc_3.1` | **1.548x** | 1.667x | 1.789x | 2.286x | 1.586x | 1.388x | 1.489x | 1.629x | **1.673x** |
| `branch_ratio_t02_top1_w01` | 1.463x | 1.689x | 1.796x | 2.255x | 1.569x | 1.404x | 1.523x | 1.619x | **1.665x** |
| `branch_change_top1_w01` | 1.477x | **1.804x** | **1.886x** | **2.392x** | **1.650x** | **1.408x** | **1.660x** | 1.629x | **1.738x** |
| *`branch_change_top2_curr_r33k`* | 1.493x | 1.726x | 1.804x | 2.315x | 1.599x | 1.389x | 1.656x | 1.643x | **1.703x** |
| *`branch_change_top2_curr_synth_r33k`* | 1.532x | 1.732x | 1.839x | 2.289x | 1.589x | 1.405x | 1.583x | **1.662x** | **1.704x** |

## Results — temp 1

### Acceptance length

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_1layer` | 1.478 | 1.335 | 1.302 | 1.490 | 1.379 | 1.245 | 1.496 | 1.424 | **1.394** |
| `banded_mix_fc_3.1` | 1.553 | 1.368 | 1.296 | 1.630 | 1.430 | 1.245 | 1.481 | 1.433 | **1.429** |
| `branch_ratio_t02_top1_w01` | 1.527 | **1.385** | 1.309 | 1.596 | 1.401 | 1.227 | **1.598** | 1.422 | **1.433** |
| `branch_change_top1_w01` | **1.604** | 1.371 | 1.327 | 1.616 | **1.450** | 1.261 | 1.525 | 1.435 | **1.449** |
| *`branch_change_top2_curr_r33k`* | 1.534 | 1.378 | **1.401** | **1.633** | 1.413 | **1.265** | 1.507 | 1.437 | **1.446** |
| *`branch_change_top2_curr_synth_r33k`* | 1.488 | 1.371 | 1.335 | 1.581 | 1.395 | 1.232 | 1.537 | **1.443** | **1.422** |

### Throughput (tok/s)

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `no_eagle_baseline` | **171.5** | **197.3** | **200.8** | 208.2 | **195.0** | **189.5** | 172.4 | **205.1** | **192.5** |
| `baseline_1layer` | 163.7 | 178.7 | 176.8 | 209.5 | 174.8 | 160.0 | 179.6 | 196.3 | **179.9** |
| `banded_mix_fc_3.1` | 156.1 | 175.3 | 174.5 | **221.5** | 178.4 | 156.3 | 167.9 | 195.9 | **178.2** |
| `branch_ratio_t02_top1_w01` | 166.3 | 175.9 | 171.1 | 221.4 | 173.0 | 152.3 | **182.0** | 193.5 | **179.4** |
| `branch_change_top1_w01` | 155.5 | 181.3 | 174.2 | 217.9 | 182.8 | 157.4 | 170.7 | 193.3 | **179.1** |
| *`branch_change_top2_curr_r33k`* | 163.8 | 171.4 | 183.9 | 218.4 | 171.9 | 161.1 | 164.9 | 191.2 | **178.3** |
| *`branch_change_top2_curr_synth_r33k`* | 164.8 | 179.6 | 176.4 | 213.4 | 169.3 | 156.2 | 168.9 | 193.4 | **177.7** |

### Speedup vs `no_eagle_baseline`

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_1layer` | 0.954x | 0.906x | 0.880x | 1.007x | 0.896x | 0.844x | 1.042x | **0.957x** | **0.936x** |
| `banded_mix_fc_3.1` | 0.910x | 0.888x | 0.869x | **1.064x** | 0.915x | 0.825x | 0.974x | 0.955x | **0.925x** |
| `branch_ratio_t02_top1_w01` | **0.969x** | 0.892x | 0.852x | **1.064x** | 0.887x | 0.804x | **1.056x** | 0.943x | **0.933x** |
| `branch_change_top1_w01` | 0.907x | **0.919x** | 0.868x | 1.047x | **0.937x** | 0.830x | 0.990x | 0.942x | **0.930x** |
| *`branch_change_top2_curr_r33k`* | 0.955x | 0.869x | **0.915x** | 1.049x | 0.882x | **0.850x** | 0.956x | 0.932x | **0.926x** |
| *`branch_change_top2_curr_synth_r33k`* | 0.961x | 0.910x | 0.878x | 1.025x | 0.868x | 0.824x | 0.979x | 0.943x | **0.924x** |

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
