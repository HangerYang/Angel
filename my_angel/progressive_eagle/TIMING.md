# progressive_eagle — acceptance & throughput

`checkpoint-66466`, K=4, N=80 prompts, `output_len=1024`, `max_num_seqs=1`, eager.
Acceptance = `mean_acceptance_length`, tok/s = `output_throughput`, both read
straight from each `results.jsonl`. The mean-acceptance column reproduces the
published comparison doc exactly for all six runs it covers.

## Temp 0

Each cell: **acceptance length** · *tok/s*.  `—` in the tok/s half = timing excluded as unreliable; acceptance is still valid there.

| run | textvqa | MMMU | MMStar | OmniDocBench | MATH-500 | COCO-Caption | chartqa | mathvista | mean acc | mean tok/s† |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_staged` | **2.015** · *110* | **2.536** · *187* | **2.039** · *52* | **2.471** · *272* | **3.445** · *389* | **2.385** · *267* | **2.025** · *193* | **1.656** · *62* | **2.322** | *145* |
| `branch_distill_w01` | **1.943** · *126* | **2.493** · *186* | **2.045** · *51* | **2.536** · — | **3.541** · — | **2.381** · *270* | **2.111** · *198* | **1.802** · *62* | **2.356** | *149* |
| `attn_match_img_w01_L1_15_23` | **1.899** · *81* | **2.493** · *186* | **2.208** · *54* | **2.339** · — | **3.346** · — | **2.322** · *260* | **2.035** · *191* | **1.706** · *62* | **2.294** | *139* |
| `banded_mix_uninit` | **2.055** · *83* | **2.633** · *183* | **2.124** · *52* | **2.258** · — | **3.468** · — | **2.545** · *274* | **2.143** · *193* | **1.869** · *62* | **2.387** | *141* |
| `per_layer_fc` | **2.170** · *107* | **2.643** · *190* | **2.263** · *53* | **2.237** · — | **3.523** · *398* | **2.567** · *279* | **2.369** · *211* | **1.939** · *61* | **2.464** | *150* |
| `per_layer_fc_feedback` | **2.201** · *110* | **2.725** · *189* | **1.960** · *50* | **2.562** · *271* | **3.605** · *400* | **2.516** · *276* | **2.170** · *192* | **1.869** · *66* | **2.451** | *147* |
| `per_layer_fc_feedback_earlyexit` | **2.144** · *111* | **2.632** · *190* | **2.248** · *53* | **2.636** · *275* | **3.418** · *377* | **2.474** · *263* | **2.121** · *198* | **1.772** · *65* | **2.431** | *147* |

## Temp 1

Each cell: **acceptance length** · *tok/s*.  `—` in the tok/s half = timing excluded as unreliable; acceptance is still valid there.

| run | textvqa | MMMU | MMStar | OmniDocBench | MATH-500 | COCO-Caption | chartqa | mathvista | mean acc | mean tok/s† |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_staged` | **1.690** · *83* | **1.396** · *104* | **1.828** · *57* | — | **1.549** · *185* | **1.434** · *162* | **1.237** · *127* | **1.480** · *63* | **1.516** | *99* |
| `branch_distill_w01` | **1.733** · *82* | **1.455** · *109* | **1.685** · *69* | **1.161** · *127* | **1.568** · *181* | **1.448** · *168* | — | — | **1.509** | *107* |
| `attn_match_img_w01_L1_15_23` | **1.735** · *84* | **1.431** · *105* | **1.936** · *55* | **1.190** · *127* | **1.593** · *187* | **1.411** · *167* | — | — | **1.549** | *103* |


† Mean tok/s is over the six datasets every run has valid timing for (textvqa,
MMMU, MMStar, COCO-Caption, chartqa, mathvista), so the column is comparable
across rows. Mean acceptance uses all available datasets.

## What was blanked, and why

The tok/s half is blank on **OmniDocBench and MATH-500 for four runs**. Those cells
were ~3.5× faster than the same dataset in every other run, at identical output
lengths:

| dataset | blanked cells (Aug 17, 21:1x) | kept cells | ratio |
|---|---|---|---|
| OmniDocBench | 799, 806, 850, 909 tok/s | 271, 272, 275 | **3.0×** |
| MATH-500 | 1377, 1417, 1449 tok/s | 328–400 | **3.6×** |

They are excluded rather than the others because COCO-Caption — same ~500-token
output length — runs a steady **260–279 tok/s in every run across all three
sessions**, which puts the kept cells at the right order of magnitude and the
blanked ones far off it. All seven blanked cells come from one Aug-17 21:1x
session; the pattern matches unintended batching (`max_num_seqs` > 1), which
`results.jsonl` does not record. Acceptance from those cells is still valid and is
shown.

Temp 1 needed no blanking — every cell is within ±17% of its column median.

## Gaps

- Only three runs were ever evaluated at temp 1; the four newest have none.
- No run covers all 8 datasets at temp 1.
- `branch_distill_w01` temp 1 uses `eval_temp1_2ep` (matches `checkpoint-66466`);
  a 1-epoch `eval_temp1` also exists. `attn_match_img`'s temp-1 epoch is unverified.
- `baseline_staged`'s MMMU / MMStar / COCO-Caption temp-0 cells date from Aug 10,
  a week before the rest of its row.
- No non-speculative baseline exists anywhere, so no absolute speedup is computable.
