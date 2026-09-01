# Where every eval result lives

Updated 2026-08-31, after the purge. Only the `answer_then_describe` (ATD) sweep
survives; every other eval tree was deleted.

## The one canonical table

**`my_angel/one-layer-results.md`** — regenerate with
`python my_angel/make_one_layer_results.py`. It reads only
`my_angel/eagle/<run>/rerun_atd/temp{0,1}/<bench>/acceptance_metrics.json` plus
`my_angel/no_eagle_baseline/atd_temp{0,1}/`. 11 draft runs + target-only,
8 benchmarks, temp 0 and 1 — **192/192 cells present**.

The one non-ATD number it still prints is the short-prompt output-token column of
the "why the long prompt" ratio table. That tree is gone, so those 8 values are
frozen as `RAW_BASELINE_OUT_TOK` in the generator.

### The duplication this replaced

Four committed versions of `one-layer-results.md` existed, each a different subset
of the *same* on-disk numbers — they agreed cell-for-cell where they overlapped,
only the row selection differed.

| version | rows | status |
|---|---|---|
| `main` / `baseline` / most feature branches (46f20c0e) | 4 | superseded |
| `backup/branch-distillation-pre-rebase` (7c495b9c) | 6 | superseded |
| `mess` (3dac8f0b) | 6 | superseded |
| `mess:show/one-layer-results.md` | byte-identical copy of `mess:my_angel/...` | duplicate |
| this branch, regenerated | **all 11** | canonical — merge this forward |

`feature/vistoken-attn-prune` had no copy at all; it forked off `baseline` before
46f20c0e, which is why the file looked missing.

## What is on disk now

| path | what |
|---|---|
| `my_angel/eagle/<run>/rerun_atd/temp{0,1}/<bench>/` | the ATD sweep, N=80, K=4, GPU0 serial — the only eval results kept |
| `my_angel/no_eagle_baseline/atd_temp{0,1}/` | target-only arm of the same sweep |
| `my_angel/eagle/<run>/checkpoint-*/` | model weights (32G), untouched |
| `output/{branch-diag-*,qsampler-n*}/` | training checkpoints, not evals |

Runs with ATD results: `baseline_1layer`, `banded_mix_fc_3.1`,
`banded_mix_wide_3.1`, `vistoken_k1`, `branch_distill_top1_w01`,
`branch_ratio_t02_top1_w01`, `branch_change_top1_w01` (the default),
`branch_change_top1_w03`, `branch_change_deltaweight_w01`,
`branch_change_top2_curr_r33k`, `branch_change_top2_curr_synth_r33k`.

Checkpoints with no eval: `branch-change-probe{A,B,C}-*-r33k`,
`vistoken-k1-attn-prune-x64-m16`.

## What was deleted (2026-08-31, ~22G)

| path | was |
|---|---|
| `my_angel/eagle/*/rerun/`, `my_angel/no_eagle_baseline/temp{0,1}/` | superseded short-`raw`-prompt sweep |
| `my_angel/eagle/*/ckpt_sweep/` | acceptance vs checkpoint |
| `my_angel/eagle/baseline_1layer/layer_swap{,_ofat_vllm}/` | 2026-08-15 aux-layer-selection study |
| `my_angel/eagle/baseline_1layer/hivis_cmp/` | HiViS comparison |
| `my_angel/eagle/branch-change-top2-curr-ws33k/` | empty shell, never evaluated |
| `my_angel/progressive_eagle/` | 18G multi-layer progressive-draft line of work |
| `my_angel/prompt_style_ab/` | 7 prompt styles, N=10, target-only |
| `my_angel/plan/` | draft-speed latency micro-benchmarks |
| `results/` | latency profiles, hawk/miracle oracle probes |
| `Odyssey/results/` | Odyssey branch, separate harness |
| `output/*/rerun_atd/` | empty stub dirs (one 0-byte log) |

Their `acceptance_metrics.json` / `results.json` / `*.sh` / `*.md` files (11M, 2046
files) were snapshotted first to
`~/tmp/angelslim-pre-purge-2026-08-31/metrics-and-scripts.tgz`.

Docs on the `mess` branch that reported the deleted trees — `RESULTS_rerun.md`,
`prompt-style-ab.md`, `progressive_eagle/TIMING.md`, `plan/plan.md` — now have no
backing data. `my_angel/atd-results.md` (prompt-style A/B, N=10) is likewise
unregenerable.
