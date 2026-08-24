# TODO — branch distillation (top-1-in-teacher-top-3), w=0.1

Status: **trained and adopted as the default.** Written 2026-08-21, updated 2026-08-23.

`branch_change_top1_w01` is the best one-layer draft measured: acceptance
length 2.815 vs the branch-free `banded_mix_fc_3.1` baseline's 2.706 at temp 0,
and 1.750x end-to-end vs the target-only 1.000x (the baseline manages 1.684x).
Full numbers in [one-layer-results.md](one-layer-results.md); the config is
`angelslim/compressor/speculative/train/configs/`
`smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top1-w01.json`.

Settings that were tried and did NOT beat it, so they are deliberately absent
from this branch (see the experimental record on `feature/branch-distillation`):

- **w=0.3** instead of 0.1: worse. On a 5k paired probe the branch effect drops
  ~35% (dCE -0.0044 vs -0.0068). At w=0.1 the branch term already outweighs the
  main CE by ~27x per active position, so more weight overshoots.
- **`target_top_k=2`** instead of 3: no difference (0.0002 dCE, inside the
  +-0.0008 noise floor). Rank-2 catches 0.081 of positions against rank-3's
  0.119, but the extra rank-3 branches are less plausible under the teacher
  (mean prob ratio 0.340 vs 0.388), and volume and quality cancel.
- **Synthetic branches** (substituting the teacher's rank-2 token at otherwise
  non-branching positions): no gain. They share one denominator with the natural
  branches, so they dilute rather than add -- the per-position loss coefficient
  halves, ~13.0 to ~6.5, for the same weight.
- **A plausibility gate** (`branch_distill_prob_ratio_threshold`): 0.2 lands at
  2.691, below the 2.706 baseline.
- **A weight ramp / late start.** Deferring the branch loss to epoch 2 costs
  about half the benefit, but that is duration, not the curriculum: branch-on
  for 33k steps instead of 66k leaves the paired CE gap still widening when
  training ends.

When the draft's top-1 at a position is *not* the teacher's top-1 but sits
inside the teacher's top-k, fork one extra draft step onto that token. The
original version trained the fork against the teacher's full post-branch
distribution. The current version trains the *change* caused by the branch:

```text
regular path: Target(...mat) vs Draft(...mat)
branch path:  Target(...rug) vs Draft(...rug)

branch-change loss:
  [TargetLogits(...rug) - TargetLogits(...mat)]
       vs
  [DraftLogits(...rug)  - DraftLogits(...mat)]
```

Implemented as masked per-position MSE over draft-vocab logit deltas. The
regular EAGLE CE still teaches the base next-token distribution; the branch
term specifically teaches how the future prediction should move when the
previous token changes. It still costs a second teacher forward per training
step.

## Review (done 2026-08-21)

Read `_branch_decide` and `_branch_loss` (`eagle3_trainer.py:517`, `:576`) plus
their wiring at `:636`/`:796`, and `branch_teacher_logits`
(`online_eagle3_trainer.py:203`). All correct:

- `_branch_decide` picks positions where draft top-1 != teacher top-1 but is in
  teacher top-k, substitutes ALL branch tokens into one copy of the real
  sequence, one teacher re-forward scores all of them. Bounds-checked so a
  branch near the sequence end is dropped rather than reading past it.
- `_branch_loss` forks `cache_hidden` (real tensors, no in-place aliasing --
  see `_fork_cache`), runs one extra `encode_layers` on the branch token,
  restores `draft._last_layer_outs` afterward so it can't corrupt the main
  rollout's bookkeeping, and applies masked-mean MSE between draft and teacher
  post-branch logit deltas.
- Wiring: `_branch_decide` runs at the END of step `idx`; at the TOP of
  step `idx+1` the pre-normal rollout state is saved, the real step produces
  regular draft logits, then the fork is scored against `branch_logits -
  regular_logits` from that same saved state.
- `branch_teacher_logits`: a real second forward on the target model, same
  images/attention mask as the batch, only branch-position ids differ. Offline
  trainers hard-raise instead of silently no-op-ing (`eagle3_trainer.py:172`).
- `branch_distill_top_k` is enforced to 1 at init (`:106`) -- the branch is
  always the draft's own top-1, never a lower-ranked guess.

Test: `tests/test_branch_distill.py`. A synthetic teacher whose logits at
position p are a deterministic function of the token AT p, so any
position/token misalignment fails the assertion. Checks the branch mask, that
the scored target delta is exactly branch-teacher minus real-teacher at the
right absolute position, and that the forked step feeds the branch token only
at branch positions and leaves everything else untouched.

No correctness issues found. Nothing to fix before training.

## Two Runs

1. **Plausibility-ratio branch CE** (`branch-ratio-t01/t02/t03-top1-w01`):
   branch selection is still draft top-1 wrong-but-in-teacher-top-3, but now
   also requires `p_T(draft_token) / p_T(teacher_top1) > tau`. Queued tau value: `0.2`. Loss is the
   original post-branch CE.

2. **Branch-change delta MSE** (`branch-change-top1-w01`): same branch
   selection as the original top-3 gate, but trains
   centered `Draft(branch)-Draft(real)` against centered `Teacher(branch)-Teacher(real)` instead
   of relearning the full post-branch distribution.

## Config

`smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top1-w01.json` --
EAGLE 3.1 `banded_mix_fc`, 1 draft layer, `fc_norm` + `norm_output`, aux
bands `[2,4,8,10] / [15,18,20] / [26,28]`. Same family as
`smolvlm-256m-eagle3-banded-mix-fc-3.1`, which is the right 3.1 baseline, not
a warm-start.

```json
"branch_distill_loss_weight": 0.1,
"branch_distill_top_k": 1,
"branch_distill_target_top_k": 3,
"branch_distill_steps": 1
```

`branch_distill_steps: 1` means only the FIRST TTT substep can branch, so the
overhead is at most one extra teacher forward per training step, not per
branch position within it.

## The run

Same recipe as `progressive_eagle/baseline_staged/checkpoint-66466`, read back
from its `training_args.bin` -- **from scratch, no warm-start**:

| | |
|---|---|
| Base | `banded_mix_fc` EAGLE 3.1, 1 draft layer, 9 aux layers -> 3 learned band mixes |
| Data | `dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl`, 132,943 samples |
| Steps | bs 1, no accum, 4 GPUs, 2 epochs = **66,466 steps** (matches baseline_staged exactly) |
| LR | **1e-4 constant**, warmup_ratio 0.05, wd 0 (baseline_staged used constant, NOT the w01 stale run's cosine 5e-5) |
| Save | every 5000 steps, no eval (as baseline_staged had it) |

```bash
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top1-w01.json \
  SAVE_STRATEGY=steps SAVE_STEPS=5000 EVAL_STRATEGY=no \
  OUTPUT_DIR=my_angel/eagle/branch-change-top1-w01 \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

## Baseline to beat

`my_angel/eagle/smolvlm-256m-eagle3-banded-mix-fc-3.1/checkpoint-66466`, vLLM, temp 0, N=80, K=4
(`rerun/temp0/*/acceptance_metrics.json` -- only 4 benchmarks were rerun there):

| benchmark | tau |
|---|---:|
| MATH-500 | 3.4452 |
| MMMU | 2.5356 |
| OmniDocBench | 2.4712 |
| MMStar | 2.0390 |

## Do not confuse with: `my_angel/progressive_eagle/branch_distill_w01`

That directory is a **stale artifact from before this implementation
existed** -- its checkpoint config has `branch_distill_top_k: 3`, which the
current `_branch_decide` explicitly rejects at init (`eagle3_trainer.py:106`:
"branch_distill_top_k must be 1"). Its training code is not in this repo (see
`README_smolvlm.md`'s note on `*-branch-distill-w03.json`). Not a valid
warm-start, not a valid comparison point -- ignore its checkpoints and eval
numbers for this run.

## Open

None blocking. This is a pure training-loss addition to an existing EAGLE 3.1 `banded_mix_fc`
recipe -- no vLLM changes needed, eval uses the same harness and same τ table
format as everything else in this repo.
