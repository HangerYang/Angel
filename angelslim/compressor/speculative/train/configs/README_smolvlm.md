# SmolVLM draft configs

Target: `HuggingFaceTB/SmolVLM-256M-Instruct` (`target_model_type: smolvlm`).

Pass via:

```bash
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/<file>.json \
  OUTPUT_DIR=output/<run_name> \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

Ops / launch modes / eval: `scripts/speculative/smolvlm/README.md`.

| Config | Mode | Draft depth | Aux (train -> vLLM) | Notes |
|---|---|---|---|---|
| `smolvlm-256m-eagle3.json` | `fused_fc` (default) | 1 | `[1,14,26]` -> `[2,15,27]` | Stock Eagle3: `fc` 3H->H + 2H L0 |
| `smolvlm-256m-eagle3-3.1.json` | `fused_fc` | 1 | same | Stock + EAGLE 3.1 (`fc_norm` + `norm_output`) |
| `smolvlm-256m-eagle3-banded-mix-fc-3.1.json` | `banded_mix_fc` | 1 | 9 aux layers in 3 bands | EAGLE 3.1 with learned band mixes before the 3H->H FC |
| `smolvlm-256m-eagle3-banded-mix-wide-3.1.json` | `banded_mix_wide` | 1 | 9 aux layers in 3 bands | Same band mixes, no FC; layer 0 attends over embedding plus the band streams |
| `smolvlm-256m-eagle3-banded-mix-fc-3.1-branch-change-top1-w01.json` | `banded_mix_fc` | 1 | same | Branch-change distillation, top-1 branch, weight 0.1 |

`num_hidden_layers` controls draft depth. `eagle_aux_injection_mode` controls how
target hidden-state streams are fused.

## Branch distillation

EAGLE 3.1 branch configs use `banded_mix_fc` + `fc_norm` + `norm_output`.
When the draft's **top-1** is not the teacher's top-1 but sits inside the
teacher's top-k, the trainer forks one extra draft step onto that token. The
stored `target_logits` only cover the real sequence, so this costs a **second
teacher forward** per branched step:
every branch position is substituted into one copy of the sequence and scored
in a single pass, which means a position's prefix may contain an earlier
position's substitution — dense signal, slightly contaminated context.

| Key | Default | Meaning |
|---|---|---|
| `branch_distill_loss_weight` | `0.0` (off) | Weight on the branch loss, added to the total loss |
| `branch_distill_objective` | `"ce"` | `ce`: full post-branch teacher distribution; `change`: centered-logit delta MSE |
| `branch_distill_top_k` | `1` | Draft side. Must be 1 — the branch is the draft's top-1 only; any other value is rejected |
| `branch_distill_target_top_k` | `3` | Candidate branch only if the draft's top-1 is inside the teacher's top-k |
| `branch_distill_prob_ratio_threshold` | `0.0` | Optional plausibility gate: keep only if `p_T(draft_top1) / p_T(teacher_top1) > threshold` |
| `branch_distill_steps` | `1` | How many leading TTT steps get a branch; each is a full extra teacher forward |

Online trainers only — an offline one has no teacher in memory to re-score the
substituted sequence and raises. Common logs: `train/branch_loss`,
`train/branch_rate`, `train/branch_ratio_survival`. CE logs entropy/KL/mass;
change-MSE logs `train/branch_target_delta_rms` and
`train/branch_draft_delta_rms`.

Prepared runs:

- `branch-ratio-t01/t02/t03-top1-w01`: CE objective with plausibility ratio
  threshold `0.2`.
- `branch-change-top1-w01`: centered-logit delta-MSE objective, no ratio gate.

The older `*-branch-distill.json` / `*-branch-distill-w03.json` records live on
the preservation branch only.
