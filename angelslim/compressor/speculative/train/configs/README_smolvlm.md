# SmolVLM draft configs

Target: `HuggingFaceTB/SmolVLM-256M-Instruct` (`target_model_type: smolvlm`).

Pass via:

```bash
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/<file>.json \
  OUTPUT_DIR=output/<run_name> \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

Ops / launch modes / eval: `scripts/speculative/smolvlm/README.md`.

| Config | Mode | Draft depth | Aux (train → vLLM) | Notes |
|---|---|---|---|---|
| `smolvlm-256m-eagle3.json` | `fused_fc` (default) | 1 | `[1,14,26]` → `[2,15,27]` | Stock Eagle3: `fc` 3H→H + 2H L0 |
| `smolvlm-256m-eagle3-3.1.json` | `fused_fc` | 1 | same | Stock + EAGLE 3.1 (`fc_norm` + `norm_output`) |
| `smolvlm-256m-eagle3-3.1-qk-norm.json` | `fused_fc` | 1 | same | 3.1 + **QK-norm**: per-head RMSNorm on Q/K pre-RoPE (`qk_norm: true`); adds `self_attn.q_norm/k_norm`, ones-init so step 0 matches 3.1 |
| `smolvlm-256m-eagle3-progressive.json` | `progressive_staged` | 3 | `[0,13,25]` → `[1,14,26]` | Per-layer 2H concat inject; needs progressive vLLM patch |
| `smolvlm-256m-eagle3-progressive-layers-1-15-23-3.1.json` | `progressive_staged` | 3 | `[1,15,23]` → `[2,16,24]` | Progressive 1/15/23 + EAGLE 3.1 `norm_output` (no `fc_norm`) |
| `smolvlm-256m-eagle3-progressive-uninit.json` | `progressive_staged` | 3 | same | Same as progressive, **no** `draft_layer_init_*` |
| `smolvlm-256m-hawk.json` | `hawk` | 3 | `[0,13,25]` → `[1,14,26]` | Progressive **H** fusion (`w1`/`w2` add); full draft layers trainable after target init; needs progressive vLLM patch |
| `smolvlm-256m-real-hawk.json` | `real_hawk` | 3 | `[0,13,25]` → `[1,14,26]` | **Real hawk**: same fuse; draft blocks = frozen target layers `[1,14,26]` + **LoRA**; train `fuse_w1/w2` + LoRA + head. Merge for vLLM hawk eval |
| `smolvlm-256m-layer-skip-lora.json` | `layer_skip_lora` | 3 | same | Back-compat **alias** of `real_hawk` |

`num_hidden_layers` ≠ progressive by itself — set `eagle_aux_injection_mode`.

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

The older `*-branch-distill.json` / `*-branch-distill-w03.json` are records of
earlier runs whose training code is not in this repo; they set
`branch_distill_top_k: 3` and will now be rejected.

## Visual row compression (`vistoken_compress`)

`smolvlm-256m-eagle3-vistoken-k{1,4,16}.json`, on top of `banded_mix_fc` 3.1.

The drafter's single attention layer receives every target aux stream at every
position, but at ~900 image rows it cannot route over them. This compresses each
tile's **64 image rows to k**, in target-aux-HS space, while leaving the depth
structure intact: one routing decision applied to all 9 aux streams, so the same
image region arrives at every depth the target read it at. Fewer rows, same
depth. Text rows and the `<row_i_col_j>` grid markers are untouched.

There is **no value projection** — the output rows are convex combinations of
real target hidden states, so they stay inside the distribution `fc_norm` / `fc`
expects. ~56k params at `k=1` (`d=576`, 32 tile slots).

Where it runs: `angelslim/compressor/vistoken/`, spliced in
`OnlineVLMEagle3Trainer.prepare_data_for_draft_model` and in
`tools/eval_smolvlm_eagle3_acceptance.py` — in both, immediately after the
target forward and **before** the left shift, while every row-aligned tensor is
still 1:1 on absolute positions.

| Key | Default | Meaning |
|---|---|---|
| `num_queries` | `1` | Compressed rows **per tile**. `k=1` on a 13–17 tile prompt gives 13–17 image rows total |
| `tile_tokens` | `64` | Rows per tile after Idefics3 pixel shuffle |
| `max_tiles` | `32` | Sizes the tile-index table; exceeding it raises |
| `key_dim` / `temperature` | `64` / `8.0` | Routing projection width and its softmax temperature |
| `routing` | `"shared"` | `shared`: one `w` for all aux streams (the depth-correspondence claim). `per_band`: one query set, one `k_proj` per band — isolates routing from query content |
| `query_mode` | `"learned"` | `mean`: uniform average over the tile's 64 rows, the null baseline — no routing params are used |
| `lr` | `1e-3` | The compressor's own optimizer group. The drafter keeps `LEARNING_RATE` |

Details worth knowing before reading a number off this:

- **Ordering.** Compression is a weighted sum over positions, the band mix a
  weighted sum over layers, `fc` linear — they commute, so the raw 9-stream aux
  concat is compressed first and `fc_norm` / band mix / `fc` run afterwards.
  The only nonlinearity still comes last.
- **The reference mix is its own parameter**, deliberately not the fc's band
  mix. One knob doing two jobs makes the routing ablations unreadable.
- **Positions are the target's original absolute positions, gaps included.**
  Renumbering would apply position-15 RoPE to a feature the target computed at
  position 900. A compressed row sits at the `w`-weighted mean of its sources,
  rounded — RoPE indexes a cached table, and a fraction of a position is far
  below its resolution at `rope_theta=1e5`.
- **The embedding half** of the drafter's input on a compressed row is one
  learned vector, shared across all k rows, carried as a **zero-initialised
  delta** on the `<image>` embedding — so an untrained compressor starts from
  exactly the vector the drafter already saw there. Compressed rows keep the
  `<image>` id, so the mask re-derives itself after every TTT shift.
- **Joint by default.** A drafter frozen at ~900 image rows is out of
  distribution at 15, and a bad number would not distinguish a failed thesis
  from a length mismatch. Frozen-drafter is a later ablation, not the short run.

Short run (does any k beat full?):

```bash
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-vistoken-k1.json \
  LEARNING_RATE=5e-5 EVAL_STEPS=250 NUM_TRAIN_EPOCHS=2 \
  OUTPUT_DIR=output/vistoken-k1 \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

Not yet ported to vLLM: `prepare_draft_config_for_vllm_eval.py` and the vLLM
bridge know nothing about `vistoken_compress`. Incremental decode is where a
TTT/decode mismatch would hide, so that port is worth doing only once the HF
acceptance number is worth having.
