# TODO — visual row compression (Idea 1), k=1

Status: **launched, trained, evaluated — a net loss.** Written 2026-08-20;
outcome recorded 2026-09-01 in **`my_angel/VISUAL-COMPRESSOR.md`**, which is now
the results record. Read that first; this file remains the design rationale.
Short version: `vistoken-k1` is `banded_mix_fc_3.1` + the compressor and lands at
tau 2.557 vs 2.706 (temp 0, 8-benchmark mean) — **-5.5%** — with the whole loss on
image benchmarks and MATH-500 at parity.

One cross-attention layer compresses each image tile's 64 target rows to k, in
target-aux-HS space, so the drafter's single attention layer routes over 13-17
image rows instead of ~900. Fewer rows, same depth: one routing decision is
applied to all 9 aux streams, so the same image region reaches the drafter at
every depth the target read it at.

## The run

Same recipe as `smolvlm-256m-eagle3-banded-mix-fc-3.1/checkpoint-66466`
(read back from its `training_args.bin`), **from scratch — no warm-start, no
baseline reuse**, plus the compressor trained jointly from step 0.

| | |
|---|---|
| Base | `banded_mix_fc` + EAGLE 3.1 (`fc_norm`, `norm_output`), 1 draft layer, 576d |
| Aux | 9 layers `[2,4,8,10,15,18,20,26,28]`, bands `[[2,4,8,10],[15,18,20],[26,28]]` |
| Data | `dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl`, 132,943 samples |
| Steps | bs 1, no accum, 4 GPUs, 2 epochs = **66,466 steps** |
| TTT | length 7 |
| LR | drafter **1e-4** (same as 66466), compressor 1e-3 (own param group), constant, warmup_ratio 0.05, wd 0 |
| Eval / save | per epoch (as 66466) |

**Exactly one delta from the 66466 recipe:** the compressor gets its own 1e-3
param group. Everything else is left at the value that run used.

- Drafter stays at **1e-4**. This run is from scratch, not a nudge of a trained
  drafter, so halving it would confound the compressor with an undertrained
  draft layer.
- Eval/save stay **per epoch**, as 66466 had them. The earlier "every 250 steps"
  plan was sized for a 30-45 min run: here it is 265 evals x 237s = **17.4h**,
  4.6x the ~3h45m training itself.
- DDP/nccl, no DeepSpeed, no FSDP; seed 42, max_grad_norm 1.0 -- all script
  defaults, all matching 66466's `training_args.bin`.

The draft config is byte-identical to the 66466 input config plus the one
`vistoken_compress` block (verified by diff; the saved `config.json` differs only
in keys HF writes at save time and a transformers-version rope rename).

```bash
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-vistoken-k1.json \
  SAVE_STRATEGY=epoch EVAL_STRATEGY=epoch \
  OUTPUT_DIR=output/vistoken-k1 \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

## Baseline to beat

`checkpoint-66466`, vLLM, temp 0, N=80, K=4 (`rerun/temp0/*/acceptance_metrics.json`):

| benchmark | tau |
|---|---:|
| MATH-500 | 3.5323 |
| OmniDocBench | 2.6535 |
| MMMU | 2.6303 |
| COCO-Caption | 2.4375 |
| MMStar | 2.2110 |
| chartqa | 2.0976 |
| textvqa | 2.0774 |
| mathvista | 1.7832 |

## The module

`angelslim/compressor/vistoken/row_compressor.py`. 56,457 params at k=1.

```
H        [B, T, 64, 9, 576]     T tiles, 9 aux streams, image rows only
Q        [k, 576]               learned queries, shared across tiles
tile_emb [32, 576]              one vector per tile index, zero-init
k_proj   [576, 64]              routing only
ref_mix  [9]                    softmax, the compressor's own parameter

H_ref = sum_s softmax(ref_mix)_s * H[:,:,:,s]
q     = k_proj(Q + tile_emb[t])
w     = softmax(q . k_proj(H_ref)^T / 8)
out   = einsum('btkn,btnsd->btksd', w, H)
```

No value projection: every output row is a convex combination of real target
hidden states and stays in `fc_norm`'s input space.

## Where it plugs in

`angelslim/compressor/vistoken/splice.py:39`, called from
`online_eagle3_trainer.py:157` and `tools/eval_smolvlm_eagle3_acceptance.py:139`
-- both immediately after the target forward and **before** the left shift,
while every row-aligned tensor is still 1:1 on absolute positions. One gather
rebuilds them all; inserted rows point at their tile's first image row, so ids
stay `<image>` and the attention mask stays 1 with no special-casing. Only the
hidden states, loss mask and positions are then overridden.

Grid markers `<row_i_col_j>` and all text rows are untouched.

## Decisions, and why

- **Compress raw aux HS first**, then `fc_norm` -> band mix -> `fc`. Position-sum
  and layer-sum commute; the only nonlinearity still comes last.
- **`ref_mix` is separate** from the fc's band-mix logits. One knob doing two
  jobs makes the routing ablations unreadable.
- **Original absolute positions, gaps kept.** Renumbering would apply
  position-15 RoPE to a feature the target computed at position 900. A
  compressed row sits at the `w`-weighted mean of its sources, rounded --
  `apply_rotary_pos_emb` (`model_utils.py:99`) indexes a cached table, and a
  fraction of a position is far below its resolution at rope_theta=1e5.
- **Embedding half**: one learned vector for all k rows, as a zero-init delta on
  the `<image>` embedding (`llama_eagle3.py:1484`), so an untrained compressor
  starts from the vector the drafter already saw there.
- **Joint, not frozen.** A drafter trained on ~900 image rows is out of
  distribution at 15; a bad number would not separate a failed thesis from a
  length mismatch. Frozen-drafter is a later ablation.

## Ablations (config flags, one each)

| flag | tests |
|---|---|
| `routing: per_band` (one query set, 3 `k_proj`s) | depth correspondence |
| `query_mode: mean` (uniform over the tile's 64) | whether learning beats averaging |
| `num_queries` 1 / 4 / 16 | expect the tau curve to peak, not saturate |

Configs already written: `smolvlm-256m-eagle3-vistoken-k{1,4,16}.json`.

## vLLM support (done)

`third_party/patches/20-vllm-v0.25.0-eagle3-vistoken.patch`, on top of the
progressive-staged patch. One file, `speculator.py`. It auto-arms whenever the
draft config carries `vistoken_compress`; `VLLM_EAGLE_VISTOKEN=0` disables.

It reuses the compact-prefill machinery HiViS already built
(`VLLM_EAGLE_HIVIS_ALL`, `llama_eagle3.py:885`), which sidesteps the KV problem
entirely: it never shrinks the cache. `compute_slot_mappings` is fed the kept
rows' **original positions**, so a surviving row at position 900 writes into the
slot for position 900 in the target's own block table
(`speculator.py:_maybe_set_l0_compact_prefill`); dropped rows are simply never
written. A private attention metadata + forward context covers the short query,
and the final hidden is scattered back to full width. No KV manager, no
scheduler, no proposer surgery.

What the patch adds:

1. `_maybe_compress_visual_rows` -- on the raw aux concat, before the band mix
   and fc (the order training uses), compress each tile and write its k
   summaries into that tile's fixed slots.
2. Records the keep mask (text rows + those slots) for the compact pass.
3. Applies the trained row-embedding delta on the surviving `<image>` rows.
4. `_maybe_load_vistoken` reads `vistoken.*` out of the checkpoint's
   `model.safetensors` and builds the module **by importing AngelSlim**, so
   training and decode run identical code and vLLM's weight loader is untouched.

`prepare_draft_config_for_vllm_eval.py` carries `vistoken_compress` through and
logs it. Eval runs exactly as the baseline did -- same
`eval_eagle3_vlm_batch.sh`, same N=80 / temp 0 / K=4 -- so the numbers are
directly comparable to the table above.

Constraints inherited from the HiViS path: **prefill only** (decode rows are all
drafted text) and **`max_num_seqs=1`**, which is what the eval harness already
uses.

One thing to check on the first real run: `seq_lens` stays the *full* prompt
length (`speculator.py:494`), so the draft's attention window still spans the
image-position slots that were never written this request. Look at what is in
them before trusting the number.

## Slot convention

The k summaries occupy fixed, evenly spaced rows **inside their own tile** --
k=1 -> row 32, k=4 -> rows 8/24/40/56 (`VisRowCompressor.slot_offsets`). Fixed
and data-independent because vLLM builds the draft's slot mapping from
`input_ids` alone, before the model runs, so the kept rows cannot depend on the
routing weights. A summary therefore keeps a real target position and the RoPE
angle computed at it -- the earlier weighted-mean-and-round scheme is gone.

## Tests

`tests/test_vistoken_splice.py`, `tests/test_vistoken_ttt.py`,
`tests/test_vistoken_vllm_parity.py` -- train and vLLM paths agree row-for-row
on both kept rows and values; text rows
byte-identical, text positions unchanged, `query_mode: mean` reproduces the
uniform tile average, outputs inside the tile's convex hull, one `w` across all
9 streams, gradients reach every parameter, `<image>` mask well-formed across 7
TTT shifts.
