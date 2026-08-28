# Official HiViS Qwen2.5-VL-3B Baseline

Ports the same two-stage HiViS draft-training recipe already validated for
SmolVLM-256M (see `scripts/speculative/smolvlm/README_official_hivis_vispec.md`)
to Qwen2.5-VL-3B-Instruct, training a fresh hivis-style draft from scratch
rather than relying on the externally-provided `ViSpec-Qwen2.5-VL-3B-Instruct`
checkpoint used by the tree/linear eval work in
`scripts/speculative/qwen2_5_vl/run_angelslim_hivis_qwen3b_*.sh` (those live
in the standalone `/home/hyang/HiViS` checkout's `scripts/` dir, not here).

A ViSpec-style variant (same training code, different data-gen prompt) is
planned as a follow-up — not built yet.

## Dataset: nothing new to set up

This reuses the exact same raw dataset already used for the SmolVLM runs —
no separate download or preprocessing needed:

```
dataset/preprocessed/mixed_sharegpt_llava665k_70k70k_b64.jsonl   (~17 GB)
```

Mixed ShareGPT text + LLaVA-665k multimodal records, format:
`{"id", "conversations":[{"role", "content":[{"type":"text"/"image", ...}]}]}`
— image entries carry either an absolute file path or a base64
`data:image/...;base64,...` URI. This raw format is target-model-agnostic;
what's target-specific and has to be freshly generated per model is the
hidden-state `.ckpt` cache (the `generate` stage below runs the raw text
through the actual Qwen2.5-VL-3B target model to capture that).

`hivis/ge_data/ge_data_qwen.py` was rewritten to read this format (it
previously expected two separate files in the old `{"from","value"}`
ShareGPT schema — `eval_data/sharegpt/sharegpt.jsonl` +
`eval_data/llava_v1_5_mix665k/llava_v1_5_mix665k_long_context.jsonl` — which
don't exist in this checkout; only their `prepare_data.py` generators do).

### text vs. multimodal split: real, one pass, not duplicated

Unlike `ge_data_smolvlm.py` (which writes the same full mixed pass into both
`--text-data-dir` and `--multimodal-data-dir` — `main_mix.py`'s `list_files()`
just concatenates whatever files exist in each directory into one training
pool, it doesn't care whether the split is real), `ge_data_qwen.py` loads the
target model and the mixed file **once** and routes each record to `--outdir`
(text) or `--multimodal-outdir` (has an image) based on that record's own
content. `sharegpt/` and `llava_v1_5_mix665k/` end up a genuine disjoint
partition of the one mixed file, at the cost of one model load and one pass
over the data — not two of each, which is what an earlier version of this
script did (see "Bugs fixed" below).

## Draft config

New: `hivis/train/qwen2_5_vl_3B_config.json` — same shape as the existing
`qwen2.5_vl_7B_config.json` (1-layer EAGLE-style draft head, `qwen2_5_vl`
architecture tag) but resized to the 3B target's actual dimensions
(`hidden_size=2048`, `intermediate_size=11008`, `num_key_value_heads=2`,
`vocab_size=151936`, `tie_word_embeddings=true` — pulled directly from
`Qwen2.5-VL-3B-Instruct/config.json`). The 7B config would silently train a
mis-sized draft head against a 3B target base model.

`hivis/ge_data/model_names.py`'s Qwen2.5-VL branch hardcoded
`hidden_size == 3584` (7B only) for output-directory naming; extended it to
map `2048 -> "3b"` alongside the existing `3584 -> "7b"`.

## Quick start

```bash
cd /home/hyang/Angel

# Full run: generate -> stage1 (2 epochs) -> stage2 (1 epoch), 8 GPUs
GPUS="0 1 2 3 4 5 6 7" bash scripts/speculative/qwen2_5_vl/run_official_hivis_qwen3b_2ep_1ep.sh
```

Output lands under `output/hivis_official/qwen25vl_3b_stage1_2ep_stage2_1ep/{stage1,stage2}`,
checkpoints as `state_<epoch>/` (accelerate `save_state` format). Stage 2
picks up stage 1's `state_0` by default (`STAGE1_CKPT`).

### Validate without launching anything

```bash
DRY_RUN=1 bash scripts/speculative/qwen2_5_vl/run_official_hivis_qwen3b_2ep_1ep.sh
```

### Running stages individually

```bash
STAGE=generate  bash scripts/speculative/qwen2_5_vl/train_official_hivis_qwen3b.sh   # one pass, both outputs
STAGE=stage1    bash scripts/speculative/qwen2_5_vl/train_official_hivis_qwen3b.sh
STAGE=stage2    bash scripts/speculative/qwen2_5_vl/train_official_hivis_qwen3b.sh
```

(No more `generate_text`/`generate_mm` — one `generate` pass produces both
now; see "Bugs fixed".)

Useful env var overrides (all have defaults, see the script header):
`GPUS`, `TARGET_MODEL_NAME_OR_PATH` (default: the local
`/home/hyang/HiViS/models/Qwen2.5-VL-3B-Instruct` checkpoint), `DATA_FILE`,
`START`/`END` (row-range slice, pre-shuffle — `END` gets clamped to the data
file's real row count regardless of what you pass, see "Bugs fixed"),
`MODEL_MAX_LENGTH`, `BS_STAGE1`/`BS_STAGE2`, `LR`, `STAGE1_EPOCHS`,
`STAGE2_EPOCHS`, `FORWARD_NUM_TOTAL`, `TOPK`, `TOPK_W`, `FAIL_FAST`,
`HIVIS_CONDA_ENV` (path to prepend to `PATH`; only applied if it exists on
this machine, set to `""` to always skip it — see "Bugs fixed").

### Smoke test

```bash
GPUS="0" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DATA_FILE=/home/hyang/Angel/dataset/preprocessed/mixed_sharegpt_llava665k_70k70k_b64.jsonl \
HIVIS_DATA_ROOT=/home/hyang/Angel/dataset/hivis_qwen25vl_3b_generated_smoke \
OUTPUT_ROOT=/home/hyang/Angel/output/hivis_official/qwen25vl_3b_smoke \
STAGE1_DIR=/home/hyang/Angel/output/hivis_official/qwen25vl_3b_smoke/stage1 \
STAGE2_DIR=/home/hyang/Angel/output/hivis_official/qwen25vl_3b_smoke/stage2 \
STAGE1_CKPT=/home/hyang/Angel/output/hivis_official/qwen25vl_3b_smoke/stage1/state_0 \
START=0 END=40 STAGE1_EPOCHS=1 STAGE2_EPOCHS=1 BS_STAGE1=1 BS_STAGE2=1 \
STAGE=all bash scripts/speculative/qwen2_5_vl/train_official_hivis_qwen3b.sh
```

Run and passed end-to-end (generate -> stage1 -> stage2) on 2026-08-28, one GPU,
40 samples, after fixing the four bugs below. Note `BS_STAGE1=1`: the default
`BS_STAGE1=4`/`BS_STAGE2=2` OOM'd on a single 24GB GPU with `--max-len 4096`
and this draft's vocab-sized (151936) lm_head -- the 8-GPU full run uses DDP
(each GPU trains its own full replica, no model/vocab sharding), so it hits
the same per-GPU memory ceiling regardless of GPU *count*. Lower `BS_STAGE1`/
`BS_STAGE2` (or `MODEL_MAX_LENGTH`) if the full run OOMs too.

## Bugs fixed to make this runnable (2026-08-28)

1. **`model_names.py` Qwen2.5-VL branch only accepted `hidden_size == 3584`**
   (raised `ValueError` for anything else, including the 3B's 2048) both in
   the config-based path and the model_reference is a string branch
   (`_name_from_reference` unconditionally returned `"qwen25vl_7b"` for any
   name containing "qwen2.5-vl"). Fixed to map `{2048: "3b", 3584: "7b"}` /
   detect "3b" vs "7b" in the reference string.
2. **`ge_data_qwen.py` targeted a dataset that doesn't exist in this
   checkout.** Its defaults were two separate old-schema files
   (`eval_data/sharegpt/sharegpt.jsonl`,
   `eval_data/llava_v1_5_mix665k/llava_v1_5_mix665k_long_context.jsonl`,
   `{"from","value"}` conversation format with `<image>` placeholders and a
   separate `--image-root`) — neither file exists, only their
   `prepare_data.py` generators do. Rewrote it to read the same mixed
   role/content format `ge_data_smolvlm.py` already handles (base64 or path
   images, explicit `Features` schema for the same reason documented in the
   SmolVLM README: text-only and multimodal records have different `content`
   shapes, and HF `datasets`' JSON loader infers the schema from the first
   chunk it reads), while keeping the Qwen-specific hidden-state generation
   logic (`Qwen2_5_VLForConditionalGeneration` loading, vision-span removal
   via `<|vision_start|>`/`<|image_pad|>` token IDs, `<|im_start|>`/`<|im_end|>`
   chat tokens, `position_ids` output required by `main_mix.py`'s
   `use_qwen_position_ids` path).
3. **Avoided reproducing a latent SmolVLM-pipeline bug.**
   `train_official_hivis_smolvlm.sh`'s `run_stage2()` points
   `--text-data-dir` at `"$TEXT_CKPT_DIR/non_code"`, but
   `ge_data_smolvlm.py`'s `generate` step never creates a `non_code`
   subdirectory (no code/non-code split anywhere in that script) — so stage 2
   would raise `FileNotFoundError` the first time it actually runs. Not fixed
   there (out of scope, and that queued run never got far enough to hit it —
   see the SmolVLM training queue script, which was still waiting on an
   unrelated 8-GPU job when this was written). For Qwen2.5-VL-3B, stage 1 and
   stage 2 both point `--text-data-dir` at the same `$TEXT_CKPT_DIR` here, no
   `/non_code` suffix — `ge_data_qwen.py` has no code/non-code concept either
   (the shared dataset has no `is_code` field), and `main_mix.py`'s
   `list_files()` treats the text/multimodal label as cosmetic (log text
   only) regardless, so there's no functional reason to introduce a split
   that doesn't exist upstream of it.
4. **The launcher relied on bare `python`/`accelerate` from `$PATH`.**
   `train_official_hivis_smolvlm.sh` does the same — presumably relying on
   `conda activate hivis` (or similar) already being active in the caller's
   shell. Running non-interactively (e.g. via `nohup ... &` from a shell that
   hadn't activated any env) resolved `python` to the base conda env, which
   is missing `torchvision` (`ge_data_qwen.py`'s `AutoProcessor.from_pretrained`
   needs it) and crashed immediately. Fixed by prepending
   `/home/hyang/anaconda3/envs/hivis/bin` to `$PATH` inside
   `train_official_hivis_qwen3b.sh` itself (override via `HIVIS_CONDA_ENV`),
   so it's correct regardless of the caller's active env.
5. **`main_mix.py`/`main_mix_topk_dyn_res.py`'s lm_head extraction only
   tried `language_model.lm_head.weight`/`lm_head.weight`, with no
   tied-embeddings fallback.** Qwen2.5-VL-3B has `tie_word_embeddings: true`
   (unlike the 7B), so its checkpoint has no separate `lm_head.weight` key at
   all — only `model.embed_tokens.weight` — and both scripts raised
   `KeyError: Could not find lm_head weight`. Fixed: when the lm_head
   candidates aren't found and `config.tie_word_embeddings` is true, fall
   back to loading the embedding table instead (same tensor, tied output
   projection = embedding weight, no transform needed).
6. **`cnets_dyn_res.py`'s self-attention sized the rope cache from the raw
   key-tensor length instead of `position_ids.max() + 1`.** `cnets_res.py`
   (stage 1) already has this fix; `cnets_dyn_res.py` (stage 2) never got the
   equivalent branch, presumably because stage 2 had never been exercised
   with a model that both uses `position_ids` (Qwen) and produces gapped
   position ids (multimodal samples, after `remove_visual_span` strips the
   image-token span but leaves each kept token's original absolute
   position). Result: `cos[position_ids]` indexed past the end of the
   (too-short) rope cache -> `CUDA error: device-side assert triggered`
   inside `apply_rotary_pos_emb`, reported asynchronously several frames
   away from the actual out-of-bounds gather (use
   `CUDA_LAUNCH_BLOCKING=1` if you need to re-diagnose something similar --
   it's what pinned this down to `cnets_dyn_res.py:98`). Fixed by adding the
   same `position_ids.max() + 1` branch, in a separate `rope_seq_len`
   variable so the pre-existing `kv_seq_len` (the real key length) stays
   correct for the `attn_weights.size()` assertion a few lines down --
   `cnets_res.py` doesn't have that downstream assertion, so it could reuse
   one variable for both purposes; `cnets_dyn_res.py` can't.

## Bugs fixed, round 2 (2026-08-28, after actually trying to run generate)

Found running `STAGE=generate` for real, past the earlier smoke test's tiny
40-sample slice:

7. **Only one GPU ever did any work.** `train_official_hivis_qwen3b.sh`
   defaults `END=1000000000000` (meaning "everything from START"). Before
   this fix, `allocation.py` divided `[0, END)` evenly across `--gpus`
   *before* knowing how many rows the data file actually has. With that
   sentinel, GPU 0's slice (`[0, 1.25e11)` for 8 GPUs) comfortably contains
   the entire real dataset, while every other GPU's slice starts far past
   the last real row -- so those workers silently process nothing and GPU 0
   does everything alone. This bug predates this recipe (the SmolVLM script
   has the identical `END=${END:-1000000000000}` default and would hit the
   same thing), but it got copied forward here without being caught. Fixed
   in `allocation.py`'s `main()`: count the data file's rows upfront
   (`count_data_file_rows`, gated on `--data-file` being a real, resolvable
   path -- a no-op fallback otherwise, so callers that don't pass `--end` an
   oversized sentinel are unaffected) and clamp `--end` to that before
   `split_range` runs. Verified with `--dry-run`: `--end 1000000000000` over
   the real 140,000-row mixed file now produces 8 non-empty, evenly-sized
   per-GPU commands instead of 1.
8. **One input file, generated twice.** `run_generate_one()` called
   `allocation.py` once per `--data-type` (`text`, `multimodal`), each
   invocation reloading the target model from scratch and rescanning the
   whole file, just to filter it down to disjoint halves -- 2x the model
   loads and 2x the dataset scanning for one file. Fixed: `ge_data_qwen.py`
   no longer takes `--data-type` at all. It loads the model once, does one
   pass over the file, and routes each record to `--outdir` (text) or the
   new `--multimodal-outdir` (has an image) based on that record's own
   content -- matching the pattern the *original* (pre-rewrite) script
   already used for its code/non-code text split, just applied to the
   text/multimodal split instead. `allocation.py` special-cases `--model
   qwen` to build one command per GPU (not two) and pass both output dirs;
   `llava`'s existing two-pass `--data-type` behavior is untouched.
   `train_official_hivis_qwen3b.sh`'s `generate_text`/`generate_mm` stages
   are gone along with it (a single pass can't produce just one half).
9. **`HIVIS_CONDA_ENV` was an unconditional `PATH` prepend to a path that
   only exists on this one machine.** Fine here, breaks completely on any
   other server (a `uv`-managed env, a different conda install, no conda at
   all) -- `python`/`accelerate` would still resolve to *something*, just
   not to an environment with the right package versions, and the failure
   mode is a confusing crash deep inside dataset loading or model init
   rather than "this env doesn't have what we need." Fixed in both this
   script and the SmolVLM one: only prepend `$HIVIS_CONDA_ENV/bin` if that
   directory actually exists; otherwise it's a silent no-op and whatever the
   caller's shell already has active (activated conda env, `uv` venv, ...)
   is left alone. Set `HIVIS_CONDA_ENV=""` to always skip it.

## Known caveats carried over from the SmolVLM pipeline (not re-verified here)

The SmolVLM README documents fixes already made to the *shared* trainer code
(`main_mix.py`, `main_mix_topk_dyn_res.py`, `cnets_dyn_res.py`) that this
Qwen2.5-VL-3B recipe also depends on: `--num-epochs`/`--max-len` CLI flags,
a `model.safetensors.index.json`-vs-single-file fallback in stage 2's
lm_head/embedding load, a `rope_scaling` missing-`"type"`-key fallback, and a
divide-by-zero guard on the "Train Accuracy" print. Qwen2.5-VL-3B ships
sharded (`model.safetensors.index.json` present, confirmed by the 2-shard
`Loading checkpoint shards` log line during the smoke test), so the
single-file fallback path isn't exercised here, but the other three fixes
are shared code this recipe relies on being correct.
