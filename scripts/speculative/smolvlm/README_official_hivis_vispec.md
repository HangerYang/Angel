# Official HiViS / ViSpec SmolVLM-256M Baselines

Two directly-comparable draft-training recipes for `HuggingFaceTB/SmolVLM-256M-Instruct`:

- **HiViS**: plain teacher-forced hidden-state capture.
- **ViSpec**: same capture, but multimodal samples get ViSpec's Vicuna-style
  system prompt plus "Please answer with at least 1000 words." appended to the
  first user turn (per upstream ViSpec's
  `ge_data_all_llava_pretrain_gen.py`), routed through `--model smolvlm_vispec`.

Both share the same two-stage trainer (`hivis.train.main_mix` then
`hivis.train.main_mix_topk_dyn_res`) and the same input data file — only the
`.ge_data` generation step and output directories differ.

Verified working end-to-end (generate -> stage1 -> stage2) on 2026-08-27 after
fixing four bugs blocking the SmolVLM path (see "Bugs fixed" below).

## Prerequisites

- `conda activate angel`
- `PYTHONPATH="$ROOT:$ROOT/HiViS"` (the scripts below set this themselves)
- Input data file present: `dataset/preprocessed/mixed_sharegpt_llava665k_70k70k_b64.jsonl`
  (~17 GB; mixed ShareGPT text + LLaVA-665k multimodal records, `{"id", "conversations":[{"role","content":[{"type":"text"/"image", ...}]}]}`)
- Free GPU(s) — each `ge_data`/train worker pins itself to one GPU via
  `CUDA_VISIBLE_DEVICES`.

## Quick start

```bash
cd /home/hyang/Angel
conda activate angel

# HiViS baseline: generate -> stage1 (2 epochs) -> stage2 (1 epoch)
GPUS="0 1 2 3" bash scripts/speculative/smolvlm/run_official_hivis_smolvlm_2ep_1ep.sh

# ViSpec-style baseline, same shape, directly comparable
GPUS="0 1 2 3" bash scripts/speculative/smolvlm/run_official_vispec_smolvlm_2ep_1ep.sh
```

Outputs land under:

- `output/hivis_official/smolvlm_256m_stage1_2ep_stage2_1ep/{stage1,stage2}`
- `output/vispec_official/smolvlm_256m_stage1_2ep_stage2_1ep/{stage1,stage2}`

Each stage's checkpoints save as `state_<epoch>/` (accelerate `save_state`
format: `model.safetensors`, `optimizer.bin`, `scheduler.bin`, ...). Stage 2
picks up stage 1's `state_0` by default (`STAGE1_CKPT`).

### Validate without launching anything

```bash
DRY_RUN=1 bash scripts/speculative/smolvlm/run_official_hivis_smolvlm_2ep_1ep.sh
DRY_RUN=1 bash scripts/speculative/smolvlm/run_official_vispec_smolvlm_2ep_1ep.sh
```

Prints the resolved paths/env and exits before touching the GPU.

## Running stages individually

The `run_official_*_2ep_1ep.sh` wrappers just set env vars and call
`train_official_{hivis,vispec}_smolvlm.sh` with `STAGE=all`. Call that script
directly for finer control:

```bash
STAGE=generate      bash scripts/speculative/smolvlm/train_official_hivis_smolvlm.sh   # both text + multimodal
STAGE=generate_text  bash scripts/speculative/smolvlm/train_official_hivis_smolvlm.sh
STAGE=generate_mm    bash scripts/speculative/smolvlm/train_official_hivis_smolvlm.sh
STAGE=stage1         bash scripts/speculative/smolvlm/train_official_hivis_smolvlm.sh
STAGE=stage2         bash scripts/speculative/smolvlm/train_official_hivis_smolvlm.sh
```

Swap `hivis` for `vispec` in the script name for the ViSpec-style variant.
Useful env var overrides (all have defaults, see the script header):
`GPUS`, `DATA_FILE`, `START`/`END` (row-range slice, pre-shuffle),
`MODEL_MAX_LENGTH`, `BS_STAGE1`/`BS_STAGE2`, `LR`, `STAGE1_EPOCHS`,
`STAGE2_EPOCHS`, `FORWARD_NUM_TOTAL`, `TOPK`, `TOPK_W`, `FAIL_FAST`.

### Smoke-testing a tiny slice

To sanity-check the pipeline without processing all ~140k rows:

```bash
export PYTHONPATH="$PWD:$PWD/HiViS"
python -m hivis.ge_data.allocation --model smolvlm --data-type text --gpus 0 \
  --data-file dataset/preprocessed/mixed_sharegpt_llava665k_70k70k_b64.jsonl \
  --outdir /tmp/smoke_hivis_text --end 20
```

(`--model smolvlm_vispec` for the ViSpec variant.) Then point
`TEXT_CKPT_DIR`/`MULTIMODAL_CKPT_DIR` at the smoke output and run
`STAGE=stage1` with a small `--bs`/`STAGE1_EPOCHS=1` override to confirm
training launches before committing to a full run.

## Bugs fixed to make this runnable (2026-08-27)

The SmolVLM `--model` choice was added to `allocation.py`/`model_names.py` but
the rest of the pipeline wasn't fully wired up. Fixed in this session:

1. **`allocation.py` passed `--data-type` to every worker script.**
   `ge_data_smolvlm[_vispec].py` auto-detect text vs. multimodal per record
   and don't accept that flag (unlike `ge_data_llava.py`/`ge_data_qwen.py`,
   which require it) — every smolvlm generate call errored immediately.
   Fixed: only pass `--data-type` for `llava`/`qwen`.
2. **`ge_data_smolvlm[_vispec].py`'s `load_dataset(...)` had no explicit
   schema.** The mixed data file has text-only records for the first ~70k
   rows and multimodal records after; HF `datasets`' JSON loader infers the
   `content` struct from the first chunk it reads (no `image` field) and then
   fails to cast later chunks that do have one. Fixed: pass an explicit
   `Features` schema so the whole file loads without a schema mismatch.
3. **`train_official_*_smolvlm.sh` passed `--num-epochs`/`--max-len` that
   neither `main_mix.py` nor `main_mix_topk_dyn_res.py` defined** — both
   training entry points would fail immediately with "unrecognized
   arguments". Fixed: added the two CLI flags, wired into `train_config`
   (previously hardcoded).
4. **Stage 2 (`main_mix_topk_dyn_res.py` / `cnets_dyn_res.py`) assumed the
   base model always ships as sharded safetensors with an
   `model.safetensors.index.json`.** SmolVLM-256M is small enough to ship as
   a single `model.safetensors` file with no index, so both the lm_head load
   in `main_mix_topk_dyn_res.py` and the embedding load in `cnets_dyn_res.py`
   crashed with `FileNotFoundError`. Also, `cnets_dyn_res.py`'s
   `_init_rope()` assumed `config.rope_scaling` always has a `"type"` key,
   but transformers auto-populates it as `{"rope_type": "default"}` for
   SmolVLM's config, so init crashed with `KeyError`. Fixed all three to
   match the already-correct fallback logic already present in
   `main_mix.py`/`cnets_res.py` (stage 1's counterparts, which is why stage 1
   worked before this session and stage 2 didn't).

Also hardened (unrelated to the above, hit during smoke-testing with a tiny
sample): the "Train Accuracy" print in both `main_mix.py` and
`main_mix_topk_dyn_res.py` divided by an accuracy-metric denominator that can
be 0 on a very small/short batch, crashing *after* the epoch's loss was
already computed and *before* `accelerator.save_state(...)` — losing that
epoch's checkpoint. Now guarded to print 0.00% instead of raising.

## Known caveat (not fixed, out of scope for training)

`HiViS/hivis/model/cnets_hivis.py`, `cnets_vispec.py`, and `cnets_eagle.py`
(used by `model_hivis.py`, the vLLM-eval-time draft loader — not by the
`train_official_*` scripts above) have the same unconditional
`model.safetensors.index.json` assumption as bug #4. Not exercised by this
training recipe, so left alone here, but expect the same `FileNotFoundError`
if/when SmolVLM-256M is loaded as a draft through that path.
