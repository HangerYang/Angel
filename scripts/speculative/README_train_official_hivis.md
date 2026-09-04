# Official HiViS Training: Unified Launcher

`train_official_hivis.sh` runs the official HiViS two-stage draft-training
recipe (stage 1: `hivis.train.main_mix`; stage 2:
`hivis.train.main_mix_topk_dyn_res`) against any of three supported target
models, selected by a single `MODEL` variable:

| `MODEL`        | Target model                                   | Draft config |
|----------------|-------------------------------------------------|--------------|
| `qwen25vl_7b`  | `/home/hyang/HiViS/models/Qwen2.5-VL-7B-Instruct` (local) | `hivis/train/qwen2.5_vl_7B_config.json` |
| `qwen25vl_3b`  | `/home/hyang/HiViS/models/Qwen2.5-VL-3B-Instruct` (local) | `hivis/train/qwen2_5_vl_3B_config.json` |
| `smolvlm256m`  | `HuggingFaceTB/SmolVLM-256M-Instruct` (HF hub)  | `hivis/train/smolvlm_256m_config.json` |

Before this script, Qwen2.5-VL-3B and SmolVLM-256M each had their own
~180-line launcher (`qwen2_5_vl/train_official_hivis_qwen3b.sh`,
`smolvlm/train_official_hivis_smolvlm.sh`) that were identical except for a
handful of per-model constants (target path, config path, `ge_data_<model>`
module name). Those two files now just `exec` into this one with `MODEL`
preset, so any existing caller that `bash`es them directly keeps working
unchanged — see "Relationship to the old per-model scripts" below.

## Why a single script instead of one per model

The actual training logic (`run_generate`/`run_stage1`/`run_stage2`, GPU/DDP
setup, env-var defaults) was byte-for-byte identical across the Qwen and
SmolVLM launchers — only the target path, draft config, and the
`hivis.ge_data.allocation --model` flag differed. Keeping two copies meant
every bug fix (GPU-split, double-generate, hardcoded-env, DDP backend — see
the per-model READMEs' "Bugs fixed" sections) had to be applied twice and
inevitably drifted (the SmolVLM script was missing the `--ddp-backend` and
code/non_code stage-2 fixes for a while after the Qwen script got them).
Adding Qwen2.5-VL-7B as a third target on top of that would have meant a
*third* near-duplicate file. One script parameterized by `MODEL` fixes that.

## Quick start

```bash
cd /home/hyang/Angel
conda activate hivis   # or: source your uv venv -- see "No hardcoded environment" below

# Validate the command construction without launching anything
MODEL=smolvlm256m DRY_RUN=1 GPUS="0 1" bash scripts/speculative/train_official_hivis.sh

# Small smoke test: 8 samples, 1 GPU, 1 epoch each stage
MODEL=smolvlm256m GPUS="0" START=0 END=8 \
STAGE1_EPOCHS=1 STAGE2_EPOCHS=1 BS_STAGE1=1 BS_STAGE2=1 \
HIVIS_DATA_ROOT=/home/hyang/tmp/smoke_smolvlm \
OUTPUT_ROOT=/home/hyang/tmp/smoke_smolvlm_out \
STAGE=all bash scripts/speculative/train_official_hivis.sh

# Full run: generate -> stage1 -> stage2, 8 GPUs
MODEL=qwen25vl_7b GPUS="0 1 2 3 4 5 6 7" bash scripts/speculative/train_official_hivis.sh
```

`STAGE` selects which phase(s) to run: `generate`, `stage1`, `stage2`, or
`all` (default). Every other setting has a `MODEL`-aware default and can be
overridden by exporting the same-named env var — see "All env vars" below.

## Dataset: one JSONL file, no manual merge step

### Input format

A single JSONL file, each line:

```json
{"id": "...", "conversations": [
  {"role": "user", "content": [{"type": "text", "text": "..."}]},
  {"role": "user", "content": [{"type": "image", "image": "<path or data:image/...;base64,...>"}]},
  {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
]}
```

Default: `dataset/preprocessed/mixed_sharegpt_llava665k_70k70k_b64.jsonl`
(mixed ShareGPT text + LLaVA-665k multimodal, images as base64 data URIs).
Override with `DATA_FILE`. The format is target-model-agnostic — what's
actually target-specific is the hidden-state `.ckpt` cache the `generate`
stage produces by running this raw data through the real target model.

### One `generate` pass, routed by content, no `--data-type`

`hivis/ge_data/ge_data_qwen.py` and `ge_data_smolvlm.py` load the target
model **once** and route each record based on its own content: has an image
→ `--multimodal-outdir`; text-only → `--outdir`, further split into
`code/`/`non_code/` via HiViS's own `is_code_heavy()` heuristic (verbatim
from `eval_data/sharegpt/prepare_data.py`, applied to the assistant turns —
this is the same split HiViS's stock two-file pipeline produces, and it's
what makes stage 2's `--text-data-dir=.../non_code` a real, populated
directory). Shared logic (schema, image loading, loss-mask construction,
`is_code_heavy`) lives in `hivis/ge_data/common.py`; each `ge_data_<model>.py`
keeps only what's actually model-specific (model class, vision-token
handling, chat markers).

### Multi-GPU splits: yes, they auto-merge — no manual step

`hivis/ge_data/allocation.py` splits the row range across `--gpus` and
launches one `ge_data_<model>.py` worker per GPU; each worker writes into
`<outdir>/<index>/data_*.ckpt` (or `<outdir>/code|non_code/<index>/...` for
text). Training's `main_mix.py`/`main_mix_topk_dyn_res.py` `list_files()`
does `Path(directory).rglob("*")` — a **recursive** glob — so it picks up
every per-GPU `<index>/` subfolder under `--text-data-dir`/
`--multimodal-data-dir` automatically. Point stage 1 at the parent
`sharegpt/`/`llava_v1_5_mix665k/` directories (the script does this by
default) and it trains on all GPUs' output as one pool, regardless of how
many GPUs generated it. Verified live: an 8-sample smoke generate on 1 GPU
produced `sharegpt/code/0/`, `sharegpt/non_code/0/`, `llava_v1_5_mix665k/0/`,
and stage 1 logged `Loaded 6 text training files` (3 code + 3 non_code
merged) + `Loaded 2 multimodal training files` without any extra step.

### A real bug this uncovered: cross-chunk schema cast failure

`mixed_sharegpt_llava665k_70k70k_b64.jsonl` isn't shuffled on disk — its
first 70,000 lines are pure-text (no `image` key anywhere in their
`content` entries) and the next 70,000 are multimodal. HF `datasets`'
`load_dataset("json", ...)` builds its arrow table in ~10MB chunks and
infers each chunk's arrow type from what's actually in it; a chunk that
lands entirely inside the pure-text run gets `content: list<struct<type,
text>>` (no `image` field), and pyarrow cannot cast that into the union
schema (`list<struct<type, text, image>>`) once it's nested inside a list
column — `TypeError: Couldn't cast array of type struct<type: string, text:
string> to {..., image, ...}`. This reproduces regardless of chunk size:
too small and most chunks are one-type-only; too large (e.g. one chunk per
70k-line block) and pyarrow hits a separate `~2GB` offset overflow trying to
combine chunks. Passing an explicit `features=` to `load_dataset` does
**not** avoid it — the cast still happens per-chunk before the explicit
schema can help.

Fixed in `common.py`'s `load_record_dataset()`: build the `Dataset` via
`Dataset.from_generator(..., features=RECORD_FEATURES)` instead, which
encodes each row against the schema individually as it's read (filling a
missing `image` key with null per-row) rather than casting a pre-built
chunk. Verified live: both the smolvlm256m and qwen25vl_7b `generate` smoke
tests above failed with the cast error before this fix and succeeded after,
against the real 140,000-line combined file.

## GPU selection and DDP backend

- `GPUS="0 1 2 3"` (space-separated indices, default `0 1 2 3`) — used both
  for `hivis.ge_data.allocation --gpus` (one data-gen worker per listed GPU)
  and as `CUDA_VISIBLE_DEVICES` for `accelerate launch` in stage 1/2. With
  more than one GPU, the script passes `--multi_gpu --num_processes <n>`
  explicitly — without an `accelerate config` file, `accelerate launch`
  otherwise silently falls back to a single process even if
  `CUDA_VISIBLE_DEVICES` lists several GPUs.
- `DDP_BACKEND={nccl,gloo}` (default: unset, meaning accelerate/torch pick
  `nccl` on CUDA). Wired into `main_mix.py`/`main_mix_topk_dyn_res.py` via
  `accelerate.utils.InitProcessGroupKwargs(backend=...)`. Use `gloo` if
  `nccl` is unavailable or misbehaving on a given machine; no measurable
  training-dynamics or wall-clock difference was found between the two in
  this repo's own comparison.

## No hardcoded environment

This script does **not** activate, reference, or prepend any conda
environment path. It relies entirely on whatever `python`/`accelerate`
already resolve to in the caller's shell — activate your environment
(`conda activate hivis`, a `uv` venv, anything else) *before* running it.
This matters because training is moving to a second server that uses `uv`
instead of conda; an earlier version of the per-model scripts prepended
`/home/hyang/anaconda3/envs/hivis/bin` unconditionally, which broke
`python`/`accelerate` resolution on any machine without that exact conda
install.

The one thing this script does still `source`, if present, is
`third_party/env.sh` (repo-local setup — e.g. `PYTHONPATH` additions for
vendored packages — not an environment/conda concern); it's a no-op if the
file doesn't exist.

## All env vars

All have per-`MODEL` defaults; override any of them by exporting before
calling the script (or prefixing the call, e.g. `VAR=val bash ...`).

| Var | Default | Meaning |
|---|---|---|
| `MODEL` | *(required)* | `qwen25vl_7b` \| `qwen25vl_3b` \| `smolvlm256m` |
| `STAGE` | `all` | `generate` \| `stage1` \| `stage2` \| `all` |
| `TARGET_MODEL_NAME_OR_PATH` | per-`MODEL` (table above) | target model path or HF repo id |
| `DATA_FILE` | `dataset/preprocessed/mixed_sharegpt_llava665k_70k70k_b64.jsonl` | input JSONL |
| `HIVIS_DATA_ROOT` | `dataset/hivis_<MODEL>_generated` | parent dir for generated `.ckpt` data |
| `TEXT_CKPT_DIR` | `$HIVIS_DATA_ROOT/sharegpt` | text `.ckpt` output (and stage 1's `--text-data-dir`) |
| `MULTIMODAL_CKPT_DIR` | `$HIVIS_DATA_ROOT/llava_v1_5_mix665k` | multimodal `.ckpt` output |
| `CONFIG_PATH` | per-`MODEL` (table above) | draft model config JSON |
| `OUTPUT_ROOT` | `output/hivis_official/<MODEL>` | parent dir for checkpoints |
| `STAGE1_DIR` / `STAGE2_DIR` | `$OUTPUT_ROOT/stage1` / `stage2` | per-stage checkpoint dirs |
| `STAGE1_CKPT` | `$STAGE1_DIR/state_0` | stage 1 checkpoint stage 2 loads. **The default is the 1-epoch state**, not the last one — `state_N` is written after epoch N, so `STAGE1_EPOCHS=2` leaves `state_0/1/2` and this points at the weakest. Set it explicitly. |
| `RESUME_FROM` | *(unset)* | a stage-2 `state_<N>` to continue at epoch N+1, optimizer/scheduler/RNG intact. Mutually exclusive with `STAGE1_CKPT`; when set, `STAGE1_CKPT` is not passed |
| `GPUS` | `0 1 2 3` | space-separated GPU indices |
| `DDP_BACKEND` | *(unset → nccl)* | `nccl` \| `gloo` |
| `START` / `END` | `0` / `1000000000000` | row-range slice, pre-shuffle; `END` is clamped to the file's real row count in `allocation.py`, so the huge sentinel just means "everything" |
| `MODEL_MAX_LENGTH` | `4096` | tokenizer/processor max length |
| `NUM_WORKERS` | `1` | DataLoader workers in `generate` |
| `BS_STAGE1` / `BS_STAGE2` | `4` / `2` | per-GPU batch size |
| `GRAD_ACCUM` | `1` | gradient accumulation steps |
| `LR` | `3e-5` | learning rate |
| `STAGE1_EPOCHS` / `STAGE2_EPOCHS` | `20` / `10` | epoch counts |
| `FORWARD_NUM_TOTAL` | `3` | stage 2 rollout length |
| `TOPK` / `TOPK_W` | `10` / `1.0` | stage 2 top-k loss params |
| `FAIL_FAST` | `false` | stop `generate` on first per-record error instead of skipping it |
| `DRY_RUN` | `0` | print the resolved config and exit without launching anything |
| `HIVIS_ROOT` | `HiViS/` next to this repo | HiViS checkout to run against |

## Resuming / continuing from a checkpoint

**Stage 2 can now resume; stage 1 still cannot.** Each script calls
`accelerator.save_state(output_dir=f"{cpdir}/state_{epoch}")` once per full
pass, so `--num-epochs N` writes `state_0` through `state_N`, i.e. N+1
checkpoints — see the loop `for epoch in range(num_epochs + 1)`. `state_N`
therefore means "epoch N is finished".

- **Stage 2** takes `--resume_from <state_N>` (env `RESUME_FROM` through
  `train_official_hivis.sh`). It calls `accelerator.load_state()` after
  `prepare()`, restoring model, optimizer, LR scheduler and RNG, and continues
  at epoch N+1. Keep `--num-epochs` at whatever the original run used, or
  raise it to extend; the script refuses a checkpoint that already completes
  it. `--ckpt_path` is ignored when `--resume_from` is given.
- **Stage 1** still has no load path of any kind — no `--resume_from`, no
  `--ckpt_path`. It always builds `Model(config, load_emb=True,
  path=base_model_path)` from scratch. Continuing stage 1 for more epochs
  would need the same treatment stage 2 just got.

Two limits on the stage-2 resume:

- **Epoch granularity only.** A run killed mid-epoch restarts that epoch.
  Mid-epoch resume would need a saved step counter plus
  `accelerate.skip_first_batches`.
- **A different GPU count is not an equivalent resume.** `total_steps` is a
  fixed 800000 so the LR schedule's shape does not depend on world size, but
  how many optimizer steps an epoch takes does, so the run advances along that
  schedule at a different rate than before.

Checkpoints written before this change hold only `random_states_0.pkl`,
because `save_state` used to run under `if accelerator.is_local_main_process`.
They still resume — `load_accelerator_state` swallows the missing per-rank
file in a `try/except` — but ranks other than 0 do not restore their RNG, and
it says so only at `logger.info`. Checkpoints written from now on have one
file per rank.

Separately from resuming, weight-chaining stage 1 → stage 2 works as it always
did, and is what a *first* stage-2 run wants:

- `main_mix_topk_dyn_res.py` (stage 2) takes `--ckpt_path <dir>`, loads
  `<dir>/model.safetensors` via `safetensors.torch.load_file` +
  `model.load_state_dict(..., strict=True)`, **before** `accelerator.prepare`
  — so it's a fresh optimizer/scheduler seeded with stage 1's model weights,
  not a resume.
- `main_mix.py` (stage 1) has **no `--ckpt_path` argument at all**, as above.

### Recipe: continue from our SmolVLM-256M stage1(20ep) checkpoint into stage2

We ran stage 1 alone for 20 epochs (21 passes) on `smolvlm256m`, final
checkpoint `state_20` (Loss=0.4472, Train Accuracy=85.64%):

```
output/hivis_official/smolvlm_256m_artifacts_stage1_20ep_only/stage1/state_20/
```

A minimal eval-ready copy (`model.safetensors` + `config.json`, no
optimizer/scheduler/rng, no aux training head) is also backed up on the HF
Hub dataset repo at
`WindUpHanger/angelslim-smolvlm-eagle3-artifacts/weight/hivis_way_smolvlm256m/checkpoint-stage1-20ep/`.

To feed that into stage 2 (fresh optimizer, stage1's weights as the start
point):

```bash
cd /home/hyang/Angel
conda activate hivis

MODEL=smolvlm256m STAGE=stage2 \
HIVIS_DATA_ROOT=/home/hyang/Angel/dataset/hivis_smolvlm_256m_generated_artifacts \
STAGE1_CKPT=/home/hyang/Angel/output/hivis_official/smolvlm_256m_artifacts_stage1_20ep_only/stage1/state_20 \
OUTPUT_ROOT=/home/hyang/Angel/output/hivis_official/smolvlm_256m_artifacts_stage1_20ep_only \
GPUS="0 1 2 3 4 5 6 7" \
STAGE2_EPOCHS=<n> \
BS_STAGE1=2 BS_STAGE2=1 \
bash scripts/speculative/train_official_hivis.sh
```

`HIVIS_DATA_ROOT` above is the actual generated-artifacts directory this
checkpoint's stage 1 trained on (**not** the script's own default
`dataset/hivis_smolvlm256m_generated` — no underscore, different path) —
must match or stage 2 trains on the wrong/nonexistent data. `BS_STAGE1`/
`BS_STAGE2` pinned to 2/1 — the script's own unset-env defaults, 4/2, OOM'd
on this machine's free GPU memory during this run; see git history
(`48fd685`) for that fix.

## Relationship to the old per-model scripts

`scripts/speculative/qwen2_5_vl/train_official_hivis_qwen3b.sh` and
`scripts/speculative/smolvlm/train_official_hivis_smolvlm.sh` are now thin
wrappers (`exec env MODEL=<model> bash train_official_hivis.sh`), kept only
so existing callers that `bash` those exact paths
(`run_official_hivis_qwen3b_2ep_1ep.sh`,
`run_official_hivis_smolvlm_2ep_1ep.sh`, `queue_hivis_smolvlm_artifacts.sh`)
keep working unchanged — verified via dry run through all three callers.
Two behavior changes those callers don't hit (they set every affected var
explicitly) but a fresh invocation of the bare wrapper scripts would:

- The SmolVLM wrapper's old default `HIVIS_DATA_ROOT`/`OUTPUT_ROOT` used
  `smolvlm_256m` (underscore); the unified default uses `smolvlm256m`
  (matching `MODEL`, no underscore).
- The SmolVLM wrapper's old `STAGE=generate` did two passes
  (`generate_text` + `generate_mm`, reloading the model each time); the
  unified `generate` does the one-pass, content-routed split described
  above. `generate_text`/`generate_mm` no longer exist as separate stages.

The per-model READMEs (`qwen2_5_vl/README_official_hivis_qwen3b.md`,
`smolvlm/README_official_hivis_vispec.md`) keep their historical "Bugs
fixed" logs as a debugging record; treat this file as the current source of
truth for how to actually run any of the three models.

## Verification performed

All of the following were run live (not just dry-run) against the real
`hivis` conda env and the real 140,000-line combined data file, in this
session:

- `DRY_RUN=1` for all three `MODEL` values — correct target path, config
  path, and output dirs each.
- `MODEL=smolvlm256m STAGE=generate`, 8 real samples, 1 GPU — produced the
  correct 2 multimodal / 3 code / 3 non_code split, no errors.
- `MODEL=smolvlm256m STAGE=stage1` on that 8-sample output — trained for 2
  epochs, logged `Loaded 6 text training files` + `Loaded 2 multimodal
  training files` (confirming the code/non_code auto-merge).
- `MODEL=qwen25vl_7b STAGE=generate`, 8 real samples, 1 GPU (this target had
  never been run through this pipeline before this session) — loaded the
  real 7B checkpoint (5 shards) and produced the identical 2/3/3 split.
- `MODEL=qwen25vl_3b STAGE=generate`, 8 real samples, 1 GPU — same 2/3/3
  split, confirming the `common.py` refactor didn't change behavior for this
  target either.
- Wrapper compatibility: dry runs through `train_official_hivis_qwen3b.sh`
  directly, and through both `run_official_hivis_qwen3b_2ep_1ep.sh` and
  `run_official_hivis_smolvlm_2ep_1ep.sh`.

Not yet run: a full-scale (non-smoke) end-to-end training for any of the
three targets.
