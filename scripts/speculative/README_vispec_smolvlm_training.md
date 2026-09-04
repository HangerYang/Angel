# Training a ViSpec draft for SmolVLM-256M on AngelSlim data

Goal: point ViSpec at
`dataset/angelslim-smolvlm-eagle3-artifacts/data/smolvlm256m/train_path.jsonl`
— the same source data the EAGLE3/HiViS baselines use — and train normally.

## Why HiViS's generated hidden states cannot be reused

The short version: **the text half can be, the image half cannot.**

`HiViS/hivis/ge_data/ge_data_smolvlm.py:141` deletes the image rows *before
saving*:

```python
if has_image:
    input_ids, target = remove_image_tokens(input_ids, target, image_token_id)
```

Sampling `dataset/hivis_smolvlm_256m_generated_artifacts/llava_v1_5_mix665k/`
confirms it — image token 49190 appears **0 times**, sequences are 65-79 tokens.
The size split gives it away too:

| | ckpts | size |
|---|---|---|
| `sharegpt/` (text) | 66,371 | 151 GB |
| `llava_v1_5_mix665k/` (multimodal) | 66,558 | 14 GB |

Near-identical counts, 10x size gap: each image sample is missing the ~1088
visual rows it should carry. ViSpec's `ImgAdaptor` exists only to compress that
span, so the data it trains on is simply not in those files and cannot be
recovered from them.

Field names differ too (`target` vs `hidden_state`, no `image_mask`, no
`inputs_embeds`), but those are renames — the missing rows are the real
blocker.

The text half *is* structurally identical: ViSpec's
`ge_data_all_idefics3_shargpt.py:113` is a plain teacher-forced
`model(input_ids, output_hidden_states=True)`, exactly what HiViS does. If you
want to skip regenerating the 151 GB text half, renaming `target` ->
`hidden_state` on those ckpts is enough. The generator below regenerates both
so one command produces a consistent set.

## The adapter

`ViSpec/vispec/ge_data/ge_data_all_idefics3_angelslim.py` reads the AngelSlim
`{"id", "conversations": [...]}` jsonl and routes each record by its own
content into the two dirs ViSpec's two training stages expect:

| record | output | capture | saved keys |
|---|---|---|---|
| text-only | `--outdir` (stage 2.1) | teacher-forced forward | `inputs_embeds`, `input_ids`, `hidden_state`, `loss_mask` |
| has image | `--multimodal-outdir` (stage 2.2) | `generate()` rollout, Vicuna system prompt + "Please answer with at least 1000 words." | `inputs_embeds`, `hidden_state`, `loss_mask`, `image_mask` |

Those schemas match ViSpec's own `shargpt` / `pretrain_gen` generators exactly,
so `vispec.train.main` and `main_mtp` consume them unmodified.

`allocation_idefics3_angelslim.py` shards a record range across GPUs, same
structure as ViSpec's other allocation scripts, but carrying the jsonl path,
the image root and both output dirs through to the workers.

### Which jsonl to point at

That folder ships the same records twice, and the two pipelines historically
wanted different ones:

| file | images | ViSpec | HiViS |
|---|---|---|---|
| `train_path.jsonl` (0.65 GB) | relative paths (`coco/...`) | yes | **no** |
| `train_path_b64.jsonl` (15.5 GB) | inline `data:image/...;base64,` | yes | yes |

The generator here accepts **both** forms, so either file works: relative paths
resolve under `--image-root` (default `/home/hyang/Angel/dataset/raw/images`,
verified present for all 58,483 coco + 8,088 textvqa references), and data URIs
are decoded inline. Verified equivalent — the same records generated from both
files produce **bit-identical visual embedding rows** (832 image rows each).
Prefer `train_path.jsonl`; it is 24x smaller for the same content.

HiViS cannot use `train_path.jsonl`: its `common.py:load_image_field` opens the
reference verbatim with no root, so a relative path fails — and the failure is
caught and printed, leaving `image=None` while the record continues, i.e. it
degrades to a silently broken sample rather than an error. Point HiViS at
`train_path_b64.jsonl`.

## Running it

```bash
cd ViSpec
DATA=/home/hyang/Angel/dataset/angelslim-smolvlm-eagle3-artifacts/data/smolvlm256m

# Stage 1: generate. Needs GPUs. 132,943 records; text is fast, the multimodal
# half runs generate() per record and is the long pole.
.venv/bin/python -m vispec.ge_data.allocation_idefics3_angelslim \
  --datapath "$DATA/train_path.jsonl" \
  --outdir            vispec_data/smolvlm/text \
  --multimodal-outdir vispec_data/smolvlm/multimodal \
  --start 0 --end 132942 \
  --gpu_ids 0 1 2 3 4 5 6 7

# Stage 2.1: initial draft training on the text half
accelerate launch --multi_gpu --mixed_precision=bf16 -m vispec.train.main \
  --basepath HuggingFaceTB/SmolVLM-256M-Instruct \
  --configpath vispec/train/smolvlm_256M_config.json \
  --tmpdir vispec_data/smolvlm/text/<generated subdir> \
  --cpdir  vispec_data/smolvlm/ckpt_stage1 --bs 1 --max-len 4096

# Stage 2.2: ViSpec proper -- trains the ImgAdaptor on the multimodal half
accelerate launch --multi_gpu --mixed_precision=bf16 -m vispec.train.main_mtp \
  --basepath HuggingFaceTB/SmolVLM-256M-Instruct \
  --configpath vispec/train/smolvlm_256M_config.json \
  --loadpath vispec_data/smolvlm/ckpt_stage1/state_20/model.safetensors \
  --tmpdir vispec_data/smolvlm/multimodal/<generated subdir> \
  --cpdir  vispec_data/smolvlm/ckpt_stage2 \
  --num-q 2 --mtp-steps 1 --use-ours True --bs 1 --max-len 4096
```

`ViSpec/run_smolvlm.sh` is the upstream 3-stage driver; it points at
LLaVA-Pretrain, which is **not on this machine**. Use the commands above
instead, or edit its stage 1.1/1.2 to call the allocation script here.

## Environment

`ViSpec/.venv` (gitignored by ViSpec's own `.gitignore`), built with uv:

```
python 3.11.5 | torch 2.7.0+cu128 | transformers 4.51.3 | accelerate 1.6.0
```

It has to be separate: HiViS's env is py3.9 and AngelSlim's is py3.12 with a
different transformers, and ViSpec pins numpy 2.1.2 (needs py>=3.10).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements.txt
uv pip install --python .venv/bin/python wandb tensorboard
```

Two things that will bite you:

- **`--index-strategy unsafe-best-match` is required.** Without it uv refuses
  to resolve `torch==2.7.0+cu128`, because it will only take a package from the
  first index that offers it and `certifi==2025.4.26` is not on the PyTorch
  index.
- **`wandb` and `tensorboard` are missing from `requirements.txt`** but
  `vispec/train/main.py` imports both (lines 91 and 95). Install them
  separately or training dies at import.

## Verification status

Smoke-tested end to end on **CPU** (all 8 GPUs were held by an unrelated
training job), 2026-09-04:

- text path: 3 records generated, schema as above, 45 supervised tokens;
- multimodal path: 48 records generated; a sample carries 934 rows of which
  **832 are image rows** with `image_mask` marking them — i.e. the visual span
  the ImgAdaptor needs is present, which is exactly what HiViS's data lacks;
- `vispec.train.main` (stage 2.1) ran its training loop over the generated text
  ckpts;
- `vispec.train.main_mtp` (stage 2.2, `--use-ours True --num-q 2`) trained on
  the generated multimodal ckpts and logged 9 steps of real
  `loss` / `acc` / `top_1_acc` to tensorboard (loss ~21.5, acc ~0 — expected,
  training from scratch with no stage-1 weights on 48 samples).

Not verified: a full GPU run, and any quality result. `--max_new_tokens 16` was
used for the smoke test; the real run wants the default 1024.

**Known CPU-only limitation:** `vispec/train/main.py:586` hardcodes
`torch.tensor(correct).cuda()` in the epoch-end metric aggregation, so a
CPU-only run crashes *after* the training loop finishes. Harmless on GPU; left
unpatched to keep upstream divergence minimal.
