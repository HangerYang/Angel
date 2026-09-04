# Evaluating ViSpec draft models

ViSpec ([KangJialiang/ViSpec](https://github.com/KangJialiang/ViSpec)) is a
VLM speculative-decoding method whose draft head keeps the visual span in its
prompt and compresses it with a learned `ImgAdaptor` — a small set of query
vectors that cross-attend over the image embeddings and squeeze them into
`num_q` tokens (default 2). This is the third option alongside `eagle` (feeds
the draft the full multimodal sequence untouched) and `hivis` (prunes the image
rows outright).

There are **two independent harnesses** in this tree. Pick by what you have:

| | `HiViS/` | `ViSpec/` |
|---|---|---|
| entry point | `hivis.evaluation.ge_hivis_answer --draft-method vispec` | `vispec.evaluation.gen_spec_answer_*` |
| can train a ViSpec draft | **no** | yes (`vispec.train.main` + `main_mtp`) |
| models wired up | Qwen2.5-VL, LLaVA, SmolVLM/Idefics3 | Qwen2.5-VL, LLaVA, SmolVLM/Idefics3 |
| benchmarks | ChartQA, ScienceQA, MathVista, DocVQA, vqav2, textvqa, gqa, mme, mmvet, seedbench, mmmu | COCO-caption, gqa, mme, mmvet, sqa, textvqa, seedbench, vizwiz, vqav2, … |

Use `HiViS/` to compare ViSpec against HiViS/EAGLE on the same benchmark
harness. Use `ViSpec/` when you need to *train* a draft — HiViS has no ViSpec
training code at all (`HiViS/hivis/train/` has zero references to it), so
through that path you can only load someone else's checkpoint.

## The vendored `ViSpec/` folder

`ViSpec/` at the repo root is upstream ViSpec as extended for SmolVLM,
exported from `HangerYang/Star` at commit `b3a89c6` ("angel-smol",
2026-07-14) — 104 files, the upstream repo's full tracked set. It carries its
own `.gitignore`, so weights and generated data stay out.

It was previously listed in the root `.gitignore` under "Local nested
repository checkouts"; that line is now removed and the tree is tracked
directly, matching how `HiViS/` is vendored.

What it already has for SmolVLM: `vispec/model/modeling_idefics3_kv.py`,
`vispec/train/smolvlm_256M_config.json`, `vispec/ge_data/allocation_idefics3_*`,
and `run_smolvlm.sh` driving all three stages. What it does **not** have is a
trained checkpoint — there is no `vispec_data/`, so stage 2 has never been run
here.

## Evaluating a Qwen2.5-VL ViSpec draft through HiViS

Note there are two HiViS locations and they are **not** the same tree. The
vendored `HiViS/` in this repo is the code only — `eval_data/`, `hivis/`,
`LICENSE`, `README.md`, `requirements.txt`. Weights, run scripts and outputs
live in the standalone checkout at `/home/hyang/HiViS` (`models/`, `scripts/`,
`outputs/`). Run evals against whichever `hivis` package you mean to test, and
point `--base-model-path` / `--ea-model-path` at the standalone checkout's
`models/`.

```bash
HIVIS_MODELS=/home/hyang/HiViS/models

PYTHONPATH=/path/to/this/repo/HiViS \
python -m hivis.evaluation.ge_hivis_answer \
  --draft-method vispec \
  --base-model-path "$HIVIS_MODELS/Qwen2.5-VL-7B-Instruct" \
  --ea-model-path   "$HIVIS_MODELS/ViSpec-Qwen2.5-VL-7B-Instruct" \
  --dataset ChartQA \
  --answer-file outputs/ChartQA_vispec_qwen25vl_7b.jsonl

# same flags, no draft -- this is the denominator for the speedup
PYTHONPATH=/path/to/this/repo/HiViS \
python -m hivis.evaluation.ge_baseline_answer_hivis \
  --draft-method vispec \
  --base-model-path "$HIVIS_MODELS/Qwen2.5-VL-7B-Instruct" \
  --ea-model-path   "$HIVIS_MODELS/ViSpec-Qwen2.5-VL-7B-Instruct" \
  --dataset ChartQA \
  --answer-file outputs/ChartQA_baseline_qwen25vl_7b.jsonl
```

Checkpoints are published on the Hub, e.g.:

```bash
huggingface-cli download JLKang/ViSpec-Qwen2.5-VL-7B-Instruct \
  --local-dir /home/hyang/HiViS/models/ViSpec-Qwen2.5-VL-7B-Instruct
```

`/home/hyang/HiViS/scripts/run_vispec_qwen3b_one_gpu.sh` is the 3B equivalent
looped over all 11 benchmarks; its results are already in
`/home/hyang/HiViS/speedup.txt`. Two of those rows (DocVQA 10.2x, seedbench
10.4x) come from a collapsed baseline denominator, not a real draft win —
don't quote them without re-running the baseline.

### What a valid ViSpec checkpoint looks like

`JLKang/ViSpec-Qwen2.5-VL-7B-Instruct` is 22 tensors:

```
embed_tokens.weight                      (152064, 3584)   # copied from the target at load time
fc.weight / fc.bias                      (3584, 7168)     # [embed ; hidden] -> hidden
img_fc.weight / img_fc.bias              (3584, 7168)     # [hidden ; last-image summary] -> hidden
imadpt.q                                 (2, 28, 128)     # num_q x heads x head_dim
imadpt.{k,v}_proj.{weight,bias}          (3584, 3584)
imadpt.o_proj.weight                     (3584, 3584)
layers.0.*                               # one decoder layer
```

`imadpt.*` is the tell — an EAGLE or HiViS checkpoint has no `ImgAdaptor`.
Note `imadpt.q`'s leading dim: `EaModel` never passes `num_q`, so the draft is
built with `ImgAdaptor`'s default of 2. A checkpoint trained with a different
`num_q` will fail the `load_state_dict(..., strict=True)` in
`model_hivis.py` — loudly, which is the good case.

## Verification status

Verified on 2026-09-04 with `JLKang/ViSpec-Qwen2.5-VL-7B-Instruct` against
`Qwen2.5-VL-7B-Instruct`, CPU-only:

- the draft config resolves through `EConfig` (hidden 3584, 28 heads,
  `qkv_bias: true`, 1 layer) and matches the target;
- `cnets_vispec.Model` constructs with `num_q=2`;
- `load_state_dict(strict=True)` matches all 22 keys;
- `embed_tokens` is copied non-zero from the target's sharded checkpoint.

**Not yet run: the actual generation loop.** All 8 GPUs were saturated by an
unrelated training job at the time. The command above is unexercised end to
end; run one dataset before trusting a batch of numbers.

## SmolVLM status

Two things had to be true for SmolVLM to work here, and only one of them is.

**Fixed (already on this branch).** `utils_hivis.py` used to pick the image
token id with `151655 if model.is_qwen_vl else 32000` -- Qwen2.5-VL's id, else
LLaVA's. SmolVLM's is **49190**, so it hit neither: the image mask came out
all-False and the `ImgAdaptor` never saw a single image row, with no error
raised. The `hivis` draft method had the same bug via
`prune_image_tokens(..., 32000)`, which is what pinned acceptance at exactly
tau=1.0 on every SmolVLM prompt. `_target_image_token_id()` now reads the id
off the target's own config (`image_token_id`, falling back to the nested
`text_config`, then 32000).

**Still open.** The draft's `embed_tokens` loader in `cnets_vispec.py`
(and `cnets_eagle.py`, `cnets_hivis.py`) assumes the target ships a sharded
`model.safetensors.index.json` and names the table `model.embed_tokens.weight`
or `language_model.model.embed_tokens.weight`. SmolVLM-256M ships **one
unsharded `model.safetensors`** and calls it
**`model.text_model.embed_tokens.weight`**. `cnets_vispec.py` falls back to
`AutoModelForImageTextToText` and then tries `m.language_model.model.embed_tokens`
/ `m.model.embed_tokens`, neither of which exists on Idefics3;
`cnets_eagle.py` and `cnets_hivis.py` have no fallback at all. Loading a
SmolVLM draft through any of the three will fail here until that loader learns
the unsharded + per-family-key case.

And regardless: **no SmolVLM ViSpec checkpoint exists yet.** Producing one
means running `ViSpec/run_smolvlm.sh` stages 1–2, which needs LLaVA-Pretrain
at `LLAVA_DATA_PATH`.

## Not to be confused with

`scripts/speculative/smolvlm/README_official_hivis_vispec.md` describes a
"ViSpec-style" SmolVLM baseline that is **not** this architecture. There, only
the data-generation prompt is ViSpec's (Vicuna system message plus "Please
answer with at least 1000 words."); the model, trainer and loss are all HiViS's,
and there is no `ImgAdaptor`. Keep the two labelled distinctly in any writeup.
