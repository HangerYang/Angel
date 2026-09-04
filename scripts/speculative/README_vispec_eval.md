# Evaluating ViSpec draft models

ViSpec ([KangJialiang/ViSpec](https://github.com/KangJialiang/ViSpec)) is a
VLM speculative-decoding method whose draft head keeps the visual span in its
prompt and compresses it with a learned `ImgAdaptor` — a small set of query
vectors that cross-attend over the image embeddings and squeeze them into
`num_q` tokens (default 2). This is the third option alongside `eagle` (feeds
the draft the full multimodal sequence untouched) and `hivis` (prunes the image
rows outright).

Evaluation runs through **one harness only**: `hivis.evaluation.ge_hivis_answer`,
with `--draft-method {eagle, hivis, vispec, angelslim_eagle3}`. ViSpec's own
evaluation code has been removed from `ViSpec/` so there is a single place
numbers come from. `ViSpec/` is kept for what HiViS cannot do — *training* a
ViSpec draft (`HiViS/hivis/train/` has no ViSpec code at all).

## What actually loads, per draft method and target

Verified by constructing each draft against each target (2026-09-04):

| draft method | Qwen2.5-VL | SmolVLM-256M |
|---|---|---|
| `hivis` | works | **works** |
| `angelslim_eagle3` | untested | **works** |
| `eagle` | works | fails — `FileNotFoundError` |
| `vispec` | loads (generation loop unexercised) | fails — `IsADirectoryError` |

The split is not about the target — it is about whether the draft method copies
the **target's** embedding table into the draft at construction.
`model_hivis.py:136` passes `path=base_model_name_or_path` for `eagle` and
`vispec` only, which turns on `load_emb`; `hivis` and `angelslim_eagle3` never
touch it and so never hit the loader below. That is why SmolVLM has been
evaluated successfully all along on those two.

That loader (`cnets_{eagle,hivis,vispec}.py`) assumes the target ships a
sharded `model.safetensors.index.json` and names the table
`model.embed_tokens.weight` or `language_model.model.embed_tokens.weight`.
SmolVLM-256M ships **one unsharded `model.safetensors`** and calls it
**`model.text_model.embed_tokens.weight`**, so every branch misses:
`cnets_eagle`/`cnets_hivis` have no fallback and raise `FileNotFoundError`;
`cnets_vispec` falls back to `AutoModelForImageTextToText` and then tries
`m.language_model.model.embed_tokens` / `m.model.embed_tokens`, neither of
which exists on Idefics3, and ends in `torch.load(<a directory>)`.

Fixing it is one loader: enumerate the checkpoint's keys and take the
non-vision `*embed_tokens.weight`, which covers Qwen (`model.embed_tokens.weight`)
and SmolVLM alike without branching per family. Not done.

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

The image-token-id bug is fixed. `utils_hivis.py` used to pick the id with
`151655 if model.is_qwen_vl else 32000` — Qwen2.5-VL's, else LLaVA's. SmolVLM's
is **49190**, so it matched neither: nothing was pruned and the drafter was
handed ~800 image rows it had never seen in training, which read as a dead
checkpoint (tau pinned at 1.000) rather than a dead code path. The id is now
read off the target's own config. On a real SmolVLM prompt the matched image
rows go from 0 to 1088 out of 1141 tokens.

What remains is the embedding loader described above, which gates `eagle` and
`vispec` on SmolVLM but not `hivis` or `angelslim_eagle3`.

And regardless: **no SmolVLM ViSpec checkpoint exists yet.** Producing one
means running `ViSpec/run_smolvlm.sh` after pointing its data stage at the
AngelSlim jsonl — see `README_vispec_smolvlm_training.md`.

## Not to be confused with

`scripts/speculative/smolvlm/README_official_hivis_vispec.md` describes a
"ViSpec-style" SmolVLM baseline that is **not** this architecture. There, only
the data-generation prompt is ViSpec's (Vicuna system message plus "Please
answer with at least 1000 words."); the model, trainer and loss are all HiViS's,
and there is no `ImgAdaptor`. Keep the two labelled distinctly in any writeup.
