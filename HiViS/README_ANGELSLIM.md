# Running an AngelSlim EAGLE-3 draft inside HiViS

This directory is a vendored checkout of [HiViS](https://arxiv.org/abs/2509.23928)
plus a small port that lets an **AngelSlim-trained EAGLE-3 drafter** run inside
HiViS's own PyTorch generation loop.

## Why

AngelSlim drafts for SmolVLM-256M only ever ran under our patched vLLM
(`third_party/patches/`). HiViS and ViSpec run in this PyTorch/`modeling_llama_kv`
harness. For the paper, every number in a table has to come off the **same
backend** — otherwise the speedup column is comparing kernel stacks, not
drafters. This port makes HiViS the common backend: our drafter, HiViS's
drafter, and the autoregressive baseline all run through the same
`EaModel`, the same target forward, and the same tree/chain verification code.

Nothing here changes how the drafter is trained or what it computes. It is a
loader plus five glue points; the arithmetic is AngelSlim's, executed by HiViS.

## What was added

| File | Change |
|---|---|
| `hivis/model/angelslim_drafter.py` | **new** — loads an AngelSlim EAGLE-3 checkpoint as a HiViS-compatible drafter |
| `hivis/model/model_hivis.py` | `draft_method="angelslim_eagle3"`; aux-hidden-state capture; SmolVLM target backbone swap; shared AR baseline path |
| `hivis/model/utils_hivis.py` | SmolVLM prefill embeds; angelslim branches in `initialize_tree` / `generate_initial_tree` / `update_inference_inputs` |
| `hivis/model/kv_cache.py` | resolve `config.text_config` and the decoder-layer path for `Idefics3ForConditionalGeneration` |
| `hivis/evaluation/benchmark_data.py` | `omnidocbench` + `mmmu_history` (our vLLM-matched rows) and `supported_benchmarks()` |
| `run_angelslim_eval.py` | **new** — acceptance-length / throughput harness, either drafter family × any benchmark |

Everything else under `hivis/` is upstream, byte for byte — including
`hivis/evaluation/data/`, the metadata for the five benchmarks HiViS does not
pull from the Hub (see "Datasets" below).

## The five things that had to be bridged

1. **Importing `angelslim` under Python 3.9.** The `hivis` env is py3.9 /
   torch 2.6 / transformers 4.54; `angelslim/__init__.py` reaches
   `qat/modules/quantizer.py` whose class body evaluates a py3.10-only
   `X | None` annotation and raises `TypeError` at import time.
   `load_angelslim_draft_module()` registers empty stub `ModuleType` objects for
   `angelslim`, `angelslim.compressor`, … so the package `__init__` never runs
   and only the one draft module is loaded from source.
   Root defaults to `/home/hyang/Angel`, override with `ANGELSLIM_ROOT`.

2. **Config translation, tf5 → tf4.** The checkpoints were written by
   transformers 5.x, which nests RoPE under `rope_parameters`. transformers 4.54
   reads `config.rope_theta` and silently falls back to `10000.0` if it is
   missing — a wrong RoPE base that degrades acceptance without erroring.
   `DrafterConfig` translates `rope_parameters.rope_theta` → `rope_theta` and
   `rope_type` → `rope_scaling` explicitly. (`rope_theta == 100000.0`,
   `rope_scaling is None` for all our checkpoints.)

3. **`banded_mix_fc` sizing.** In this mode the drafter takes 9 target aux
   streams, mixes them into 3 bands with a learned per-band softmax
   (`band{i}_mix_logits`), then feeds the stock EAGLE-3.1 `fc_norm` + `fc`
   (3H→H). AngelSlim's base drafter sizes `fc`/`fc_norm` from
   `len(aux_hidden_states_layer_ids)`, which would give 9 streams and a shape
   mismatch. `DrafterConfig` therefore advertises the **band** count to the
   constructor, and the real 9-layer list is restored on the built module
   afterwards for the aux-capture hook. `fused_fc` (stock EAGLE-3) takes the
   plain 3-stream path.

4. **Aux hidden states out of a SmolVLM target.** HiViS calls the target for
   `last_hidden_state` only. The angelslim path sets `output_hidden_states=True`
   and concatenates `hidden_states[i + 1] for i in aux_layer_ids` — `+1` because
   HF index 0 is the embedding output, so `hidden_states[i+1]` is the output of
   layer `i`, which is what AngelSlim's vLLM hook records.

5. **A tree-capable SmolVLM target.** Stock `Idefics3` uses transformers' own
   `LlamaModel`, which neither accepts HiViS's legacy list-based KV cache nor
   honours `tree_mask`, so tree verification silently degenerated. At load time
   the text tower is replaced by HiViS's forked
   `modeling_llama_kv.LlamaModel` (eager attention), weights copied across with
   a strict check. Prefill goes through a new `smolvlm_input_embeds()` that
   reproduces Idefics3's `get_image_features` + `inputs_merger` merge, since
   the stock `Idefics3` forward cannot be called for embeds alone here.
   `naivegenerate` was routed through the *same* backbone, so the speedup
   denominator is not measured on a different kernel path from the numerator.

`topK_genrate` returns 5 values in AngelSlim and 4 in HiViS; the mixin trims it
and raises if the 5th is ever non-`None`.

## Verification

The port is faithful, not approximate: feeding identical aux hidden states and
identical input ids to AngelSlim's training-stack drafter and to this
inference-stack drafter produces the **same argmax token**.

Acceptance also matches the vLLM reference. Under chain decoding (K=4,
1024 output tokens) vLLM reports τ = 2.87 for `branch-distill-top1-w01`;
this harness reports 2.913. Acceptance is not degraded by the port, which was
the requirement.

## Environment

Nothing here is tied to a particular environment manager. What the code needs is
a **Python 3.9+ interpreter with HiViS's own dependencies** (`requirements.txt`,
notably `transformers` 4.5x — *not* 5.x, whose `Cache` objects this fork's
`modeling_llama_kv` does not speak) and `HiViS/` importable.

Pick whichever the machine already uses:

```bash
# uv
uv venv && uv pip install -r requirements.txt
source .venv/bin/activate

# conda / mamba
conda create -n hivis python=3.9 && conda activate hivis
pip install -r requirements.txt

# or any interpreter you already have
```

Then, from `HiViS/`:

```bash
python run_angelslim_eval.py ...
```

or from anywhere:

```bash
PYTHONPATH=/path/to/HiViS python /path/to/HiViS/run_angelslim_eval.py ...
```

Only two paths are environment-sensitive and both are overridable, never
hardcoded at call time:

- `ANGELSLIM_ROOT` — where to find the `angelslim` source tree. Left unset it
  is discovered by walking up from this file (HiViS is vendored inside the
  AngelSlim checkout), so it needs setting only if the two live apart.
- `--base` / `--draft` — take local paths or Hub repo ids interchangeably.

## Running

```bash
CKPT=../dataset/angelslim-smolvlm-eagle3-artifacts/weight/branch-distill-top1-w01/checkpoint-66466

# tree decoding (EAGLE-2 style)
python run_angelslim_eval.py --draft $CKPT --n 40 --max_new_tokens 1024 \
    --total_token 60 --depth 5 --top_k 10 --out ../results_tree.json

# chain / linear decoding (K=4) -- what the drafter was trained for
python run_angelslim_eval.py --draft $CKPT --n 40 --max_new_tokens 1024 \
    --total_token 5 --depth 4 --top_k 1 --out ../results_chain.json

# autoregressive baseline (speedup denominator, same backbone)
python run_angelslim_eval.py --draft $CKPT --n 40 --max_new_tokens 1024 \
    --naive --out ../results_naive.json
```

## Running *their* model on *our* benchmarks

`--draft_method` selects the drafter family and `--dataset` the rows; the two are
independent, so all four combinations work through one script:

```bash
# HiViS's published drafter, on the benchmark our vLLM eval uses
python run_angelslim_eval.py --draft_method hivis \
    --base Qwen/Qwen2.5-VL-7B-Instruct \
    --draft Irisssme/HiViS-Qwen2.5-VL-7B-Instruct \
    --dataset omnidocbench --n 40 --max_new_tokens 1024 \
    --total_token 60 --depth 5 --top_k 10 --out ../results_hivis_omni.json

# ViSpec's, likewise
python run_angelslim_eval.py --draft_method vispec \
    --base Qwen/Qwen2.5-VL-3B-Instruct \
    --draft JLKang/ViSpec-Qwen2.5-VL-3B-Instruct --dataset omnidocbench ...

# ...and ours on one of theirs
python run_angelslim_eval.py --draft $CKPT --dataset DocVQA ...
```

`--base` must be the target the drafter was trained against — a drafter is tied
to its target's hidden size, vocabulary and `d2t` map, so mixing them is a shape
error at best and silent garbage at worst. The script does not guess it.

**What this does and does not license you to compare.** Two different drafters
on the same rows, in this harness, share a backend and a decoding loop, so their
τ and their speedup-over-own-baseline are comparable. Their *absolute* tok/s is
not: HiViS's checkpoint drives Qwen2.5-VL-7B and ours drives SmolVLM-256M, so
the tok/s columns are 7B numbers next to 256M numbers. Always quote speedup
against each model's own `--naive` run in the same configuration.

## Datasets

`--dataset` accepts thirteen names, all resolved by
`hivis/evaluation/benchmark_data.py`:

| Source | Benchmarks | Sampling |
|---|---|---|
| ours, Hub | `omnidocbench` (default) `mmmu_history` | **first N, unshuffled** |
| HiViS, Hub | `ScienceQA` `ChartQA` `MathVista` `DocVQA` `vqav2` `mmmu` | shuffled, seed 42 |
| HiViS, local images | `gqa` `textvqa` `mme` `mmvet` `seedbench` | shuffled, seed 42 |

The last row needs image directories downloaded per HiViS README section 1.1;
the JSON/JSONL metadata is already vendored under `hivis/evaluation/data/`.

`omnidocbench` and `mmmu_history` are ours, added so a HiViS or ViSpec drafter
can be measured on the rows our vLLM eval uses. They deliberately sample
**first-N and unshuffled**, because `tools/vllm_offline_eagle3_vlm_batch.py`
does — same rows is the whole reason a PyTorch τ can be held next to a vLLM τ.
Do not "fix" them to use the seed-42 shuffle the HiViS benchmarks use; that
silently breaks the cross-backend comparison and every number in the table
below.

## Results

OmniDocBench, 40 prompts, greedy, `max_new_tokens=1024`, SmolVLM-256M-Instruct
target, single A-series GPU. All rows share one backbone and one loop.

| | τ (accept. length) | tok/s | speedup | avg output |
|---|---:|---:|---:|---:|
| autoregressive baseline | 1.000 | 34.17 | 1.00× | 565.4 |
| **chain** · `baseline_1layer` (stock EAGLE-3) | 2.589 | 65.92 | 1.93× | 549.1 |
| chain · `banded-mix-fc-3.1` | 2.727 | 68.47 | 2.00× | 502.3 |
| chain · `branch-distill-top1-w01` | **2.913** | **72.33** | **2.12×** | 594.4 |
| **tree** · `baseline_1layer` (stock EAGLE-3) | 3.610 | 85.08 | 2.49× | 532.3 |
| tree · `branch-distill-top1-w01` | 3.907 | 86.13 | 2.52× | 512.1 |
| tree · `banded-mix-fc-3.1` | 3.927 | 89.30 | 2.61× | 524.7 |

`baseline_1layer` is the stock EAGLE-3 drafter (`fused_fc`, 3 aux streams, no
`fc_norm`, no `norm_output`) trained for the same 66466 steps on the same data —
the honest ablation denominator. Against it:

| | Δτ | Δ speedup |
|---|---:|---:|
| chain · `branch-distill-top1-w01` | **+0.324 (+12.5 %)** | +0.19× (1.93→2.12) |
| chain · `banded-mix-fc-3.1` | +0.138 (+5.3 %) | +0.07× |
| tree · `branch-distill-top1-w01` | +0.297 (+8.2 %) | +0.03× |
| tree · `banded-mix-fc-3.1` | +0.317 (+8.8 %) | +0.12× |

### Reading the numbers

- **Acceptance rises with generation length on OmniDocBench.** Position 0 alone
  is 0.150, the first 96 tokens 0.432, the first 256 0.571, the first 1024
  0.714. It is an OCR task: hard boilerplate up front, easy transcription after.
  Any τ quoted from this benchmark is meaningless without the output length
  next to it. A short run reads as a regression when it is not.
- **Both band models beat stock EAGLE-3 in both modes**, by 5–12 % acceptance.
  That gap is the one the paper should claim.
- **The ranking *between the two band models* flips between modes.** Chain puts
  `branch-distill` ahead of `banded-mix-fc-3.1` by +0.186; tree puts it behind
  by 0.020 (noise). Branch distillation trains exactly the top-1 quality that a
  10-candidate first tree level dilutes, so its advantage is real under chain
  decoding — what it was trained for — and washes out under tree. Worth saying
  out loud rather than reporting only the mode that favours the method.
- **Acceptance gains compress into speedup gains.** +12.5 % τ buys +10 % tok/s
  under chain; under tree, +8 % τ buys ~1 %. Eager attention makes the target
  forward dominate, so a longer accepted run has less to win back.

## Limitations

- HiViS's forked Llama runs **eager attention only**, so absolute tok/s here is
  lower than vLLM's and the reported speedups are conservative. Rows are fair
  against each other and against HiViS's own baselines; they are not directly
  comparable to vLLM numbers.
- Only `banded_mix_fc` and `fused_fc` injection modes are wired up.
  `build_drafter` raises `NotImplementedError` on anything else rather than
  loading a drafter that silently computes the wrong thing.
- SmolVLM / Idefics3 targets only. The Qwen2.5-VL branch raises if aux capture
  is requested.
- Single-layer drafters only; `build_drafter` raises if `num_hidden_layers != 1`.
