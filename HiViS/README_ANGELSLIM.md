# Running an AngelSlim EAGLE-3 draft inside HiViS

This directory is a vendored checkout of [HiViS](https://arxiv.org/abs/2509.23928)
plus a port that lets an **AngelSlim-trained EAGLE-3 drafter** run inside HiViS's
own PyTorch generation loop — and, in the other direction, lets HiViS's and
ViSpec's published drafters run on the benchmarks our vLLM eval uses.

## Contents

- [Why](#why) · [Which environment runs what](#which-environment-runs-what)
- [What was added](#what-was-added) · [Checkpoints](#checkpoints)
- [What the drafter actually computes](#what-the-drafter-actually-computes)
- [The six things that had to be bridged](#the-six-things-that-had-to-be-bridged)
- [Environment](#environment) · [CLI reference](#cli-reference) · [Running](#running)
- [Datasets](#datasets) · [Output format](#output-format)
- [Results](#results) · [Reproducing the table](#reproducing-the-table)
- [Verification](#verification) · [Troubleshooting](#troubleshooting)
- [Limitations](#limitations) · [Known and deliberately out of scope](#known-and-deliberately-out-of-scope)

## Why

AngelSlim drafts for SmolVLM-256M only ever ran under our patched vLLM
(`third_party/patches/`). HiViS and ViSpec run in this PyTorch/`modeling_llama_kv`
harness. For the paper, every number in a table has to come off the **same
backend** — otherwise the speedup column compares kernel stacks, not drafters.

This port makes HiViS the common backend. Our drafter, the stock EAGLE-3
baseline, HiViS's own drafter, ViSpec's, and the autoregressive baseline all run
through the same `EaModel`, the same target forward, and the same tree/chain
verification code.

Nothing here changes how a drafter is trained or what it computes. It is a
loader plus six glue points; the arithmetic is AngelSlim's, executed by HiViS.

## Which environment runs what

Two conda/uv environments exist in this project and they are **not**
interchangeable. Everything in this README runs in the **HiViS environment**:

| | HiViS env | AngelSlim env |
|---|---|---|
| Python | 3.9 | 3.12 |
| torch | 2.6 + cu124 | 2.11 + cu130 |
| transformers | **4.54** | 5.16 |
| runs | this harness, HiViS, ViSpec | vLLM eval, EAGLE-3 training |

The AngelSlim env is never used by anything in this directory. That is the whole
reason bridge #1 below exists: the drafter's *source* lives in the AngelSlim
tree, but it has to be **imported into the HiViS interpreter**, which is a
Python version and two transformers majors behind. The checkpoint files are
read directly, so no AngelSlim runtime is involved — only the one draft module,
loaded from source.

Do not try to run this in the AngelSlim env. transformers 5.x replaced the
legacy list-based KV cache with `Cache` objects, which this fork's
`modeling_llama_kv` does not speak, and it will fail at the first decode step.

## What was added

| File | Change |
|---|---|
| `hivis/model/angelslim_drafter.py` | **new** — loads an AngelSlim EAGLE-3 checkpoint as a HiViS-compatible drafter |
| `hivis/model/model_hivis.py` | `draft_method="angelslim_eagle3"`; aux-hidden-state capture; SmolVLM target backbone swap; shared AR baseline path; `cache_max_len` |
| `hivis/model/utils_hivis.py` | SmolVLM prefill embeds; angelslim branches in `initialize_tree` / `generate_initial_tree` / `update_inference_inputs` |
| `hivis/model/kv_cache.py` | `Idefics3` config/layer-path resolution; `max_len` cap on the preallocated static cache |
| `hivis/evaluation/benchmark_data.py` | `omnidocbench` + `mmmu_history` (our vLLM-matched rows) and `supported_benchmarks()` |
| `run_angelslim_eval.py` | **new** — acceptance-length / throughput harness, either drafter family × any benchmark |

Everything else under `hivis/` is upstream, byte for byte — including
`hivis/evaluation/data/`, the metadata for the five benchmarks HiViS does not
pull from the Hub.

## Checkpoints

Under `dataset/angelslim-smolvlm-eagle3-artifacts/weight/`, all trained to step
66466 on the same data against SmolVLM-256M-Instruct, all single-layer drafters:

| Checkpoint | Mode | Aux layers | `fc_norm` / `norm_output` | Notes |
|---|---|---|---|---|
| `baseline_1layer` | `fused_fc` | `[1, 14, 26]` | ✗ / ✗ | stock EAGLE-3 — the ablation denominator |
| `banded-mix-fc-3.1` | `banded_mix_fc` | 9 → 3 bands | ✓ / ✓ | band mix, no branch distillation |
| `branch-distill-top1-w01` | `banded_mix_fc` | 9 → 3 bands | ✓ / ✓ | band mix + branch distillation |

Both band checkpoints use aux layers `[2, 4, 8, 10, 15, 18, 20, 26, 28]` grouped
as `[[2,4,8,10], [15,18,20], [26,28]]`, and ship a `banded_aux_mix_weights.json`
recording the learned softmax mix (e.g. band0 = .506/.221/.107/.166 over layers
2/4/8/10). `build_drafter` cross-checks `softmax(band{i}_mix_logits)` against
that file, so a mismatched config and weight file is caught at load rather than
producing quietly wrong hidden states.

Branch distillation is a **training-only** objective. It adds no inference-time
structure, which is why both band checkpoints load through exactly the same
code path and differ only in their weights.

## What the drafter actually computes

Useful when reading `angelslim_drafter.py` or debugging a shape error.

**Aux injection.** The target exposes nine intermediate hidden states. In
`banded_mix_fc` they are grouped into three bands; each band is collapsed to one
stream by a learned softmax over its members (`band{i}_mix_logits`); the three
resulting streams go through three RMSNorms (`fc_norm`) and one `fc`
(3H→H, 576×1728). In `fused_fc` there are simply three streams and no mixing.
Either way the drafter's layer 0 sees one H-wide vector, so the two modes differ
only in front of `fc`.

**The EAGLE-3 shift.** At training time `h[t] = aux[t]` (unshifted),
`x[t] = ids[t+1]`, and the draft predicts `ids[t+2]`. This falls out of
`online_eagle3_trainer.prepare_data_for_draft_model`, which left-shifts
`input_ids` and `target_logits` but *not* `hidden_states`. Get this wrong by one
and acceptance collapses to near zero without any error.

**Vocabulary mapping.** The drafter has a 32000-entry draft vocabulary over the
target's 49280. `d2t` is an **additive delta**, not a lookup table:
`target_tok = draft_tok + d2t[draft_tok]` (values span [3, 17280]). Treating it
as a direct index is the single most common way to get τ ≈ 1.000 — it was in
fact the first bug found in the earlier attempt at this port.

**Aux layer indexing.** `hidden_states[i + 1]` is the output of layer `i`,
because HF index 0 is the embedding output. This matches AngelSlim's vLLM hook,
which records `all_hs[layer_id]` as the output of `layer_id`.

## The six things that had to be bridged

1. **Importing `angelslim` under Python 3.9.** `angelslim/__init__.py` reaches
   `qat/modules/quantizer.py`, whose class body evaluates a py3.10-only
   `X | None` annotation and raises `TypeError` at import time.
   `load_angelslim_draft_module()` registers empty stub `ModuleType` objects for
   `angelslim`, `angelslim.compressor`, … so the package `__init__` never runs
   and only the one draft module is loaded from source. The checkout is found by
   walking up from the module (HiViS is vendored inside it); `ANGELSLIM_ROOT`
   overrides.

2. **Config translation, tf5 → tf4.** The checkpoints were written by
   transformers 5.x, which nests RoPE under `rope_parameters`. transformers 4.54
   reads `config.rope_theta` and **silently falls back to `10000.0`** if it is
   missing — a wrong RoPE base that degrades acceptance without erroring.
   `DrafterConfig` translates `rope_parameters.rope_theta` → `rope_theta` and
   `rope_type` → `rope_scaling` explicitly. For all our checkpoints the correct
   values are `rope_theta == 100000.0`, `rope_scaling is None`.

3. **`banded_mix_fc` sizing.** AngelSlim's base drafter sizes `fc`/`fc_norm`
   from `len(aux_hidden_states_layer_ids)`, which is 9 here and gives a shape
   mismatch against a 3H→H `fc`. `DrafterConfig` therefore advertises the
   **band** count (3) to the constructor, and the real 9-layer list is restored
   on the built module afterwards for the aux-capture hook.

4. **Aux hidden states out of a SmolVLM target.** HiViS calls the target for
   `last_hidden_state` only. The angelslim path sets `output_hidden_states=True`
   and concatenates `hidden_states[i + 1] for i in aux_layer_ids`.

5. **A tree-capable SmolVLM target.** Stock `Idefics3` uses transformers' own
   `LlamaModel`, which neither accepts HiViS's legacy list-based KV cache nor
   honours `tree_mask` — so tree verification silently degenerated to something
   that ran and produced plausible-looking wrong numbers. At load time the text
   tower is replaced by HiViS's forked `modeling_llama_kv.LlamaModel` (eager
   attention), weights copied across with a strict check. Prefill goes through a
   new `smolvlm_input_embeds()` reproducing Idefics3's `get_image_features` +
   `inputs_merger` merge, since the stock forward cannot be called for embeds
   alone here. `naivegenerate` was routed through the *same* backbone, so the
   speedup denominator is not measured on a different kernel path from the
   numerator.

6. **A KV cache that fits.** `initialize_past_key_values` preallocates a static
   cache sized at `config.max_position_embeddings`. That is 8k on SmolVLM and
   **128k on Qwen2.5-VL** — 28 layers × 2 × 4 kv-heads × 128000 × 128 in bf16 is
   **6.84 GiB**, on top of ~16 GiB of 7B weights, so HiViS's own published
   checkpoint OOMs on a 24 GiB card before emitting a token. It now takes an
   optional `max_len` cap (`--cache_len`), defaulting to upstream behaviour and
   validated against `max_position_embeddings`.

`topK_genrate` returns 5 values in AngelSlim and 4 in HiViS; the mixin trims it
and raises if the 5th is ever non-`None`.

## Environment

Nothing here is tied to a particular environment manager. What the code needs is
a **Python 3.9+ interpreter with HiViS's dependencies** (`requirements.txt`,
notably `transformers` 4.5x — *not* 5.x, see
[Which environment runs what](#which-environment-runs-what)) and `HiViS/`
importable.

```bash
# uv
uv venv && uv pip install -r requirements.txt && source .venv/bin/activate

# conda / mamba
conda create -n hivis python=3.9 && conda activate hivis
pip install -r requirements.txt

# or any interpreter that already satisfies requirements.txt
```

Then either run from `HiViS/`:

```bash
python run_angelslim_eval.py ...
```

or from anywhere:

```bash
PYTHONPATH=/path/to/HiViS python /path/to/HiViS/run_angelslim_eval.py ...
```

Only one path is environment-sensitive, and it is discovered rather than
hardcoded: `ANGELSLIM_ROOT`, where the `angelslim` source tree lives. Left unset
it is found by walking up from `angelslim_drafter.py`, so it needs setting only
if HiViS and AngelSlim are checked out apart. `--base` and `--draft` take local
paths and Hub repo ids interchangeably.

## CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--base` | `HuggingFaceTB/SmolVLM-256M-Instruct` | target VLM; **must** be the one the drafter was trained against |
| `--draft` | *(required)* | drafter checkpoint directory or Hub repo id |
| `--draft_method` | `angelslim_eagle3` | `angelslim_eagle3` \| `hivis` \| `vispec` \| `eagle` |
| `--dataset` | `omnidocbench` | any of the thirteen below |
| `--n` | `4` | number of prompts |
| `--max_new_tokens` | `256` | generation budget per prompt |
| `--total_token` | `60` | total nodes in the draft tree |
| `--depth` | `5` | tree depth |
| `--top_k` | `10` | candidates per level |
| `--max_pixels` | *(processor default)* | cap image resolution in pixels; see [Running a 7B target](#running-a-7b-target-on-a-24-gib-card) |
| `--cache_len` | *(model max)* | cap on the preallocated static KV cache; must be ≥ prompt + `--max_new_tokens` |
| `--device` | `cuda:0` | `device_map` for the target |
| `--naive` | off | autoregressive baseline — the speedup denominator |
| `--out` | none | write the result JSON here |

**Tree vs chain.** `--total_token 60 --depth 5 --top_k 10` is EAGLE-2 tree
decoding. `--total_token 5 --depth 4 --top_k 1` is a linear chain of K=4 — one
candidate per level, so the tree degenerates to a path. Our drafters were
trained for linear decode, so chain is the mode the method targets; tree is
reported because HiViS uses it and because it is faster in wall-clock terms.

`--base` must match the drafter: a drafter is tied to its target's hidden size,
vocabulary and `d2t` map, so mixing them is a shape error at best and silent
garbage at worst. The script does not guess it.

**Sizing `--cache_len`.** It must cover prompt + `--max_new_tokens`, and prompt
length is dominated by *visual* tokens, which differ enormously between targets:
one OmniDocBench page is a few hundred tokens for SmolVLM-256M but **~16.3k for
Qwen2.5-VL**, which does not downsample image tokens. The runner checks each
prompt against the cap before generating and names the number to use, rather
than failing deep inside `KVCache` as `start (0) + length (16298) exceeds
dimension size (4096)`. Cost is modest — 20480 on Qwen2.5-VL-7B is about 1.1 GiB
against the 6.84 GiB the 128k default wants.

### Running a 7B target on a 24 GiB card

A 7B target on full-resolution document images does not fit, and the reason is
worth understanding before working around it. Three costs stack:

| | on Qwen2.5-VL-7B, OmniDocBench |
|---|---|
| weights (bf16) | ~16 GiB |
| preallocated KV cache at the 128k default | 6.84 GiB |
| eager attention over a 16.3k-token prompt | quadratic, several GiB more |

The third is the binding one and cannot be tuned away: this fork's Llama runs
**eager attention only** (bridge #5), so the attention matrix is materialised in
full. Shortening the prompt is the only lever, and `--max_pixels` is it —
Qwen2.5-VL emits roughly one visual token per 28×28 pixels, so

```bash
--max_pixels 1605632   # ~2k visual tokens instead of ~16.3k
--cache_len 8192
```

brings a 7B target inside 24 GiB. **This changes what the model sees**, so it
changes acceptance: hold `--max_pixels` fixed across every run you intend to
compare, and state it beside the numbers. Comparing a capped run against an
uncapped one is not a valid comparison.

If you have more memory, prefer leaving `--max_pixels` unset and raising
`--cache_len` instead — that changes nothing about the model's inputs.

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

### Their model on our benchmarks, and ours on theirs

`--draft_method` selects the drafter family and `--dataset` the rows. The two
axes are independent, so all four combinations are one code path — HiViS's
`prepare_inputs` was already model-agnostic, building messages and handing them
to `model.processor`, so Qwen2.5-VL and SmolVLM need no separate branch.

```bash
# HiViS's published drafter, on the benchmark our vLLM eval uses.
# --cache_len is required here: see bridge #6.
python run_angelslim_eval.py --draft_method hivis \
    --base Qwen/Qwen2.5-VL-7B-Instruct \
    --draft Irisssme/HiViS-Qwen2.5-VL-7B-Instruct \
    --dataset omnidocbench --n 40 --max_new_tokens 1024 \
    --max_pixels 1605632 --cache_len 8192 \
    --total_token 60 --depth 5 --top_k 10 --out ../results_hivis_omni.json

# ViSpec's, likewise
python run_angelslim_eval.py --draft_method vispec \
    --base Qwen/Qwen2.5-VL-3B-Instruct \
    --draft JLKang/ViSpec-Qwen2.5-VL-3B-Instruct \
    --dataset omnidocbench --max_pixels 1605632 --cache_len 8192 ...

# ...and ours on one of theirs
python run_angelslim_eval.py --draft $CKPT --dataset DocVQA ...
```

Both directions are exercised: `branch-distill-top1-w01` on `omnidocbench`
reproduces its reference acceptance, and `Irisssme/HiViS-Qwen2.5-VL-7B-Instruct`
generates on `omnidocbench` under `--max_pixels 1605632 --cache_len 8192`
(τ ≈ 3.4 on a two-prompt wiring check — a smoke test, not a result; the
published table below is the only measured comparison).

**What this licenses you to compare.** Two drafters on the same rows in this
harness share a backend and a decoding loop, so their τ and their
speedup-over-own-baseline are comparable. Their **absolute tok/s is not**:
HiViS's checkpoint drives Qwen2.5-VL-7B and ours drives SmolVLM-256M, so those
columns are 7B numbers beside 256M numbers. Always quote speedup against each
model's own `--naive` run in the same configuration.

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
does — identical rows is the whole reason a PyTorch τ can be held next to a vLLM
τ. Their OCR prompt is copied byte for byte from that script for the same
reason. Do not "fix" either to use the seed-42 shuffle the HiViS benchmarks use;
that silently breaks the cross-backend comparison and every number below.

## Output format

`--out` writes:

```json
{
  "tau": 2.9123,          // mean acceptance length over all rounds
  "rounds": 1346,         // total speculation rounds across all prompts
  "tok_per_s": 72.80,     // total tokens / total wall time
  "per_prompt": [{"tokens": 1028, "rounds": 322, "tau": 3.193, "time": 12.70}],
  "cfg": {}               // every CLI flag, for provenance
}
```

A **round** is one speculate-then-verify cycle. τ is
`mean(accepted_count + 1)` over rounds — the `+1` is the token the target
produces even when every draft token is rejected, so τ = 1.000 means "no draft
token ever accepted" and is exactly the autoregressive baseline. Speedup is
`tok_per_s` divided by the `--naive` run's `tok_per_s` **in the same
configuration**; the harness does not compute it for you, because the
denominator has to be a run you actually made.

## Results

OmniDocBench, 40 prompts, greedy, `max_new_tokens=1024`, SmolVLM-256M-Instruct
target, one RTX A5000. All rows share one backbone and one loop.

| | τ (accept. length) | tok/s | speedup | avg output |
|---|---:|---:|---:|---:|
| autoregressive baseline | 1.000 | 34.17 | 1.00× | 565.4 |
| **chain** · `baseline_1layer` (stock EAGLE-3) | 2.589 | 65.92 | 1.93× | 549.1 |
| chain · `banded-mix-fc-3.1` | 2.727 | 68.47 | 2.00× | 502.3 |
| chain · `branch-distill-top1-w01` | **2.913** | **72.33** | **2.12×** | 594.4 |
| **tree** · `baseline_1layer` (stock EAGLE-3) | 3.610 | 85.08 | 2.49× | 532.3 |
| tree · `branch-distill-top1-w01` | 3.907 | 86.13 | 2.52× | 512.1 |
| tree · `banded-mix-fc-3.1` | 3.927 | 89.30 | 2.61× | 524.7 |

Against the stock EAGLE-3 baseline:

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
  A τ from this benchmark is meaningless without the output length beside it,
  and a short run reads as a regression when it is not. This is what made the
  first chain measurements look broken; they were measured at 256 tokens.
- **Both band models beat stock EAGLE-3 in both modes**, by 5–12 % acceptance.
  That gap is the one to claim.
- **The ranking *between the two band models* flips between modes.** Chain puts
  `branch-distill` ahead of `banded-mix-fc-3.1` by +0.186; tree puts it behind
  by 0.020, which is noise. Branch distillation trains exactly the top-1 quality
  that a 10-candidate first tree level dilutes, so its advantage is real under
  chain decoding — what it was trained for — and washes out under tree. Worth
  saying out loud rather than reporting only the mode that favours the method.
- **Acceptance gains compress into speedup gains.** +12.5 % τ buys +10 % tok/s
  under chain; under tree, +8 % τ buys ~1 %. Eager attention leaves the target
  forward dominant, so a longer accepted run has less to win back.

### Reproducing the table

```bash
W=../dataset/angelslim-smolvlm-eagle3-artifacts/weight
for ckpt in baseline_1layer banded-mix-fc-3.1 branch-distill-top1-w01; do
  for mode in "60 5 10 tree" "5 4 1 chain"; do
    set -- $mode
    python run_angelslim_eval.py --draft $W/$ckpt/checkpoint-66466 \
      --dataset omnidocbench --n 40 --max_new_tokens 1024 \
      --total_token $1 --depth $2 --top_k $3 --out ../res_$4_$ckpt.json
  done
done
python run_angelslim_eval.py --draft $W/baseline_1layer/checkpoint-66466 \
  --dataset omnidocbench --n 40 --max_new_tokens 1024 --naive --out ../res_naive.json
```

`--naive` ignores the drafter, so any checkpoint gives the same baseline.

## Verification

Three independent checks, in increasing scope:

1. **Weight load.** `build_drafter` raises on any missing key and reports
   unexpected ones; `gist_norm` is the only key filtered as known-unused. All
   three checkpoints load with zero missing and zero unexpected.
2. **Single-step parity.** `tools/eagle3_step_parity.py` feeds identical aux
   hidden states and identical input ids to AngelSlim's *training-stack* drafter
   and to this *inference-stack* drafter. They produce the **same argmax token**.
   It also asserts `softmax(band{i}_mix_logits)` matches
   `banded_aux_mix_weights.json`.
3. **End-to-end acceptance against vLLM.** Under chain decoding (K=4, 1024
   output tokens) vLLM reports τ = 2.87 for `branch-distill-top1-w01`; this
   harness reports 2.913. Acceptance is not degraded by the port, which was the
   requirement the port had to meet.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError` inside `qat/modules/quantizer.py` on import | running under py3.9 with `angelslim/__init__` executed | you bypassed `load_angelslim_draft_module`; import the drafter through it |
| `RuntimeError: Could not find an angelslim checkout` | HiViS checked out apart from AngelSlim | set `ANGELSLIM_ROOT` |
| τ ≈ 1.000, nothing ever accepted | `d2t` treated as a lookup instead of an additive delta, or the EAGLE-3 shift off by one | see [What the drafter actually computes](#what-the-drafter-actually-computes) |
| τ plausible but well below the vLLM reference | measured at too few output tokens on OmniDocBench | compare at equal `--max_new_tokens`; 1024 for the vLLM reference |
| `size mismatch for fc.weight` | band count vs aux-layer count | `DrafterConfig` must advertise bands, not the 9-layer list |
| `AttributeError: 'Idefics3Config' object has no attribute 'num_hidden_layers'` | config resolution | fixed in `kv_cache.py`; check you are on this fork |
| `AttributeError: 'list' object has no attribute 'get_seq_length'` | stock transformers `LlamaModel` got HiViS's legacy list cache | the text tower swap did not happen; check `is_smolvlm` |
| `ValueError: LlamaModel does not support sdpa` | forked Llama is eager-only | `_attn_implementation = "eager"` |
| `torch.OutOfMemoryError` in `initialize_past_key_values` | 128k static cache on a 7B target | `--cache_len 8192` |
| `torch.OutOfMemoryError` in the target forward | eager attention over a ~16k-token visual prompt | `--max_pixels 1605632`, and read the section above before comparing |
| `--cache_len N is too small for prompt M` | visual tokens; a Qwen2.5-VL page is ~16.3k | raise `--cache_len` to the number the message names |
| `start (0) + length (N) exceeds dimension size` | same, but from inside `KVCache` | you are on an older revision without the preflight check |
| `ValueError: too many values to unpack (expected 4)` | `topK_genrate` returned AngelSlim's 5-tuple | the `HiViSInterfaceMixin` trim was bypassed |
| missing `rotary_emb.inv_freq` keys | derived buffers, not weights | already filtered from the strict check |

## Limitations

- HiViS's forked Llama runs **eager attention only**, so absolute tok/s here is
  lower than vLLM's and the reported speedups are conservative. Rows are fair
  against each other and against HiViS's own baselines; they are not directly
  comparable to vLLM numbers. It also puts a hard quadratic ceiling on prompt
  length — see [Running a 7B target](#running-a-7b-target-on-a-24-gib-card).
- Only `banded_mix_fc` and `fused_fc` injection modes are wired up.
  `build_drafter` raises `NotImplementedError` on anything else rather than
  loading a drafter that silently computes the wrong thing.
- SmolVLM / Idefics3 targets only for the *angelslim* path. The Qwen2.5-VL
  branch raises if aux capture is requested. HiViS's and ViSpec's own drafters
  run on Qwen normally — that path is upstream's and untouched.
- Single-layer drafters only; `build_drafter` raises if `num_hidden_layers != 1`.
- Batch size 1 throughout, greedy only in the measured configurations.

## Known and deliberately out of scope

Recorded so nobody re-chases them from this directory:

- **The two band checkpoints converge in performance here** more than they do
  elsewhere. Known; a separate problem from the port.
- **The target model's output differs on this server** from another where these
  checkpoints were evaluated. Cause not established. It does not affect the
  port's faithfulness, which is verified against the *local* target.
- **vLLM run-to-run drift** on the same checkpoint and benchmark. Out of scope
  here; this harness is deterministic given fixed rows and greedy decoding.
