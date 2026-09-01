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
| `run_angelslim_eval.py` | **new** — acceptance-length / throughput harness |

Everything else under `hivis/` is upstream, byte for byte.

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

## Running

```bash
conda activate hivis
cd HiViS

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

`--dataset` accepts `opendatalab/OmniDocBench` (default) and `MMMU/MMMU`.

## Results

OmniDocBench, 40 prompts, greedy, `max_new_tokens=1024`, SmolVLM-256M-Instruct
target, single A-series GPU. All rows share one backbone and one loop.

| | τ (accept. length) | tok/s | speedup | avg output |
|---|---:|---:|---:|---:|
| autoregressive baseline | 1.000 | 34.17 | 1.00× | 565.4 |
| tree · `branch-distill-top1-w01` | 3.907 | 86.13 | 2.52× | 512.1 |
| tree · `banded-mix-fc-3.1` | 3.927 | 89.30 | 2.61× | 524.7 |
| chain · `branch-distill-top1-w01` | 2.913 | 72.33 | 2.12× | 594.4 |
| chain · `banded-mix-fc-3.1` | 2.727 | 68.47 | 2.00× | 502.3 |

### Reading the numbers

- **Acceptance rises with generation length on OmniDocBench.** Position 0 alone
  is 0.150, the first 96 tokens 0.432, the first 256 0.571, the first 1024
  0.714. It is an OCR task: hard boilerplate up front, easy transcription after.
  Any τ quoted from this benchmark is meaningless without the output length
  next to it. A short run reads as a regression when it is not.
- **The ranking flips between modes.** Chain puts `branch-distill` ahead by
  +0.186; tree puts it behind by 0.020 (noise). Branch distillation trains
  exactly the top-1 quality that a 10-candidate first tree level dilutes.
  Worth saying out loud in the paper rather than reporting only the mode that
  favours the method.

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
