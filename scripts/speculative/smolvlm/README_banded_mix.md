# Progressive banded aux mix (SmolVLM-256M Eagle3)

`eagle_aux_injection_mode: progressive_banded_mix` — a progressive-staged draft
where the target's aux hidden states are **not** taken one-per-draft-layer.
Instead N aux streams (N > num_draft_layers) are grouped into `num_hidden_layers`
contiguous **bands**, and each band is collapsed to one stream by a learned
softmax mix. Draft layer *i* consumes the mix of band *i*.

Rationale: picking one target layer per draft layer is a hard, hand-tuned choice.
Banding lets training learn *which* layer inside a depth region matters, without
widening the draft's input.

Config: [`smolvlm-256m-eagle3-progressive-banded-mix-uninit.json`](../../../angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-banded-mix-uninit.json)

```json
"eagle_aux_injection_mode": "progressive_banded_mix",
"aux_hidden_states_layer_ids":       [2, 4, 8, 10, 15, 18, 20, 26, 28],
"eagle_aux_hidden_state_layer_ids":  [3, 5, 9, 11, 16, 19, 21, 27, 29],
"eagle_aux_layer_bands":     [[2, 4, 8, 10], [15, 18, 20], [26, 28]],
"eagle_aux_band_init_layer_ids": [2, 15, 26]
```

`uninit` = no `draft_layer_init_from_target`, so the draft starts random and the
band mix is the only thing under study.

The learned mixture is dumped at every save to
`<checkpoint>/banded_aux_mix_weights.json`. From the 2-epoch run, the mix is
sharply peaked rather than uniform — band0 puts 0.76 on layer 2, band2 puts 0.88
on layer 26 — i.e. training does express a preference inside each band.

## Where the code lives

| Piece | File |
|---|---|
| Draft model (band construction, mixing, forward) | `angelslim/compressor/speculative/train/models/draft/llama_eagle3.py` |
| Mix-weight dump on save | `angelslim/compressor/speculative/train/trainer/eagle3_trainer.py` |
| vLLM-facing config prep | `scripts/speculative/smolvlm/prepare_draft_config_for_vllm_eval.py` |
| vLLM runtime support | `third_party/patches/vllm-v0.25.0-eagle3-progressive-staged.patch` |

## Reproducing on another server

### 1. vLLM

The banded decode path is **not** in stock vLLM — it arrives through the tracked
patch. `link_local_vllm.sh` rsyncs a stock wheel over `third_party/vllm` (which
is gitignored) and then runs `apply_vllm_patches.sh`, so the patch file is the
only transport between machines. Anything edited directly in `third_party/vllm`
without being folded back into the patch does not travel.

```bash
git clone --branch v0.25.0 --depth 1 https://github.com/vllm-project/vllm.git third_party/vllm
bash third_party/install_local_vllm.sh                  # CUDA 13.0 (default)
# VLLM_CUDA=12.6 bash third_party/install_local_vllm.sh # CUDA 12.6
bash third_party/link_local_vllm.sh                     # overlay .so + apply patches
```

Verify banded support actually landed:

```bash
grep -c progressive_banded_mix third_party/vllm/vllm/model_executor/models/llama_eagle3.py   # expect 14
```

### 2. Data

`dataset/smolvlm_256m_target_gen_mixed_70k70k/` is **not** in git (train.jsonl is
132,943 rows, eval.jsonl 6,997). It is target-generated, so it must be rebuilt:

```bash
# a) mixed ShareGPT + LLaVA-Instruct source (LLaVA image paths must be absolute
#    and present on this machine)
python dataset/build_mixed_text_vl_jsonl.py --help

# b) resample answers with SmolVLM-256M through a local vLLM OpenAI server
bash scripts/speculative/smolvlm/run_vllm_server.sh
MAX_CLIENTS=4 NUM_THREADS=32 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
```

Draft acceptance is sensitive to the target-generated text, so a rebuilt dataset
will not reproduce the published numbers to the third decimal.

### 3. Train

```bash
TRAIN_MODE=nccl NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-banded-mix-uninit.json \
OUTPUT_DIR=output/smolvlm-256m-eagle3-progressive-banded-mix-uninit \
NUM_TRAIN_EPOCHS=2 SAVE_STRATEGY=epoch EVAL_STRATEGY=epoch \
LOAD_FROM_CACHE_FILE=true TARGET_HS_WARMUP_STEPS=0 \
bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

### 4. Eval

Per dataset, at the settings the published numbers were taken at:

```bash
for DS in lmms-lab/textvqa MMMU/MMMU Lin-Chen/MMStar opendatalab/OmniDocBench \
          HuggingFaceH4/MATH-500 lmms-lab/COCO-Caption lmms-lab/chartqa AI4Math/MathVista; do
  DRAFT_MODEL=output/smolvlm-256m-eagle3-progressive-banded-mix-uninit/checkpoint-66466 \
  DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-banded-mix-uninit.json \
  DATASET="${DS}" NUM_PROMPTS=80 NUM_SPEC_TOKENS=4 MAX_NUM_SEQS=1 TEMP=0 \
  bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
done
```

Results land at `<DRAFT_MODEL>/eval/<dataset>/results.jsonl`.

`prepare_draft_config_for_vllm_eval.py` runs first and stamps
`num_aux_hidden_states` (= 9, the flattened band total, **not** the draft depth)
plus the two band fields into the checkpoint's `config.json`. Without that step
vLLM builds the wrong number of aux streams.

## Measured result

2-epoch run, `checkpoint-66466`, temp=0, K=4 spec tokens, N=80 prompts,
`MAX_NUM_SEQS=1`, early exit disabled. `mean_acceptance_length` from the real
vLLM speculative-decode path:

| | textvqa | MMMU | MMStar | OmniDocBench | MATH-500 | COCO-Caption | chartqa | mathvista | mean |
|---|---|---|---|---|---|---|---|---|---|
| eagle3 (regular) | 1.937 | 2.358 | 2.048 | 2.000 | 3.095 | 2.331 | 2.012 | 1.672 | **2.182** |
| progressive | 2.015 | 2.536 | 2.039 | 2.471 | 3.445 | 2.385 | 2.025 | 1.656 | **2.322** |
| **banded_mix_uninit** | 2.055 | 2.633 | 2.124 | 2.258 | 3.468 | 2.545 | 2.143 | 1.869 | **2.387** |

Banded mix beats both the regular Eagle3 baseline (+0.205) and plain progressive
staged (+0.065) on the 8-dataset mean, and wins on all 8 datasets against
regular Eagle3 (7 of 8 against progressive — it loses OmniDocBench, 2.258 vs 2.471).
It is not the best variant measured overall — per-layer FC (2.464) is ahead of
it — but it is the strongest result that needs no extra draft parameters beyond
the band mix logits.
