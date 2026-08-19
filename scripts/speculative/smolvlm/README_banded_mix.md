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
`<checkpoint>/banded_aux_mix_weights.json`. Read it against the **initial**
distribution, not against uniform: this run was spike-initialised (see below),
so every band started at 95-98% on one layer and training moved weight *away*
from it. band1 diffused hardest (0.965 -> 0.756 -> 0.488 across epochs,
spreading onto layers 18 and 20), band0 moderately, band2 least. The other
members of each band do contribute.

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

## Trying other layer setups

"More layers" is two different experiments, and only one of them is cheap.

| config | draft layers | aux streams | concat width | mix params | total params |
|---|---|---|---|---|---|
| `...-banded-mix-uninit` | 3 | 9 | 5184 | 9 | 59.10M |
| `...-banded-mix-L4` | 4 | 12 | 6912 | 12 | 63.20M |
| `...-banded-mix-dense14` | 3 | 14 | 8064 | 14 | 59.10M |

- **More target aux layers (denser bands)** costs one float per added layer. The
  band mix collapses each band to a single stream, so the draft does not grow:
  9 -> 14 streams left it at 59.10M. This is the axis banded mix exists for —
  `progressive_per_layer_fc` cannot widen this way without growing its `nH->H`
  projections.
- **More draft layers** costs +4.1M params *and* a slower draft step, paid on
  every speculation. Judge it on end-to-end throughput, not
  `mean_acceptance_length` — acceptance can rise while wall-clock speedup falls.

### Fields to set

Four fields must stay mutually consistent or `Eagle3LlamaForCausalLM.__init__`
raises:

```json
"num_hidden_layers": 4,
"eagle_aux_layer_bands": [[2,4,6],[9,11,13],[16,18,20],[23,26,28]],
"aux_hidden_states_layer_ids":      [2,4,6,9,11,13,16,18,20,23,26,28],
"eagle_aux_hidden_state_layer_ids": [3,5,7,10,12,14,17,19,21,24,27,29],
"eagle_aux_band_init_layer_ids": [2,9,16,23]
```

- `len(eagle_aux_layer_bands) == num_hidden_layers`, every band non-empty
- `aux_hidden_states_layer_ids` must equal the **flattened bands in order** — it
  is compared element-wise, not as a set
- `eagle_aux_hidden_state_layer_ids[i] == aux_hidden_states_layer_ids[i] + 1`
- ids are 0..29 (SmolVLM-256M's text tower is 30 layers, hidden size 576)

`eagle_aux_band_init_layer_ids` **matters** — it picks the layer each band
starts concentrated on:

```python
logits = torch.zeros(len(band), dtype=torch.float32)
logits[band.index(int(init_layer_id))] = 4.0     # spike init
```

These are pre-softmax logits, so a 4.0 spike in a 4-member band means
`e**4 / (e**4 + 3) = 0.948` initial weight on that layer. The id must be a
member of its own band or init raises. The `banded-mix-uninit` run used
`[2, 15, 26]`; `uninit` in that name refers to `draft_layer_init_from_target`,
not to these logits.

An all-zeros alternative gives `softmax([0,...,0]) = 1/n`, i.e. a genuinely
uniform start that lets training find the mix unaided. That variant is **not**
on this branch — changing the init changes what a new run is comparable to, so
switch deliberately and note it when reporting numbers.

Nothing needs changing on the eval side —
`prepare_draft_config_for_vllm_eval.py` derives `num_aux_hidden_states` from the
bands.

### Generating and validating a new variant

```python
import json, pathlib
base = json.load(open("angelslim/compressor/speculative/train/configs/"
                      "smolvlm-256m-eagle3-progressive-banded-mix-uninit.json"))

def make(name, num_layers, bands):
    c = dict(base)
    c["num_hidden_layers"] = num_layers
    c["eagle_aux_layer_bands"] = bands
    flat = [i for b in bands for i in b]
    c["aux_hidden_states_layer_ids"] = flat
    c["eagle_aux_hidden_state_layer_ids"] = [i + 1 for i in flat]
    c["eagle_aux_band_init_layer_ids"] = [b[0] for b in bands]
    pathlib.Path(f"angelslim/compressor/speculative/train/configs/{name}.json"
                 ).write_text(json.dumps(c, indent=2) + "\n")

make("smolvlm-256m-eagle3-progressive-banded-mix-L4",
     4, [[2,4,6],[9,11,13],[16,18,20],[23,26,28]])
```

Confirm it builds before spending GPU-hours:

```bash
python - <<'EOF'
import json
from transformers import LlamaConfig
from angelslim.compressor.speculative.train.models.draft.llama_eagle3 import (
    Eagle3LlamaForCausalLM,
)
name = "smolvlm-256m-eagle3-progressive-banded-mix-L4"
d = json.load(open(f"angelslim/compressor/speculative/train/configs/{name}.json"))
m = Eagle3LlamaForCausalLM(
    LlamaConfig(**{k: v for k, v in d.items() if k != "architectures"})
)
print("OK", sum(p.numel() for p in m.parameters()) / 1e6, "M params")
EOF
```

Then train as in section 3, swapping `DRAFT_MODEL_CONFIG_PATH` and `OUTPUT_DIR`.

### What to watch

- Smoke-eval a new **draft depth** with `NUM_PROMPTS=2` first. Depth != 3 is
  verified to build in the training path but has not been run through vLLM's
  banded decode.
- Read `banded_aux_mix_weights.json` after epoch 1 and compare against the
  band's *initial* weight, not against uniform. Under spike init the question is
  how far weight diffuses away from the seeded layer; if a denser band stays
  pinned near its 0.95 start, the extra layers are not buying anything and the
  experiment has answered itself early.
- Keep `eagle_aux_band_init_layer_ids` fixed across variants you intend to
  compare. Seeding a band at a layer the previous run never contained makes the
  comparison two-variable.

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
