# SmolVLM speculative decoding (data gen + Eagle3 train)

Target: `HuggingFaceTB/SmolVLM-256M-Instruct`

SmolVLM does **not** use a separate trainer. Online training uses the shared VLM Eagle3 path; SmolVLM only specializes **config → data → target forward**.

---

## Target ↔ draft communication (training)

Online Eagle3 never trains SmolVLM. Each step:

```
batch (input_ids, loss_mask, pixel_*)
        │
        ▼
┌───────────────────────────────┐
│ TARGET (frozen SmolVLM)       │  VLMTransformersBackend
│  forward + output_hidden_states
│  pick 3 aux layers → concat   │  → hidden_states [B,S, 3×576]
│  logits                       │  → target_logits [B,S,V]
└───────────────┬───────────────┘
                │ tensors in-process (same GPU step)
                ▼
┌───────────────────────────────┐
│ DRAFT (Eagle3Llama, trainable)│
│  embed(input_ids)  (frozen)   │  same token ids as target
│  fc: 3×576 → 576              │
│  layers[0]: cat(embed, hs)    │  2H QKV Eagle layer
│  layers[1..]: standard H      │  optional multi-layer stack
│  lm_head → draft vocab        │
│  CE vs softmax(target_logits) │
│  repeat `length` draft steps  │
└───────────────────────────────┘
```

### Alignment

Token embeds and target HS are **position-aligned on the same `input_ids` sequence**. Draft embeds those tokens; target HS/logits come from the same prompt (plus pixels on the target only). At image-token slots the target stream may carry injected vision features while the draft still has `embed(image_token_id)` — same index, not always the same vector.

### What is passed

| Tensor | From → to | Shape (SmolVLM-256M) | Meaning |
|---|---|---|---|
| `input_ids`, `attention_mask`, `loss_mask` | collator → both | `[B,S]` | shared tokens; loss only on assistant spans |
| `pixel_values`, `pixel_attention_mask` | collator → **target only** | tiles × C×H×W | draft never sees pixels |
| `hidden_states` | target → draft | `[B,S,1728]` = `3×576` | concat of 3 aux target layers |
| `target_logits` | target → draft | `[B,S,vocab]` | teacher next-token dist (then sliced by draft vocab `t2d`) |
| `position_ids` | target hook → draft | optional | may be `None` for SmolVLM; draft can fall back |

No separate RPC/cache file in online mode: one Python training step, target forward then draft forward.

### How many intermediate (aux) layers?

**3 target layers**, defaulted in `_get_default_aux_layer_ids` (or set explicitly in draft config):

```text
[1,  total_layers//2 - 1,  total_layers - 4]
```

For SmolVLM-256M (`text_config.num_hidden_layers = 30`):

| Slot | Index | Role |
|---|---|---|
| early | **1** | near bottom |
| mid | **14** | middle |
| late | **26** | near top (not the very last) |

Indexing into HF `outputs.hidden_states` uses `layer_id + 1` to skip the embedding output, then the three tensors are **concatenated on the feature dim** → `3 × 576 = 1728`. Draft `fc` projects that back to `576` before / inside the Eagle layer path.

Config fields: `aux_hidden_states_layer_ids` (AngelSlim train; applied as HF `hs[id+1]`) and `eagle_aux_hidden_state_layer_ids` (vLLM indices; for SmolVLM-30L use `[2, 15, 27]` to match train’s `[1, 14, 26]`).

### Inside the draft

- **Layer 0**: token embeds + fused target HS → dual RMSNorm → concat `2H` → QKV; residual/MLP on the HS stream (Eagle).
- **Layers 1…N−1** (if `num_hidden_layers > 1`): standard Llama `H` blocks; fused HS is **not** re-injected.

Optional init via `draft_layer_init_from_target: [L0, L1, …]` (length = N): copy that target layer’s weights into each draft layer. Layer 0 puts target QKV on the **HS half** of the `2H` matrix and zeros the embed half.

---

## Code map (where each process lives)

| Process | Code location |
|---|---|
| Launch train | `scripts/speculative/smolvlm/train_eagle3_vlm_online.sh` → `tools/train_eagle3_online.py` |
| Draft config (width, `num_hidden_layers`, `target_model_type`) | `angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3.json` |
| Build draft module | `.../models/draft/llama_eagle3.py` → `Eagle3LlamaForCausalLM` |
| Load frozen target + processor | `.../models/target/target_model_wrapper.py` → `VLMTransformersBackend.load_model` |
| Target forward + aux HS / logits | same file → `get_hidden_states_and_logits` / `_extract_auxiliary_hidden_states` / `_get_default_aux_layer_ids` |
| Hook language tower (embeds / pos) | `VLMTransformersBackend._register_language_model_hook` → `model.model.text_model` for SmolVLM |
| Pixel kwargs (no `image_grid_thw`) | `VLMTransformersBackend._build_vlm_forward_kwargs` |
| Online VLM train step (call target, then draft loop) | `.../trainer/online_eagle3_trainer.py` → `OnlineVLMEagle3Trainer.prepare_data_for_draft_model` |
| Speculative draft loss loop (`length` steps) | `.../trainer/eagle3_trainer.py` → `Eagle3Trainer` training step / encode loop |
| Dataset tokenize + loss mask | `.../data/dataset_builder/online_dataset_builder.py` → `OnlineSmolVLMDatasetBuilder` |
| Batch pixels from `image_paths` | `.../data/data_utils.py` → `VLMSmolVLMDataCollatorWithPadding` |
| Chat headers for loss mask | `.../data/chat_templates.py` → `smolvlm` |
| Embed weight key | `.../models/model_utils.py` → `MODEL_TYPE_PARAM_MAP["smolvlm"]` |
| Draft vocab prune cache | `output_dir/vocab_mapping_cache.pt` (from `train_eagle3_online.py`) |

---

## Where SmolVLM is wired (summary)

| Stage | File | What it does |
|---|---|---|
| Entry script | `scripts/speculative/smolvlm/train_eagle3_vlm_online.sh` | → `tools/train_eagle3_online.py` |
| Draft dims / type | `configs/smolvlm-256m-eagle3.json` | `hidden_size=576`, `num_hidden_layers`, `target_model_type=smolvlm` |
| Chat / loss headers | `chat_templates.py` | `User:` / `Assistant:` |
| Dataset + collate | `OnlineSmolVLMDatasetBuilder` + `VLMSmolVLMDataCollatorWithPadding` | image tokens; pixels on the fly |
| Target HS extract | `VLMTransformersBackend` | hook `text_model`; `pixel_*` |
| Train step | `OnlineVLMEagle3Trainer` | **shared** with other VLMs |

Aliases: `smolvlm` / `idefics3`.

---

## Multi-layer **draft** (not aux layers)

Do not confuse:

- **3 aux target layers** — HS concat into `fc` (unchanged).
- **Draft depth** — `num_hidden_layers` in draft config; `layers` is an `nn.ModuleList` (layer 0 = Eagle `2H`, rest = standard `H`). Checkpoints save as `layers.{i}.*` (vLLM also remaps legacy `midlayer.` → `layers.0.`).

Example 3-layer draft init from target:

```json
"num_hidden_layers": 3,
"draft_layer_init_from_target": [8, 16, 26]
```

`--training_time_test_length` / trainer `length` = speculative **decode steps**, not draft depth and not aux count.

---

## Recommended: dp=4 (4 GPU replicas) for data gen

```bash
GPU_NUM=4 BASE_PORT=6000 bash scripts/speculative/smolvlm/run_vllm_server.sh

MAX_CLIENTS=4 NUM_THREADS=32 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
```

---

## 1. Start vLLM server

```bash
bash scripts/speculative/smolvlm/run_vllm_server.sh
```

## 2. Generate target-model samples

```bash
bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
```

Output: `dataset/smolvlm_256m_target_gen/data_*.jsonl`

## 3. Online Eagle3 training

```bash
bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

Draft config: `angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3.json`
