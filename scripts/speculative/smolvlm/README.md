# SmolVLM speculative decoding (data gen + Eagle3 train)

Target: `HuggingFaceTB/SmolVLM-256M-Instruct`

SmolVLM does **not** use a separate trainer. Online training uses the shared VLM Eagle3 path; SmolVLM only specializes **config → data → target forward**.

---

## Where SmolVLM is wired in training

| Stage | File | What it does |
|---|---|---|
| Entry script | `scripts/speculative/smolvlm/train_eagle3_vlm_online.sh` | → `tools/train_eagle3_online.py` |
| Draft dims / type | `angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3.json` | `hidden_size=576`, `num_hidden_layers=1`, `target_model_type=smolvlm` |
| Chat / loss headers | `.../train/data/chat_templates.py` | `User:` / `Assistant:` |
| Dataset + collate | `.../online_dataset_builder.py` → `OnlineSmolVLMDatasetBuilder` | expand image tokens; `image_url` → image |
| Pixel batching | `.../train/data/data_utils.py` → `VLMSmolVLMDataCollatorWithPadding` | `pixel_values` + `pixel_attention_mask` |
| Target HS extract | `.../target/target_model_wrapper.py` → `VLMTransformersBackend` | hook `model.text_model`; **no** `image_grid_thw` |
| Embed key | `.../train/models/model_utils.py` | `model.text_model.embed_tokens.weight` |
| Train step | `.../trainer/online_eagle3_trainer.py` → `OnlineVLMEagle3Trainer` | **shared** with Qwen/Hunyuan VLM |

Draft model itself is the shared `Eagle3LlamaForCausalLM` — there is no SmolVLM-specific draft class.

Aliases: `target_model_type` / chat template accept both `smolvlm` and `idefics3`.

---

## Multi-layer draft (not config-only)

Draft config already has:

```json
"num_hidden_layers": 1
```

But `Eagle3LlamaForCausalLM` currently builds **one** layer only:

- `angelslim/compressor/speculative/train/models/draft/llama_eagle3.py`
  - `__init__`: `self.midlayer = LlamaDecoderLayeremb(config)` (single module)
  - `encode_layers`: forwards through that one `midlayer`

Changing `num_hidden_layers` to `2` / `4` in the JSON alone will **not** stack layers.

To add multi-layer support, change `Eagle3LlamaForCausalLM`:

1. `self.midlayer` → `nn.ModuleList([LlamaDecoderLayeremb(config) for _ in range(config.num_hidden_layers)])`
2. Loop those layers in `encode_layers` (and the gradient-checkpointing path in the same file)

Reference for a stacked draft: `qwen_dflash.py` / `qwen_dflare.py` already use `nn.ModuleList` over `config.num_hidden_layers`.

Note: `--training_time_test_length` / trainer `length` is speculative **decode steps**, not draft depth.

---

## Recommended: dp=4 (4 GPU replicas) for data gen

```bash
GPU_NUM=4 BASE_PORT=6000 bash scripts/speculative/smolvlm/run_vllm_server.sh

MAX_CLIENTS=4 NUM_THREADS=32 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
```

Starts vLLM on ports **6000–6003** (one replica per GPU); the generator load-balances across them.

---

## 1. Start vLLM server

```bash
bash scripts/speculative/smolvlm/run_vllm_server.sh
# defaults: GPU_NUM=4 BASE_PORT=6000
# override e.g. GPU_NUM=1 MODEL_LOCAL_PATH=HuggingFaceTB/SmolVLM-256M-Instruct
```

## 2. Generate target-model samples

Uses `dataset/mixed_text_vl_36/mixed_text_vl_36.jsonl` (`openai_vl` format) by default.

```bash
bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
# defaults: MAX_CLIENTS=4 NUM_THREADS=32
# optional: DATA_NAME_OR_PATH=... OUTPUT_DIR=... MAX_TOKENS=512
```

Output: `dataset/smolvlm_256m_target_gen/data_*.jsonl`

## 3. Online Eagle3 training

```bash
bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
# defaults: 1 GPU, 1 epoch on dataset/smolvlm_256m_target_gen/data_0-36.jsonl
# optional: CUDA_VISIBLE_DEVICES=0,1 NPROC=2 SAMPLE_NUM=8 OUTPUT_DIR=...
```

Draft config: `angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3.json`
