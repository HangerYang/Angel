# SmolVLM speculative decoding (data gen + Eagle3 train + vLLM eval)

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

Config fields for **which target layers feed the fusion HS** (Eagle3 default early/mid/late):

| Field | Role | SmolVLM-30L example |
|---|---|---|
| `aux_hidden_states_layer_ids` | AngelSlim **train** (applied as HF `hs[id+1]`) | `[1, 14, 26]` |
| `eagle_aux_hidden_state_layer_ids` | **vLLM** eval indices | `[2, 15, 27]` |

If only `aux_hidden_states_layer_ids` is set, train scripts auto-set `eagle_aux` to `id+1`. Override either list in the draft JSON to change fusion layers.

### Inside the draft

- **Layer 0**: token embeds + fused target HS → dual RMSNorm → concat `2H` → QKV; residual/MLP on the HS stream (Eagle).
- **Layers 1…N−1** (if `num_hidden_layers > 1`): standard Llama `H` blocks; fused HS is **not** re-injected.

Optional init:

```json
"draft_layer_init_from_target": [14, 16, 26]
```

| Draft piece | Source |
|---|---|
| Layer 0 emb path (`input_layernorm` + QKV emb half) | random (not copied) |
| Layer 0 HS path (`hidden_norm` + QKV HS half + o_proj/MLP/post-norm) | `draft_layer_init_from_target[0]` |
| Layers 1…N−1 | full copy from `draft_layer_init_from_target[i]` |

---

## Code map (where each process lives)

| Process | Code location |
|---|---|
| Launch train | `scripts/speculative/smolvlm/train_eagle3_vlm_online.sh` → `tools/train_eagle3_online.py` |
| Launch vLLM Eagle3 eval | `scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh` → `tools/vllm_offline_eagle3_vlm_batch.py` |
| Sync draft config for vLLM (1-/multi-layer + aux ids) | `scripts/speculative/smolvlm/prepare_draft_config_for_vllm_eval.py` |
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
| Eval script | `scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh` | → prepare draft config → `tools/vllm_offline_eagle3_vlm_batch.py` |
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
"draft_layer_init_from_target": [14, 16, 26],
"aux_hidden_states_layer_ids": [1, 14, 26],
"eagle_aux_hidden_state_layer_ids": [2, 15, 27]
```

`--training_time_test_length` / trainer `length` = speculative **decode steps** unrolled in the train loop (default **7** → logs `acc_0…acc_6`), not draft depth and not aux count. Same unroll is used for HF `--eval_data_path`; real acceptance length is measured only via vLLM offline eval.

Draft config index: `angelslim/compressor/speculative/train/configs/README_smolvlm.md`.

### Hawk fusion (progressive-only experiment)

Config: `angelslim/.../configs/smolvlm-256m-hawk.json`

Hawk is **progressive like** `progressive_staged`, but each inject is **H add-fusion** (not 2H concat):

```text
# requires len(aux)==num_hidden_layers (default 3)
step 0 (target aux):
  L0: ê0 = HS₀  @ w1 + embed @ w2  → H-block
  L1: ê1 = HS₁₃ @ w1 + h0    @ w2  → H-block
  L2: ê2 = HS₂₅ @ w1 + h1    @ w2  → H-block

steps 1+ (same-depth draft outs; same as progressive):
  injects ← (h0_prev, h1_prev, h2_prev)
```

| | Progressive Eagle | Hawk |
|---|---|---|
| Aux layout | 1 stream / draft layer | same |
| Per-layer combine | **concat → 2H** QKV | **`w1`/`w2` add → H** |
| Draft blocks | 2H Eagle | standard **H** Llama |
| Train loop | Eagle online | same |
| Steps 1+ feedback | draft `h0/h1/h2` | same |
| vLLM eval | same progressive patch | same (`fuse_w1`/`fuse_w2`) |

Do **not** set `progressive_staged` for hawk — use `"eagle_aux_injection_mode": "hawk"`.

```bash
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-hawk.json \
  OUTPUT_DIR=output/smolvlm_256m_hawk \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

DRAFT_MODEL=output/smolvlm_256m_hawk/checkpoint-* \
  DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-hawk.json \
  bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh

# Miracle mode (oracle GT target-HS): fused_fc / progressive / hawk.
# A) target-only GT tokens  B) capture aux tape  C) timed eagle with tape[pos]
MIRACLE_MODE=1 \
  DRAFT_MODEL=output/smolvlm_256m_hawk/checkpoint-30000 \
  DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-hawk.json \
  bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
```

`DRAFT_MODEL` must contain `config.json` (or a `checkpoint-*` child that does —
the eval script auto-picks the latest). Miracle requires `max_num_seqs=1`
(enforced). Look for `MIRACLE_MODE=1`, `eagle_miracle_mode: True`, and
`Eagle3 miracle mode`. Timed metrics are **phase C only** (GT capture is not
timed). Re-apply the progressive patch after pull (includes `eagle_miracle.py`).

### Progressive staged injection (experiment)

Config: `angelslim/.../configs/smolvlm-256m-eagle3-progressive.json`

Instead of early `fc(3H→H)` and Eagle-only on draft L0, each draft layer is 2H and injects one predecessor aux stream:

```text
inject train ids: [0, 13, 25]   # inputs to target layers 1 / 14 / 26
init weights from: [1, 14, 26]  # consumer layers (left half from predecessor)
mode: "eagle_aux_injection_mode": "progressive_staged"
```

```text
step 0 (target aux):
  L0: residual=HS₀;  attn=cat(norm(embed), norm(HS₀))
  L1: residual=h0;   attn=cat(norm(h0),    norm(HS₁₃))
  L2: residual=h1;   attn=cat(norm(h1),    norm(HS₂₅)) → lm_head

steps 1+ (same-depth draft outs; no shifted target aux):
  L0: residual=h0_prev;  attn=cat(norm(embed), norm(h0_prev))
  L1: residual=h0';       attn=cat(norm(h0'),    norm(h1_prev))
  L2: residual=h1';       attn=cat(norm(h1'),    norm(h2_prev))
```

Train / eval:

```bash
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive.json \
  OUTPUT_DIR=output/smolvlm_256m_eagle3_progressive \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

DRAFT_MODEL=output/smolvlm_256m_eagle3_progressive/checkpoint-* \
  DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive.json \
  bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
```

Requires the local vLLM progressive patch
(`third_party/patches/vllm-v0.25.0-eagle3-progressive-staged.patch`).

After `git pull`, if an older progressive patch was already applied under
`third_party/vllm`, reset those files then re-apply (otherwise `git apply`
can fail on a dirty tree):

```bash
cd third_party/vllm
git checkout -- vllm/envs.py \
  vllm/model_executor/models/llama_eagle3.py \
  vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
rm -f vllm/model_executor/models/eagle_miracle.py
cd ../..
bash third_party/apply_vllm_patches.sh
source third_party/env.sh
```

If those files were never patched on this machine, you can skip the
`git checkout` and only run `apply_vllm_patches.sh` (or
`bash third_party/link_local_vllm.sh`). Look for log line
`Eagle3 progressive_staged enabled` at draft load.

Stock `fused_fc` remains the default when the mode field is omitted.

**Train/eval feedback (progressive):** after the first draft token, both train and
vLLM reuse per-layer draft outs (`L0←h0`, `L1←h1`, `L2←h2`) — not shifted /
stale target aux. Residual materialize before L1+ inject is still required in
vLLM (`h_prev = mlp_out + residual`).

---

## Recommended: dp=4 (4 GPU replicas) for data gen

```bash
GPU_NUM=4 BASE_PORT=6000 bash scripts/speculative/smolvlm/run_vllm_server.sh

MAX_CLIENTS=4 NUM_THREADS=32 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
```

### Single-GPU vLLM (data gen only)

```bash
GPU_NUM=1 CUDA_VISIBLE_DEVICES=0 bash scripts/speculative/smolvlm/run_vllm_server.sh
MAX_CLIENTS=1 NUM_THREADS=8 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
```

Same target samples / same jsonl schema — only wall-clock slower (~4× for the same dataset vs 4 replicas). **Not** a training-step multiplier; training equivalence is controlled by `EQUIV_NPROC` / `grad_accum` above.

---

## 1. Start vLLM server

```bash
bash scripts/speculative/smolvlm/run_vllm_server.sh
```

## 2. Generate target-model samples

Default input: `dataset/preprocessed/mixed_sharegpt_llava665k_70k70k.jsonl`
(70k ShareGPT text + 70k LLaVA VL; images under `dataset/preprocessed/llava_images/`).

vLLM servers must already be up (4 replicas on 6000–6003 by default):

```bash
bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
```

Output: `dataset/smolvlm_256m_target_gen_mixed_70k70k/data_*.jsonl`

Then train with:

```bash
TRAIN_DATA_PATH=dataset/smolvlm_256m_target_gen_mixed_70k70k \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```


## 3. Online Eagle3 training

```bash
bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

Draft config: `angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3.json`

Default `SAVE_STRATEGY=epoch` writes the draft under `OUTPUT_DIR` (needed for eval).  
For a throwaway smoke run: `SAVE_STRATEGY=no bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh`.

### Launch modes (NCCL issues / single GPU)

| Mode | Command | Same math as NCCL 4-GPU? |
|---|---|---|
| torchrun + NCCL | `NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 DIST_BACKEND=nccl bash ...` | reference |
| torchrun + Gloo | `NPROC=4 ... DIST_BACKEND=gloo bash ...` | **Yes** — same DDP, different collective backend (slower) |
| plain python 1 GPU | `LAUNCH=python EQUIV_NPROC=4 CUDA_VISIBLE_DEVICES=0 bash ...` | **Yes if** `grad_accum=EQUIV_NPROC` (auto) so effective batch matches |

```bash
# Gloo 4-GPU (drop-in when NCCL is broken; same steps / same effective batch)
DIST_BACKEND=gloo NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh

# Plain python, 1 GPU, matched to a 4-GPU DDP run
LAUNCH=python EQUIV_NPROC=4 CUDA_VISIBLE_DEVICES=0 NUM_PROC=1 \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

**Equivalence (do you need 4× steps?)**

Effective batch ≈ `per_device_bs × NPROC × grad_accum` (for `LAUNCH=python`, `NPROC=1`).

| Setup | effective batch | optimizer steps / epoch (dataset size N, bs=1) |
|---|---|---|
| NCCL or Gloo, `NPROC=4`, `grad_accum=1` | 4 | ≈ N/4 |
| `LAUNCH=python`, `grad_accum=4` (`EQUIV_NPROC=4`) | 4 | ≈ N/4 — **same** |
| `LAUNCH=python`, `grad_accum=1` (no EQUIV) | 1 | ≈ N — **not** the same (4× more updates, smaller batch) |

So: Gloo 4-GPU ≡ NCCL 4-GPU (same step count). Plain python does **not** need 4× epochs if you set `EQUIV_NPROC=4` (uses grad accum). Without that, you get a different optimization trajectory.

Also lower HF datasets workers if mp is flaky: `NUM_PROC=1`.

---

## 4. Offline vLLM Eagle3 eval

Entry script (SmolVLM defaults → shared batch tool):

```bash
# After training, DRAFT_MODEL points at the saved checkpoint dir
DRAFT_MODEL=output/smolvlm_256m_eagle3_online \
  bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh

# Baseline (no draft)
USE_EAGLE=0 bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh

# Local jsonl smoke
DATASET=dataset/smolvlm_256m_target_gen/data_0-36.jsonl NUM_PROMPTS=4 \
  DRAFT_MODEL=output/smolvlm_256m_eagle3_online \
  bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
```

### Miracle mode (oracle GT target-HS)

Upper-bound eval: draft steps inject **ground-truth target aux HS** along the
target-generated trajectory (not draft feedback, not frozen last-verify).
Works for `fused_fc`, `progressive_staged`, and `hawk`. Requires `TEMP=0` and
`max_num_seqs=1` (enforced).

**Phases (automatic):** A) target-only GT tokens → B) capture aux tape →
C) timed eagle with `tape[pos]` inject. Metrics are from **C only**.

| Phase | What | Timed? |
|---|---|---|
| A | Target-only generate → `gt_tokens.json` | no |
| B | Eagle **capture**: force draft onto GT path; record verify aux at absolute positions → `{i:05d}.pt` | no |
| C | Eagle **use**: inject `tape[pos]` each draft step | **yes** |

**After `git pull`**, reset then re-apply the progressive patch (ships `eagle_miracle.py`):

```bash
cd third_party/vllm
git checkout -- vllm/envs.py \
  vllm/model_executor/models/llama_eagle3.py \
  vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
rm -f vllm/model_executor/models/eagle_miracle.py
cd ../..
bash third_party/apply_vllm_patches.sh
source third_party/env.sh
```

**Smoke test** (local jsonl, 2 prompts, short decode):

```bash
MIRACLE_MODE=1 TEMP=0 CUDA_VISIBLE_DEVICES=0 NUM_PROMPTS=2 OUTPUT_LEN=32 \
  DATASET=dataset/smolvlm_256m_target_gen/data_0-36.jsonl \
  DRAFT_MODEL=output/smolvlm_256m_hawk/checkpoint-30000 \
  DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-hawk.json \
  OUTPUT_FILE=results/miracle_smoke_test.jsonl \
  MIRACLE_HS_DIR=results/miracle_smoke_hs \
  bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
```

Expect: `captured 2/2 tapes`, phase C finishes (no rotary crash), and
`results/miracle_smoke_hs/{00000,00001}.pt` are dense along the GT length.
Optional debug: `VLLM_EAGLE_MIRACLE_DEBUG=1` logs each capture write.

**Hawk example:**

```bash
MIRACLE_MODE=1 TEMP=0 \
  DRAFT_MODEL=output/smolvlm_256m_hawk/checkpoint-30000 \
  DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-hawk.json \
  OUTPUT_FILE=results/smolvlm-256m-hawk-miracle.jsonl \
  bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
```

**Fused Eagle3 example:**

```bash
MIRACLE_MODE=1 TEMP=0 \
  DRAFT_MODEL=output/smolvlm_256m_eagle3_online \
  DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3.json \
  OUTPUT_FILE=results/smolvlm-256m-eagle3-miracle.jsonl \
  bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
```

Expect logs: `MIRACLE_MODE=1`, `eagle_miracle_mode: True`,
`Eagle3 miracle mode (capture|use)`. Optional: `MIRACLE_HS_DIR=...` to keep
tapes. (`ASSISTANCE_MODE` is a deprecated alias for `MIRACLE_MODE`.)

**Notes / expectations**

- Capture records against **target** positions (`input_batch.positions`), not
  draft buffers. Offline req ids are `{prompt_index}-{hex}`; warmup ids are
  ignored so USE init does not skip hawk draft-HS refresh.
- Output defaults to `{DRAFT_MODEL}/eval/{data}_miracle/results.jsonl`.
- **Hawk / progressive:** train uses target aux on draft step 0 and **draft
  feedback** on steps 1+. Miracle still injects GT target tape on 1+ as an
  oracle upper bound; that is OOD vs hawk training, so mean acceptance may be
  close to (or slightly below) non-miracle hawk. Prefer **fused Eagle3** when
  you want a clear target-shift upper bound.

Before calling vLLM, the eval script runs:

`scripts/speculative/smolvlm/prepare_draft_config_for_vllm_eval.py`

That helper reads **1- or multi-layer** draft settings from the checkpoint (and fills gaps from the train draft JSON):

| Field | Role |
|---|---|
| `num_hidden_layers` | Draft depth (`1` = single Eagle layer; `>1` = Eagle layer 0 + standard layers) |
| `eagle_aux_hidden_state_layer_ids` | **vLLM** target aux layers for fused HS |
| `aux_hidden_states_layer_ids` | AngelSlim train ids; if vLLM list missing → `eagle_aux = id+1` |
| `draft_layer_init_from_target` | Optional; length must equal `num_hidden_layers` |

Underlying runner (shared with Qwen3-VL / Hunyuan):

`tools/vllm_offline_eagle3_vlm_batch.py`

Requires local vLLM overlay with the **tracked** SmolVLM Eagle3 patch applied
(see below). Stock vLLM raises `Model does not support EAGLE3 interface`.

### Portability (other servers)

Do **not** hand-edit only `third_party/vllm` on one machine — that tree is not
what other servers get. Eagle3 for SmolVLM is shipped as:

`third_party/patches/vllm-v0.25.0-smolvlm-eagle3.patch`

On every server (clones vLLM v0.25.0 if needed):

```bash
# CUDA 13.0 (default)
bash third_party/install_local_vllm.sh

# CUDA 12.6 server (source build; needs toolkit / nvcc)
VLLM_CUDA=12.6 bash third_party/install_local_vllm.sh

source third_party/env.sh
```

Or only re-apply patches: `bash third_party/apply_vllm_patches.sh`  
Details: `third_party/README.md`.

---

## Where to update vLLM eval (SmolVLM)

When Eagle3 eval breaks or you need to change behavior, these are the touch points:

| What to change | File | Where / what |
|---|---|---|
| **SmolVLM eval launcher** (defaults, dataset, draft path) | `scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh` | Env vars → CLI for the batch tool |
| **Draft config sync** (1-/multi-layer, aux ids) | `scripts/speculative/smolvlm/prepare_draft_config_for_vllm_eval.py` | Ensures `config.json` has `num_hidden_layers` + `eagle_aux_hidden_state_layer_ids` |
| **Shared offline batch / metrics** | `tools/vllm_offline_eagle3_vlm_batch.py` | `LLM(..., speculative_config=...)`, acceptance metrics, datasets |
| **Portable Eagle3 enablement (edit this, not a dirty vllm tree)** | `third_party/patches/vllm-v0.25.0-smolvlm-eagle3.patch` | Applied by `apply_vllm_patches.sh` / `link_local_vllm.sh` |
| **Target: advertise Eagle3 + set aux layers** *(after patch)* | `third_party/vllm/.../idefics3.py` | `SupportsEagle3`; `set_aux_hidden_state_layers` → `model.text_model` |
| **SmolVLM class** (inherits Idefics3) | `third_party/vllm/.../smolvlm.py` | Thin subclass; Eagle3 hooks live on Idefics3 |
| **Wire target embed/HS into draft** *(after patch)* | `third_party/vllm/.../eagle/utils.py` | `load_eagle_model`: `text_model` is LlamaModel directly |
| **Apply aux layer ids from draft config** | `third_party/vllm/.../eagle/eagle3_utils.py` | reads `eagle_aux_hidden_state_layer_ids` |
| **Draft forward (1- or multi-layer)** | `third_party/vllm/.../llama_eagle3.py` | `num_hidden_layers` blocks; layer 0 = 2H Eagle |
| **Propose / multi-step draft loop** | `third_party/vllm/.../autoregressive/speculator.py` | Model Runner V2 Eagle3 propose path |
| **Miracle GT-HS (capture/use)** | `third_party/vllm/.../eagle_miracle.py` + progressive patch | Oracle tape inject; `MIRACLE_MODE=1` |
| **Train draft JSON** | `angelslim/.../configs/smolvlm-256m-eagle3.json` | `num_hidden_layers`, aux ids, optional init |

Setup: `bash third_party/link_local_vllm.sh && source third_party/env.sh` (see `third_party/README.md`).