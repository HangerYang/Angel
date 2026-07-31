# Local vLLM (`third_party/vllm`) — Eagle3 debug guide

Portable local checkout of **vLLM `v0.25.0`** for editing / debugging speculative decoding.
Works on any server path and any env (conda, uv, venv).

## 1. Setup (every machine / every env)

Paths and compiled `.so` files are **not** portable. On each server:

```bash
# From the AngelSlim repo root on THAT machine
git clone --branch v0.25.0 --depth 1 \
  https://github.com/vllm-project/vllm.git third_party/vllm   # if missing

# Activate THAT server's Python env, then:
uv pip install vllm==0.25.0          # or: pip install vllm==0.25.0
bash third_party/link_local_vllm.sh  # overlays .so, applies AngelSlim patches, wires imports
source third_party/env.sh            # PYTHONPATH=<repo>/third_party/vllm
```

`link_local_vllm.sh` also runs `third_party/apply_vllm_patches.sh`, which applies
tracked patches under `third_party/patches/` (e.g. SmolVLM/Idefics3 Eagle3).
Do **not** rely on hand-edited files inside `third_party/vllm` — those do not
travel to other servers. Edit the `.patch` files in AngelSlim instead.

Verify:

```bash
python -c "import vllm, os; print(vllm.__version__, os.path.realpath(vllm.__file__))"
# expect: 0.25.0  .../<any-path>/third_party/vllm/vllm/__init__.py

python -c "from vllm.model_executor.models.idefics3 import Idefics3ForConditionalGeneration as C; from vllm.model_executor.models.interfaces import SupportsEagle3 as S; assert S in C.__mro__, 'SmolVLM Eagle3 patch missing — run bash third_party/apply_vllm_patches.sh'; print('SmolVLM Eagle3: OK')"
```

Qwen3-VL + Eagle3 uses **Model Runner V2** (`Using V2 Model Runner` in logs).
Edit / break under `third_party/vllm/vllm/v1/worker/gpu/`, not the older `v1/spec_decode/llm_base_proposer.py` path.

---

## 2. Regular Eagle3 inference

```bash
# optional but recommended after setup
source third_party/env.sh

python tools/vllm_offline_eagle3_vlm_batch.py \
  --target_model Qwen/Qwen3-VL-2B-Instruct \
  --draft_model AngelSlim/Qwen3-VL-2B-Instruct_eagle3 \
  --use_eagle \
  --num_spec_tokens 4 \
  --num_prompts 80 \
  --temp 0 \
  --max_num_seqs 1 \
  --output_len 1024 \
  --output_file results/qwen3-vl-2b-eagle3-textvqa.jsonl
```

Baseline (no draft):

```bash
python tools/vllm_offline_eagle3_vlm_batch.py \
  --target_model Qwen/Qwen3-VL-2B-Instruct \
  --num_prompts 80 \
  --temp 0 \
  --max_num_seqs 1 \
  --output_len 1024 \
  --output_file results/qwen3-vl-2b-baseline-textvqa.jsonl
```

---

## 3. Eagle3 inference with debug (SSH / tmux / no IDE)

`--debug` runs EngineCore **in-process** so breakpoints inside local vLLM hit.  
`--pdb` makes `breakpoint()` open **ipdb**.

```bash
source third_party/env.sh

# 1) Uncomment the BP you care about (see §4)
# 2) Run a tiny job:
python tools/vllm_offline_eagle3_vlm_batch.py \
  --target_model Qwen/Qwen3-VL-2B-Instruct \
  --draft_model AngelSlim/Qwen3-VL-2B-Instruct_eagle3 \
  --use_eagle \
  --num_spec_tokens 4 \
  --num_prompts 1 \
  --output_len 32 \
  --debug --pdb \
  --output_file results/debug-eagle3-vlm.jsonl
```

ipdb keys: `n` next · `s` step · `c` continue · `l` list · `p expr` · `bt` · `q` quit

---

## 4. Eagle breakpoints (Model Runner V2)

Uncomment `# breakpoint()` at the site you need. Search the tree for `ANGELSLIM EAGLE BP#`.

```text
Target forward ──► aux hidden states ──► draft propose (step 0) ──► draft loop (step 1..K-1)
     BP#1                                      BP#2                         BP#3
```

### BP#1 — Target model processes the input

**File:** `third_party/vllm/vllm/v1/worker/gpu/model_runner.py`  
**Where:** `execute_model`, immediately before `self.model(**model_inputs)` (eager path)

**Look at:**
- `model_inputs` — `input_ids` / `positions` / `inputs_embeds`
- `input_batch`
- After `c`ontinue: `aux_hidden_states` (Eagle3 layers) stored on `execute_model_state`

### BP#2 — Target sends hidden states to the draft model

**File:** `third_party/vllm/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`  
**Where:** `propose`, after `combine_hidden_states` / `self.hidden_states[:num_tokens].copy_(...)`

**Look at:**
- `aux_hidden_states` — list of target layer HS (Eagle3)
- `hidden_states` — combined tensor fed to draft
- `self.hidden_states[:num_tokens]` — draft input buffer
- `self.method` — should be `"eagle3"`

Called from `sample_tokens` → `self.speculator.propose(...)` in the same `model_runner.py`.

### BP#3 — Draft model loops; previous draft token → next draft step

**File:** same `autoregressive/speculator.py`  
**Where:** `_multi_step_decode`, top of `for step in range(1, self.num_speculative_steps)`

**Look at:**
- `step` / `self.current_draft_step`
- `self.draft_tokens[:num_reqs]` — tokens drafted so far
- `self.input_buffers.input_ids[:num_reqs]` — next-token ids into draft
- `self.hidden_states[:num_reqs]` — HS carried into the next draft forward  

Token write-back happens in `update_draft_inputs(...)` at the end of `_generate_draft`.

> **Note:** Draft **step 0** (first draft token after the target HS handoff) runs in `_prefill` / first `_run_model` inside `propose`, not in this loop. BP#2 is the handoff into that first draft forward; BP#3 is steps `1 .. num_spec_tokens-1`.

---

## 5. Layout cheat-sheet

| Path | Role |
|------|------|
| `third_party/vllm/` | Full upstream repo (`pyproject.toml`, `csrc/`) |
| `third_party/vllm/vllm/v1/worker/gpu/model_runner.py` | Target forward (BP#1) |
| `…/gpu/spec_decode/autoregressive/speculator.py` | Eagle draft propose + loop (BP#2, BP#3) |
| `…/gpu/spec_decode/eagle/` | EagleSpeculator thin wrapper |
| `third_party/env.sh` | Portable `PYTHONPATH` |
| `third_party/link_local_vllm.sh` | Per-machine `.so` overlay + import wiring |
