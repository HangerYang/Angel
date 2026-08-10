# public_weight — compact SmolVLM Eagle drafts

Minimal publishable weights. Reload with a local/HF **SmolVLM** path.

| Pack | Source train run | What’s stored | What’s taken from SmolVLM |
|---|---|---|---|
| `hawk_warmup/` | `output/smolvlm_256m_hawk_warmup` | layers + fuse + norm + lm_head + vocab maps | `embed_tokens` |
| `hawk_nccl/` | `output/smolvlm_256m_hawk_nccl` | same as hawk_warmup | `embed_tokens` |
| `real_hawk_lora/` | `output/smolvlm_256m_real_hawk_nccl` | LoRA A/B + fuse + norm + lm_head + vocab maps | `embed_tokens` + frozen base layers `[1,14,26]` |

## Export (from local full checkpoints)

```bash
python public_weight/export_public_weights.py
# optional overrides:
#   --hawk_ckpt output/smolvlm_256m_hawk_warmup/warmup_end
#   --hawk_nccl_ckpt output/smolvlm_256m_hawk_nccl/checkpoint-66466
#   --real_hawk_ckpt output/smolvlm_256m_real_hawk_nccl/checkpoint-66466
```

## Load hawk

```bash
python public_weight/load_hawk.py \
  --smolvlm HuggingFaceTB/SmolVLM-256M-Instruct \
  --pack public_weight/hawk_warmup \
  --save_full ~/tmp/hawk_full

python public_weight/load_hawk.py \
  --smolvlm HuggingFaceTB/SmolVLM-256M-Instruct \
  --pack public_weight/hawk_nccl \
  --save_full ~/tmp/hawk_nccl_full
```

```python
from public_weight.load_hawk import load_hawk_draft
model = load_hawk_draft("HuggingFaceTB/SmolVLM-256M-Instruct", "public_weight/hawk_warmup")
```

## Load real_hawk LoRA

**Important:** the compact pack and train checkpoints use LoRA module keys
(`*.base.weight`, `*.lora_A/B`). Plain `Eagle3LlamaForCausalLM.from_pretrained`
will **not** load those — Linear weights stay random and adapters are dropped.

For HF / vLLM eval, write a **merged** full folder:

```bash
python public_weight/load_real_hawk_lora.py \
  --smolvlm HuggingFaceTB/SmolVLM-256M-Instruct \
  --pack public_weight/real_hawk_lora \
  --save_full ~/tmp/real_hawk_full \
  --save_lora_only ~/tmp/real_hawk_lora_only.safetensors
```

Then HF-load the **merged** dir:

```python
from pathlib import Path
from angelslim.compressor.speculative.train.models.draft.llama_eagle3 import (
    Eagle3LlamaForCausalLM,
)
model = Eagle3LlamaForCausalLM.from_pretrained(
    str(Path.home() / "tmp" / "real_hawk_full")
)
```

Or load train / pack paths in one shot:

```python
from public_weight.load_real_hawk_lora import (
    load_real_hawk_lora,
    load_real_hawk_checkpoint,
)

# from compact pack (returns LoRA model or merged if merge_lora=True)
model, lora_only = load_real_hawk_lora(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "public_weight/real_hawk_lora",
    merge_lora=True,
)

# from train checkpoint OR --save_full dir (HF-compatible when merge_lora=True)
model = load_real_hawk_checkpoint(
    "output/smolvlm_256m_real_hawk_nccl/checkpoint-66466",
    merge_lora=True,
)
```
