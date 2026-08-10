# public_weight — compact SmolVLM Eagle drafts

Minimal publishable weights. Reload with a local/HF **SmolVLM** path.

| Pack | Source train run | What’s stored | What’s taken from SmolVLM |
|---|---|---|---|
| `hawk_warmup/` | `output/smolvlm_256m_hawk_warmup` | layers + fuse + norm + lm_head + vocab maps | `embed_tokens` |
| `real_hawk_lora/` | `output/smolvlm_256m_real_hawk_nccl` | LoRA A/B + fuse + norm + lm_head + vocab maps | `embed_tokens` + frozen base layers `[1,14,26]` |

## Export (from local full checkpoints)

```bash
python public_weight/export_public_weights.py
# optional overrides:
#   --hawk_ckpt output/smolvlm_256m_hawk_warmup/warmup_end
#   --real_hawk_ckpt output/smolvlm_256m_real_hawk_nccl/checkpoint-66466
```

## Load hawk

```bash
python public_weight/load_hawk.py \
  --smolvlm HuggingFaceTB/SmolVLM-256M-Instruct \
  --pack public_weight/hawk_warmup \
  --save_full /tmp/hawk_full
```

```python
from public_weight.load_hawk import load_hawk_draft
model = load_hawk_draft("HuggingFaceTB/SmolVLM-256M-Instruct", "public_weight/hawk_warmup")
```

## Load real_hawk LoRA

```bash
# full draft with LoRA modules + optional LoRA-only file
python public_weight/load_real_hawk_lora.py \
  --smolvlm HuggingFaceTB/SmolVLM-256M-Instruct \
  --pack public_weight/real_hawk_lora \
  --save_lora_only /tmp/real_hawk_lora_only.safetensors \
  --save_full /tmp/real_hawk_full

# merged hawk-shaped weights for vLLM:
python public_weight/load_real_hawk_lora.py \
  --smolvlm HuggingFaceTB/SmolVLM-256M-Instruct \
  --merge_lora --save_full /tmp/real_hawk_merged
```

```python
from public_weight.load_real_hawk_lora import load_real_hawk_lora
model, lora_only = load_real_hawk_lora(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "public_weight/real_hawk_lora",
)
```

`lora_only` is `{layers.*.*.lora_A/B: Tensor}` — the adapter-only payload.
