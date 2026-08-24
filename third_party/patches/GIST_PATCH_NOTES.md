# vLLM Oracle Gist Patches

After `bash third_party/sync_vllm_latest.sh`, re-apply these edits to enable vLLM decode with oracle gist conditioning.

## 1. Add eagle_gist.py (new file)

**File**: `third_party/vllm/vllm/model_executor/models/eagle_gist.py`

Copy the file from this repo's `third_party/patches/eagle_gist.py.txt` (or see ORACLE_GIST_SETUP.md for a reference implementation).

This module:
- Reads a reference results.jsonl at engine init
- Encodes all reference outputs with Qwen3-Embedding-0.6B
- Maps request IDs to reference rows
- Arms the per-request gist vector each forward pass

Enable with environment variables:
```
VLLM_EAGLE_GIST_MODE=1
VLLM_EAGLE_GIST_REF=/path/to/results.jsonl
VLLM_EAGLE_GIST_ENCODER=Qwen/Qwen3-Embedding-0.6B
VLLM_EAGLE_GIST_DEVICE=cuda:0
```

## 2. Extend llama_eagle3.py

**File**: `third_party/vllm/vllm/model_executor/models/llama_eagle3.py`

### A. Add import at the top
After the existing imports, add:
```python
from . import eagle_gist
```

### B. Extend LlamaAttentionLayer.__init__ (around line 59-70)

After the existing QKV setup, add:
```python
# Oracle gist, "qkv"/"both" injection: layer 0 takes a third stream
self.gist_qkv_stream = (
    layer_idx == 0
    and bool(getattr(config, "gist_conditioning", False))
    and str(getattr(config, "gist_injection", "qkv")) in ("qkv", "both")
)
if self.gist_qkv_stream:
    qkv_input_size += self.hidden_size
```

And add after the layer_idx assignment (around line 105):
```python
self.gist_norm = (
    RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    if self.gist_qkv_stream
    else None
)
self.gist_projector = None
```

### C. Extend forward method (around line 130-140)

Find the layer-0 concat block:
```python
elif self.layer_idx == 0:
    embeds = self.input_layernorm(embeds)
    hidden_states, residual = self._residual_norm(hidden_states=hidden_states)
    hidden_states = torch.cat([embeds, hidden_states], dim=-1)
```

Replace with:
```python
elif self.layer_idx == 0:
    embeds = self.input_layernorm(embeds)
    hidden_states, residual = self._residual_norm(hidden_states=hidden_states)
    streams = [embeds, hidden_states]
    if self.gist_qkv_stream:
        streams.append(self._gist_stream(hidden_states))
    hidden_states = torch.cat(streams, dim=-1)
```

### D. Add _gist_stream helper method

After the get_quant_config method, add:
```python
def _gist_stream(self, like: torch.Tensor) -> torch.Tensor:
    """Projected+normed oracle gist for this step, zeros when unarmed."""
    num_rows = int(like.shape[0])
    gist = eagle_gist.current(num_rows, like.device, like.dtype)
    if gist is None or self.gist_projector is None:
        return torch.zeros_like(like)
    return self.gist_norm(self.gist_projector(gist))
```

### E. Extend Eagle3DraftModel.__init__ (around line 380-410)

Replace the old gist guard that rejected "qkv" with full support:
```python
# AngelSlim oracle gist. Both injection points are supported:
#   fc   -- one more FC input stream
#   qkv  -- a third layer-0 attention stream
#   both -- both of the above
self.gist_conditioning = bool(
    getattr(self.config, "gist_conditioning", False)
)
self.gist_injection = str(getattr(self.config, "gist_injection", "qkv"))
if self.gist_conditioning and self.gist_injection not in ("qkv", "fc", "both"):
    raise ValueError(f"unknown gist_injection {self.gist_injection!r}")

self.gist_fc_stream = self.gist_conditioning and self.gist_injection in ("fc", "both")
self.gist_qkv_stream = self.gist_conditioning and self.gist_injection in ("qkv", "both")
self.gist_embedding_dim = int(getattr(self.config, "gist_embedding_dim", 0) or 0)
self.gist_projector = None
self.gist_stream_norm = None

if self.gist_conditioning:
    if self.gist_embedding_dim <= 0:
        raise ValueError("gist_conditioning requires gist_embedding_dim > 0")
    if self.gist_fc_stream:
        self.fc_input_size += self.config.hidden_size
    eagle_gist.init_from_env()
```

Then, where gist_projector is built (around line 570):
```python
if self.gist_conditioning:
    self.gist_projector = ReplicatedLinear(
        input_size=self.gist_embedding_dim,
        output_size=self.config.hidden_size,
        bias=False,
        params_dtype=vllm_config.model_config.dtype,
        quant_config=None,
        prefix=maybe_prefix(prefix, "gist_projector"),
        return_bias=False,
    )
    if self.gist_fc_stream:
        self.gist_stream_norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
    if self.gist_qkv_stream:
        self.layers[0].gist_projector = self.gist_projector
```

### F. Extend load_weights (around line 1140-1160)

Replace:
```python
skip_substrs.append("gist_norm.")
if not getattr(self.model, "gist_fc_stream", False):
    skip_substrs.append("gist_projector.")
    skip_substrs.append("gist_stream_norm.")
else:
    for required in ("model.gist_projector.weight", "model.gist_stream_norm.weight"):
        ...
```

With:
```python
if not getattr(self.model, "gist_qkv_stream", False):
    skip_substrs.append("gist_norm.")
if not getattr(self.model, "gist_conditioning", False):
    skip_substrs.append("gist_projector.")
if not getattr(self.model, "gist_fc_stream", False):
    skip_substrs.append("gist_stream_norm.")
if getattr(self.model, "gist_fc_stream", False):
    for required in ("model.gist_projector.weight", "model.gist_stream_norm.weight"):
        if required not in model_weights:
            raise ValueError(f"gist_injection='fc' draft is missing {required}")
```

## 3. Extend speculator.py

**File**: `third_party/vllm/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`

### A. Add import at the top
```python
from vllm.model_executor.models import eagle_gist
```

### B. Extend propose method (around line 230-240)

Find where `combine_hidden_states` is called:
```python
if aux_hidden_states:
    assert self.method == "eagle3"
    hidden_states = self.model.combine_hidden_states(
        torch.cat(aux_hidden_states, dim=-1)
    )
```

Replace with:
```python
if aux_hidden_states:
    assert self.method == "eagle3"
    if not dummy_run and eagle_gist.is_enabled():
        eagle_gist.arm(
            req_ids=list(input_batch.req_ids),
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            num_tokens=int(input_batch.num_tokens),
            num_reqs=int(num_reqs),
            device=self.hidden_states.device,
            dtype=self.hidden_states.dtype,
        )
    hidden_states = self.model.combine_hidden_states(
        torch.cat(aux_hidden_states, dim=-1)
    )
```

---

## Validation

After applying all patches, verify:
1. `eagle_gist.py` exists and imports cleanly
2. `llama_eagle3.py` has both `gist_fc_stream` and `gist_qkv_stream` flags
3. Layer 0 has `_gist_stream()` method
4. `speculator.py` calls `eagle_gist.arm()` before combine_hidden_states

Then test with a small eval:
```bash
GIST_REF=... DATASET=Lin-Chen/MMStar NUM_PROMPTS=8 \
bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
```

Look for `Oracle gist first arm: X/Y requests matched (LIVE)` in the log.
