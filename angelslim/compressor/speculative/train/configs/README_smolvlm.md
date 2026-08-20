# SmolVLM draft configs

Target: `HuggingFaceTB/SmolVLM-256M-Instruct` (`target_model_type: smolvlm`).

Pass via:

```bash
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/<file>.json \
  OUTPUT_DIR=output/<run_name> \
  bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
```

Ops / launch modes / eval: `scripts/speculative/smolvlm/README.md`.

| Config | Mode | Draft depth | Aux (train → vLLM) | Notes |
|---|---|---|---|---|
| `smolvlm-256m-eagle3.json` | `fused_fc` (default) | 1 | `[1,14,26]` → `[2,15,27]` | Stock Eagle3: `fc` 3H→H + 2H L0 |
| `smolvlm-256m-eagle3-3.1.json` | `fused_fc` | 1 | same | Stock + EAGLE 3.1 (`fc_norm` + `norm_output`) |
| `smolvlm-256m-eagle3-3.1-qk-norm.json` | `fused_fc` | 1 | same | 3.1 + **QK-norm**: per-head RMSNorm on Q/K pre-RoPE (`qk_norm: true`); adds `self_attn.q_norm/k_norm`, ones-init so step 0 matches 3.1 |
| `smolvlm-256m-eagle3-progressive.json` | `progressive_staged` | 3 | `[0,13,25]` → `[1,14,26]` | Per-layer 2H concat inject; needs progressive vLLM patch |
| `smolvlm-256m-eagle3-progressive-layers-1-15-23-3.1.json` | `progressive_staged` | 3 | `[1,15,23]` → `[2,16,24]` | Progressive 1/15/23 + EAGLE 3.1 `norm_output` (no `fc_norm`) |
| `smolvlm-256m-eagle3-progressive-uninit.json` | `progressive_staged` | 3 | same | Same as progressive, **no** `draft_layer_init_*` |
| `smolvlm-256m-hawk.json` | `hawk` | 3 | `[0,13,25]` → `[1,14,26]` | Progressive **H** fusion (`w1`/`w2` add); full draft layers trainable after target init; needs progressive vLLM patch |
| `smolvlm-256m-real-hawk.json` | `real_hawk` | 3 | `[0,13,25]` → `[1,14,26]` | **Real hawk**: same fuse; draft blocks = frozen target layers `[1,14,26]` + **LoRA**; train `fuse_w1/w2` + LoRA + head. Merge for vLLM hawk eval |
| `smolvlm-256m-layer-skip-lora.json` | `layer_skip_lora` | 3 | same | Back-compat **alias** of `real_hawk` |

`num_hidden_layers` ≠ progressive by itself — set `eagle_aux_injection_mode`.
