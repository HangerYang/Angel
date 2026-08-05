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
| `smolvlm-256m-eagle3-progressive.json` | `progressive_staged` | 3 | `[0,13,25]` → `[1,14,26]` | Per-layer 2H concat inject; needs progressive vLLM patch |
| `smolvlm-256m-eagle3-progressive-uninit.json` | `progressive_staged` | 3 | same | Same as progressive, **no** `draft_layer_init_*` |
| `smolvlm-256m-hawk.json` | `hawk` | 3 | `[0,13,25]` → `[1,14,26]` | Progressive **H** fusion (`w1`/`w2` add); needs progressive vLLM patch |

`num_hidden_layers` ≠ progressive by itself — set `eagle_aux_injection_mode`.
