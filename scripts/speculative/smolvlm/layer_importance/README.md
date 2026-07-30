# SmolVLM target layer importance

Target-only analysis (no Eagle draft). Ranks text-tower layers to help choose
`aux_hidden_states_layer_ids` / init sources.

## Metrics / bash runners

| Script | Metric |
|---|---|
| `run_ce_grad.sh` | masked `‖∂L/∂h_ℓ‖` (target CE) |
| `run_agreement.sh` | probe KL / CE vs final |
| `run_delta.sh` | relative embedding change |
| `run_info.sh` | HS effective rank + variance |
| `run_image_attn.sh` | attn to **image tokens** (skips text-only) |
| `run_all.sh` | all of the above |
| `run_analyze.sh` | alias → `run_all.sh` |

Image attention is foolproof: text-only samples never request attentions.

Layer ids match AngelSlim `aux_hidden_states_layer_ids` (HF `hidden_states[id+1]`).

## Run elsewhere on larger data

From repo root (or this folder). **Set `DATA_PATH`.**

```bash
cd scripts/speculative/smolvlm/layer_importance

export MODEL_PATH=HuggingFaceTB/SmolVLM-256M-Instruct
export DATA_PATH=/path/to/large.jsonl
export OUTPUT_DIR=/path/to/layer_imp_out
export MAX_LENGTH=2048
# export MAX_SAMPLES=1000   # optional cap
# export DEVICE=cuda
# export PYTHON=/path/to/env/bin/python   # must have torch+transformers

bash run_ce_grad.sh
bash run_agreement.sh
bash run_delta.sh
bash run_info.sh
bash run_image_attn.sh
# or everything:
bash run_all.sh
```

Outputs land in `OUTPUT_DIR`:
- `layer_importance_<metric>.json` / `.csv` for single-metric runs
- `layer_importance.json` / `.csv` also written for `run_all.sh`

## Notes

- Prefer single-metric scripts on large data if you want cheaper / parallel jobs.
- `image_attn` uses eager attention (slower / more memory); others use SDPA.
- Depth-only baseline in the JSON is `[1, N/2-1, N-4]`.
