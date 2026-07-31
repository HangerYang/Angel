# SmolVLM target layer importance

Target-only analysis (no Eagle draft).

## Outputs

1. **Per metric** (`metric_reports` in JSON): full **global ranking** + **band ranking** (early/mid/late), plus `top3_global` / `top3_band`.
2. **Metric summaries**: `metric_summaries_*.json`
3. **Eval** each candidate set by reconstructing **final-layer HS** (lstsq) from selected layers vs `full_layers` (all but last):
   - `{metric}__band` — band top-3
   - `{metric}__global` — global top-3
   - `depth_baseline` — `[1, N/2-1, N-4]`
   - `random_*` — if `EVAL_RANDOM=1`
4. **Final comparison** ranked by `final_score = recon_cos / full_recon_cos` → `final_comparison_*.csv`

## Run

```bash
DATA_PATH=/path/to/data.jsonl bash run_all.sh
EVAL_RANDOM=1 bash run_all.sh
```

Layer ids match AngelSlim `aux_hidden_states_layer_ids` (HF `hidden_states[id+1]`).
Image attn skips text-only samples.
