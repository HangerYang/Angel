#!/usr/bin/env python3
"""Where, on the 8x8 tile grid, does the compressor's one-hot pick land -- and why?

Extends vistoken_routing_probe.py with:
  * the attn-prune run, evaluated the way it TRAINED (compressor sees only the
    target-attention top-16 of each tile, not all 64)
  * the 8x8 spatial position of the pick, so it can be drawn as a heatmap
  * a test of the "it just picks the biggest vector" hypothesis: with no value
    projection and a draft loss that barely uses visual tokens, the routing may
    be tracking row norm rather than content.

Dumps per-tile grids to JSON for the heatmap artifact.
"""
import argparse, json, os, sys, types
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from angelslim.compressor.speculative.train.data.dataset import DatasetManager
from angelslim.compressor.speculative.train.models.target.target_model_wrapper import (
    create_target_model,
)
from angelslim.compressor.vistoken.row_compressor import VisRowCompressor, VisRowCompressorConfig
from angelslim.compressor.vistoken.splice import image_tiles
from angelslim.compressor.vistoken.target_attn_prune import TargetQKCapture, image_row_scores

AUX_LAYERS = [2, 4, 8, 10, 15, 18, 20, 26, 28]
IMG_TOKEN, TILE, GRID = 49190, 64, 8

RUNS = {
    "vistoken_k1": ("my_angel/eagle/vistoken-k1/checkpoint-66466", 0),
    "vistoken_k1_attn_prune": ("my_angel/eagle/vistoken-k1-attn-prune-x64-m16/checkpoint-66466", 16),
}


def load_compressor(ckpt, device):
    from safetensors.torch import load_file
    cj = json.load(open(os.path.join(ckpt, "config.json")))
    opts = dict(cj["vistoken_compress"])
    opts.setdefault("hidden_size", cj["hidden_size"])
    opts.setdefault("num_streams", len(cj["aux_hidden_states_layer_ids"]))
    comp = VisRowCompressor(VisRowCompressorConfig(**opts))
    sd = load_file(os.path.join(ckpt, "model.safetensors"))
    comp.load_state_dict({k[len("vistoken."):]: v for k, v in sd.items()
                          if k.startswith("vistoken.")}, strict=False)
    return comp.float().to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--target", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--data", default="dataset/smolvlm_256m_target_gen_mixed_70k70k/train_images_only.jsonl")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--dump", default="my_angel/vistoken_heatmap/grids.json")
    a = ap.parse_args()
    device = "cuda"

    target = create_target_model(backend="hf", model_path=a.target, modal_type="VLM",
                                 torch_dtype=torch.bfloat16, trust_remote_code=True,
                                 target_model_type="smolvlm")
    da = types.SimpleNamespace(
        modal_type="VLM", target_model_name_or_path=a.target, train_data_path=a.data,
        eval_data_path=None, shuffle_seed=0, output_dir=None, gist_conditioning=False,
        num_samples=a.n, training_mode="online", gist_encoder_model_name_or_path=None,
        max_len=a.max_len, cache_key=None, num_workers=0, num_proc=1,
        sample_num=a.n, load_from_cache_file=False)
    dm = DatasetManager(data_args=da, tokenizer=target.tokenizer, model_max_length=a.max_len,
                        chat_template_type="smolvlm", target_model_type="smolvlm")
    train_ds, _, collator = dm.create_online_datasets()

    tc = getattr(target.model, "config", None)
    tc = getattr(tc, "text_config", tc)
    head_dim = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads

    comps = {n: load_compressor(p, device) for n, (p, _) in RUNS.items()}
    st = {n: dict(cell=Counter(), norm_hit=[], tgt_hit=[], corr_norm=[], corr_tgt=[],
                  top1=[], tgt_cell=torch.zeros(TILE), n=0) for n in RUNS}
    dump = {n: [] for n in RUNS}
    n_seen = 0

    for i in range(min(a.n, len(train_ds))):
        b = collator([train_ds[i]])
        ids, am, lm = b["input_ids"].to(device), b["attention_mask"].to(device), b["loss_mask"].to(device)
        kw = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()
              if k not in ("input_ids", "attention_mask", "loss_mask")}
        tiles = image_tiles(ids[0], am.bool()[0], IMG_TOKEN, TILE)
        qpos = lm[0].nonzero().flatten()
        if tiles is None or qpos.numel() == 0:
            continue
        with TargetQKCapture(target.get_text_decoder_layers(), AUX_LAYERS) as cap:
            with torch.no_grad():
                hs, _, _, _ = target.get_hidden_states_and_logits(
                    input_ids=ids, attention_mask=am,
                    aux_hidden_states_layer_ids=AUX_LAYERS, **kw)
        ipos = tiles.reshape(-1)
        tgt = image_row_scores(cap, AUX_LAYERS, qpos, ipos, head_dim, 0).float().view(tiles.shape)
        rows_all = hs[0].index_select(0, ipos).view(1, tiles.shape[0], TILE, -1).float()
        T = tiles.shape[0]
        n_seen += 1

        for name, (_, keep_m) in RUNS.items():
            comp = comps[name]
            if keep_m:                      # replicate training: top-M by target attention
                sel = tgt.topk(keep_m, dim=-1).indices.sort(dim=-1).values   # [T, M]
                rows = torch.gather(rows_all[0], 1,
                                    sel.unsqueeze(-1).expand(-1, -1, rows_all.shape[-1])).unsqueeze(0)
                tgt_sub = torch.gather(tgt, 1, sel)
            else:
                sel, rows, tgt_sub = None, rows_all, tgt
            with torch.no_grad():
                _, w = comp(rows, T, expected_rows=rows.shape[2])
            w = w[0, :, 0].float()                       # [T, M or 64]
            nrm = rows[0].norm(dim=-1)                   # [T, M or 64]
            s = st[name]
            s["top1"] += w.max(-1).values.tolist()
            s["norm_hit"] += (w.argmax(-1) == nrm.argmax(-1)).float().tolist()
            s["tgt_hit"] += (w.argmax(-1) == tgt_sub.argmax(-1)).float().tolist()
            for x, y, key in ((w, nrm, "corr_norm"), (w, tgt_sub, "corr_tgt")):
                xc, yc = x - x.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
                s[key] += ((xc * yc).sum(-1) /
                           (xc.norm(dim=-1) * yc.norm(dim=-1)).clamp_min(1e-12)).tolist()
            # map the pick back to its absolute 0..63 slot in the tile
            pick = w.argmax(-1)
            abs_pick = pick if sel is None else torch.gather(sel, 1, pick.unsqueeze(-1))[:, 0]
            for v in abs_pick.tolist():
                s["cell"][v] += 1
            s["n"] += T
            if n_seen <= 3:
                dump[name].append({
                    "sample": i, "tiles": T,
                    "pick": abs_pick.tolist(),
                    "target_attn": tgt.tolist(),
                })
        st["vistoken_k1"]["tgt_cell"] += tgt.mean(0).cpu()

    import statistics as S
    os.makedirs(os.path.dirname(a.dump), exist_ok=True)
    out = {"n_samples": n_seen, "runs": {}}
    print(f"\nsamples {n_seen}\n")
    for name in RUNS:
        s = st[name]
        pool = 64 if not RUNS[name][1] else RUNS[name][1]
        g = torch.zeros(TILE)
        for c, v in s["cell"].items():
            g[c] = v
        g = (g / g.sum()).view(GRID, GRID)
        print("=" * 76)
        print(f"{name}   (compressor chooses from {pool} rows)")
        print(f"  top-1 weight                       {S.mean(s['top1']):.4f}   [uniform = {1/pool:.4f}]")
        print(f"  pick == argmax row NORM            {S.mean(s['norm_hit']):.1%}   [chance = {1/pool:.1%}]")
        print(f"  pick == argmax TARGET attention    {S.mean(s['tgt_hit']):.1%}   [chance = {1/pool:.1%}]")
        print(f"  Pearson r(w, row norm)             {S.mean(s['corr_norm']):+.4f}")
        print(f"  Pearson r(w, target attention)     {S.mean(s['corr_tgt']):+.4f}")
        print(f"  where the pick lands on the 8x8 tile grid (fraction of {s['n']} tiles):")
        for r in range(GRID):
            print("      " + " ".join(f"{g[r][c]:5.3f}" for c in range(GRID)))
        out["runs"][name] = {
            "pool": pool, "top1": S.mean(s["top1"]),
            "norm_hit": S.mean(s["norm_hit"]), "tgt_hit": S.mean(s["tgt_hit"]),
            "corr_norm": S.mean(s["corr_norm"]), "corr_tgt": S.mean(s["corr_tgt"]),
            "grid": g.tolist(), "n_tiles": s["n"], "examples": dump[name],
        }
    tg = st["vistoken_k1"]["tgt_cell"]
    tg = (tg / tg.sum()).view(GRID, GRID)
    print("=" * 76)
    print("for reference -- where the TARGET's own attention mass sits on the 8x8 grid:")
    for r in range(GRID):
        print("      " + " ".join(f"{tg[r][c]:5.3f}" for c in range(GRID)))
    out["target_grid"] = tg.tolist()
    json.dump(out, open(a.dump, "w"))
    print(f"\nwrote {a.dump}")


if __name__ == "__main__":
    main()
