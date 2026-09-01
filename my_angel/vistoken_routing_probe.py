#!/usr/bin/env python3
"""Did the vistoken compressor's learned query actually learn to select rows?

The k=1 compressor is a Q-Former-shaped module: one learned query cross-attends
over a tile's 64 target rows and emits a convex combination of them. If its
softmax is flat, the "compressor" is an expensive mean pool and the whole
learned-routing thesis is dead regardless of how long it trains.

This runs the REAL target on real image samples, rebuilds the trained compressor
from a checkpoint, and reports, per tile:

  * effective support exp(H(w)) -- 1 = hard pick, 64 = uniform mean pool
  * how close the output is to the plain tile mean (the query_mode="mean" null)
  * whether the chosen row is content-dependent or a fixed habit
  * whether the choice agrees with what the TARGET's own attention reads

No training, one target forward per sample.
"""
import argparse, json, math, os, sys, types
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from angelslim.compressor.speculative.train.data.dataset import DatasetManager
from angelslim.compressor.speculative.train.models.target.target_model_wrapper import (
    create_target_model,
)
from angelslim.compressor.vistoken.row_compressor import (
    VisRowCompressor,
    VisRowCompressorConfig,
)
from angelslim.compressor.vistoken.splice import image_tiles
from angelslim.compressor.vistoken.target_attn_prune import TargetQKCapture, image_row_scores

AUX_LAYERS = [2, 4, 8, 10, 15, 18, 20, 26, 28]
IMG_TOKEN, TILE = 49190, 64


def load_compressor(ckpt, device):
    from safetensors.torch import load_file

    cfg_json = json.load(open(os.path.join(ckpt, "config.json")))
    opts = dict(cfg_json["vistoken_compress"])
    opts.setdefault("hidden_size", cfg_json["hidden_size"])
    opts.setdefault("num_streams", len(cfg_json["aux_hidden_states_layer_ids"]))
    comp = VisRowCompressor(VisRowCompressorConfig(**opts))
    sd = load_file(os.path.join(ckpt, "model.safetensors"))
    sub = {k[len("vistoken.") :]: v for k, v in sd.items() if k.startswith("vistoken.")}
    missing, unexpected = comp.load_state_dict(sub, strict=False)
    assert not unexpected, unexpected
    assert not [m for m in missing], missing
    return comp.float().to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[
        "my_angel/eagle/vistoken-k1/checkpoint-33233",
        "my_angel/eagle/vistoken-k1/checkpoint-66466",
    ])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--target", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--data", default="dataset/smolvlm_256m_target_gen_mixed_70k70k/train_images_only.jsonl")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--topm", type=int, default=16)
    ap.add_argument("--temps", type=float, nargs="+", default=[8.0, 32.0, 128.0, 512.0, 2048.0])
    a = ap.parse_args()

    device = "cuda"
    target = create_target_model(
        backend="hf", model_path=a.target, modal_type="VLM",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        target_model_type="smolvlm",
    )
    data_args = types.SimpleNamespace(
        modal_type="VLM", target_model_name_or_path=a.target,
        train_data_path=a.data, eval_data_path=None, shuffle_seed=0,
        output_dir=None, gist_conditioning=False, num_samples=a.n,
        training_mode="online", gist_encoder_model_name_or_path=None,
        max_len=a.max_len, cache_key=None, num_workers=0,
        num_proc=1, sample_num=a.n, load_from_cache_file=False,
    )
    dm = DatasetManager(data_args=data_args, tokenizer=target.tokenizer,
                        model_max_length=a.max_len, chat_template_type="smolvlm",
                        target_model_type="smolvlm")
    train_ds, _, collator = dm.create_online_datasets()

    tcfg = getattr(target.model, "config", None)
    tcfg = getattr(tcfg, "text_config", tcfg)
    head_dim = getattr(tcfg, "head_dim", None) or tcfg.hidden_size // tcfg.num_attention_heads

    comps = {c: load_compressor(c, device) for c in a.ckpts}
    stats = {c: dict(eff=[], top1=[], topk=[], cos_mean=[], rel_mean=[],
                     argmax=[], jac=[], corr=[], per_tile_argmax={},
                     logit_range=[], temp={t: dict(eff=[], corr=[], jac=[]) for t in a.temps})
             for c in a.ckpts}
    n_seen = n_tiles = 0

    for i in range(min(a.n, len(train_ds))):
        batch = collator([train_ds[i]])
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        lm = batch["loss_mask"].to(device)
        kw = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
              if k not in ("input_ids", "attention_mask", "loss_mask")}
        valid = am.bool()[0]
        tiles = image_tiles(ids[0], valid, IMG_TOKEN, TILE)
        if tiles is None:
            continue
        qpos = lm[0].nonzero().flatten()
        if qpos.numel() == 0:
            continue

        with TargetQKCapture(target.get_text_decoder_layers(), AUX_LAYERS) as cap:
            with torch.no_grad():
                hs, _, _, _ = target.get_hidden_states_and_logits(
                    input_ids=ids, attention_mask=am,
                    aux_hidden_states_layer_ids=AUX_LAYERS, **kw)
        if n_seen == 0:
            print(f"hidden_states {tuple(hs.shape)}  tiles {tuple(tiles.shape)}", flush=True)

        ipos = tiles.reshape(-1)
        tgt_sc = image_row_scores(
            cap, AUX_LAYERS, qpos, ipos, head_dim, 0
        ).float().view(tiles.shape)

        # [1, T, 64, 9*576] exactly as splice.py gathers it
        rows = hs[0].index_select(0, ipos).view(1, tiles.shape[0], TILE, -1).float()
        n_seen += 1
        n_tiles += tiles.shape[0]

        for ck, comp in comps.items():
            with torch.no_grad():
                out, w = comp(rows, tiles.shape[0])          # out [1,T,1,F], w [1,T,1,64]
            w = w[0, :, 0].float()                            # [T, 64]
            st = stats[ck]
            p = w.clamp_min(1e-12)
            H = -(p * p.log()).sum(-1)
            st["eff"] += H.exp().tolist()
            st["top1"] += w.max(-1).values.tolist()
            st["topk"] += w.sort(-1, descending=True).values[:, : a.topm].sum(-1).tolist()

            mean_out = rows.mean(2)[0]                        # [T, F]
            o = out[0, :, 0]                                  # [T, F]
            st["cos_mean"] += torch.nn.functional.cosine_similarity(o, mean_out, dim=-1).tolist()
            st["rel_mean"] += ((o - mean_out).norm(dim=-1) / mean_out.norm(dim=-1)).tolist()

            am_idx = w.argmax(-1)
            st["argmax"] += am_idx.tolist()
            for t, v in enumerate(am_idx.tolist()):
                st["per_tile_argmax"].setdefault(t, []).append(v)

            # agreement with the target's own attention over the same rows
            ctop = w.topk(a.topm, dim=-1).indices
            ttop = tgt_sc.topk(a.topm, dim=-1).indices
            for t in range(tiles.shape[0]):
                s1, s2 = set(ctop[t].tolist()), set(ttop[t].tolist())
                st["jac"].append(len(s1 & s2) / len(s1 | s2))
            wc = w - w.mean(-1, keepdim=True)
            tc = tgt_sc - tgt_sc.mean(-1, keepdim=True)
            st["corr"] += ((wc * tc).sum(-1) /
                           (wc.norm(dim=-1) * tc.norm(dim=-1)).clamp_min(1e-12)).tolist()

            # Temperature sweep on the SAME learned query direction. If the
            # correlation with the target's attention only appears once the
            # softmax is un-saturated, the query learned something real and the
            # temperature is what destroyed it. If it stays flat, the direction
            # itself carries no signal.
            with torch.no_grad():
                ref = comp._reference(rows.view(1, tiles.shape[0], TILE, comp.cfg.num_streams,
                                                comp.cfg.hidden_size))
                q = comp.queries.unsqueeze(0) + comp.tile_embed[: tiles.shape[0]].unsqueeze(1)
                proj = comp.k_proj[0]
                logits = torch.einsum("tkc,btnc->btkn", proj(q), proj(ref))[0, :, 0]  # [T,64]
            st["logit_range"] += (logits.max(-1).values - logits.min(-1).values).tolist()
            for T in a.temps:
                wt = torch.softmax(logits / T, dim=-1).float()
                pt = wt.clamp_min(1e-12)
                st["temp"][T]["eff"] += (-(pt * pt.log()).sum(-1)).exp().tolist()
                wtc = wt - wt.mean(-1, keepdim=True)
                st["temp"][T]["corr"] += ((wtc * tc).sum(-1) /
                                          (wtc.norm(dim=-1) * tc.norm(dim=-1)).clamp_min(1e-12)).tolist()
                ctop_t = wt.topk(a.topm, dim=-1).indices
                for t in range(tiles.shape[0]):
                    s1, s2 = set(ctop_t[t].tolist()), set(ttop[t].tolist())
                    st["temp"][T]["jac"].append(len(s1 & s2) / len(s1 | s2))

    import statistics as S
    chance_jac = a.topm / (2 * TILE - a.topm)
    print(f"\nsamples with images: {n_seen}   tiles: {n_tiles}   rows/tile: {TILE}\n")
    for ck in a.ckpts:
        st = stats[ck]
        am_hist = Counter(st["argmax"])
        amp = torch.tensor([c / len(st["argmax"]) for c in am_hist.values()])
        am_eff = float((-(amp * amp.log()).sum()).exp())
        # how often the SAME tile index picks the SAME row across different images
        same = [max(Counter(v).values()) / len(v) for v in st["per_tile_argmax"].values() if len(v) > 1]
        print("=" * 78)
        print(ck)
        print(f"  routing softmax over the tile's 64 rows")
        print(f"    effective support exp(H)   mean {S.mean(st['eff']):7.2f}   median {S.median(st['eff']):7.2f}"
              f"   min {min(st['eff']):6.2f}   max {max(st['eff']):6.2f}     [1 = hard pick, 64 = mean pool]")
        print(f"    top-1 weight               mean {S.mean(st['top1']):7.4f}                            [uniform = {1/TILE:.4f}]")
        print(f"    top-{a.topm} mass              mean {S.mean(st['topk']):7.4f}                            [uniform = {a.topm/TILE:.4f}]")
        print(f"  distance from the query_mode='mean' null")
        print(f"    cos(output, tile mean)     mean {S.mean(st['cos_mean']):7.4f}   median {S.median(st['cos_mean']):7.4f}")
        print(f"    ||out - mean|| / ||mean||  mean {S.mean(st['rel_mean']):7.4f}")
        print(f"  is the choice content-dependent?")
        print(f"    distinct argmax rows used  {len(am_hist)}/{TILE}   effective {am_eff:.2f}")
        print(f"    same tile index -> same row {S.mean(same):.1%} of the time (across different images)")
        print(f"  agreement with the TARGET's own attention over the same rows")
        print(f"    top-{a.topm} Jaccard            mean {S.mean(st['jac']):7.4f}   [chance = {chance_jac:.4f}]")
        print(f"    Pearson r(w, target score) mean {S.mean(st['corr']):7.4f}")
        print(f"  is the softmax saturated, or is the query direction just uninformative?")
        print(f"    raw logit range (max-min)  mean {S.mean(st['logit_range']):8.2f}   [trained temperature = {comps[ck].cfg.temperature}]")
        print(f"    {'temperature':>12}  {'exp(H)':>8}  {'r(w,target)':>12}  {'top-'+str(a.topm)+' Jaccard':>16}")
        for T in a.temps:
            d = st["temp"][T]
            print(f"    {T:12.0f}  {S.mean(d['eff']):8.2f}  {S.mean(d['corr']):12.4f}  {S.mean(d['jac']):16.4f}")
        print(f"    {'':12}  {'':8}  {'':12}  {'chance = '+format(chance_jac,'.4f'):>16}")


if __name__ == "__main__":
    main()
