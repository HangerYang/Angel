#!/usr/bin/env python3
"""Does image_mass query weighting actually change which image rows survive?

Runs the REAL SmolVLM-256M target on real image samples, scores each tile's
rows both ways (uniform vs image_mass), and reports how much the kept top-M
set differs. Cheap: one target forward per sample, no training.
"""
import argparse, os, sys, types
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from angelslim.compressor.speculative.train.data.dataset import DatasetManager
from angelslim.compressor.speculative.train.models.target.target_model_wrapper import (
    create_target_model,
)
from angelslim.compressor.vistoken.splice import image_tiles
from angelslim.compressor.vistoken.target_attn_prune import (
    TargetQKCapture,
    image_row_scores,
    prune_tiles_by_score,
)

AUX_LAYERS = [2, 4, 8, 10, 15, 18, 20, 26, 28]
IMG_TOKEN, TILE, GROUP, KEEP_M = 49190, 64, 64, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--target", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--data", default="dataset/smolvlm_256m_target_gen_mixed_70k70k/train_images_only.jsonl")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--keep-m", type=int, default=KEEP_M)
    a = ap.parse_args()

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

    cfg = getattr(target.model, "config", None)
    cfg = getattr(cfg, "text_config", cfg)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    jac, n_tiles_seen, w_top1, w_gini = [], 0, [], []
    jac_rand, conc = [], []
    for i in range(min(a.n, len(train_ds))):
        batch = collator([train_ds[i]])
        ids = batch["input_ids"].cuda()
        am = batch["attention_mask"].cuda()
        lm = batch["loss_mask"].cuda()
        kw = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()
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
                target.get_hidden_states_and_logits(
                    input_ids=ids, attention_mask=am,
                    aux_hidden_states_layer_ids=AUX_LAYERS, **kw)
        ipos = tiles.reshape(-1)
        vpos = valid.nonzero().flatten()
        # How concentrated is the attention WITHIN a tile? If the top-M rows
        # hold no more mass than M/64, top-M selection is near-arbitrary and
        # pruning cannot help however the queries are weighted.
        kept = {}
        for wname in ("uniform", "image_mass"):
            rs = image_row_scores(cap, AUX_LAYERS, qpos, ipos, head_dim, 0,
                                  valid_positions=vpos, query_weighting=wname)
            sc = torch.zeros(ids.shape[1], device=ids.device, dtype=rs.dtype)
            sc[ipos] = rs
            kept[wname] = prune_tiles_by_score(tiles, sc, GROUP, a.keep_m, "target_attn")
        rs_u = torch.zeros(ids.shape[1], device=ids.device)
        rs_u[ipos] = image_row_scores(cap, AUX_LAYERS, qpos, ipos, head_dim, 0,
                                      valid_positions=vpos, query_weighting="uniform")
        rnd = prune_tiles_by_score(tiles, rs_u, GROUP, a.keep_m, "random")
        for t in range(tiles.shape[0]):
            s1 = set(kept["uniform"][t].tolist()); s2 = set(kept["image_mass"][t].tolist())
            jac.append(len(s1 & s2) / len(s1 | s2)); n_tiles_seen += 1
            sr = set(rnd[t].tolist())
            jac_rand.append(len(s1 & sr) / len(s1 | sr))
            tile_sc = rs_u[tiles[t]]
            tile_p = tile_sc / tile_sc.sum().clamp_min(1e-9)
            topm, _ = tile_p.sort(descending=True)
            conc.append(topm[: a.keep_m].sum().item())
        # how peaked the image_mass weights are
        q_all = cap.captured[AUX_LAYERS[0]][0][0].view(ids.shape[1], -1, head_dim)
        k_all = cap.captured[AUX_LAYERS[0]][1][0].view(ids.shape[1], -1, head_dim).mean(1).float()
        qa = q_all.index_select(0, qpos).mean(1).float()
        s_img = (qa @ k_all[ipos].T) / head_dim**0.5
        s_all = (qa @ k_all[vpos].T) / head_dim**0.5
        w = (torch.logsumexp(s_img, -1) - torch.logsumexp(s_all, -1)).exp()
        w = w / w.sum()
        w_top1.append(w.max().item())
        ws, _ = w.sort(descending=True)
        w_gini.append(ws[: max(1, len(ws) // 10)].sum().item())  # mass in top 10% of queries

    import statistics as st
    print(f"\nsamples with images: {len(w_top1)}   tiles compared: {n_tiles_seen}   keep_m={a.keep_m}/{GROUP}")
    print(f"kept-set Jaccard (uniform vs image_mass): mean {st.mean(jac):.3f}  median {st.median(jac):.3f}  min {min(jac):.3f}")
    ident = sum(1 for j in jac if j == 1.0) / len(jac)
    print(f"tiles where the kept set is IDENTICAL:   {ident:.1%}")
    print(f"image_mass weights: max single query {st.mean(w_top1):.3f}   top-10% of queries hold {st.mean(w_gini):.3f} of the mass")
    print(f"(perfectly uniform weighting would give: top-10% holds 0.100)")
    print()
    print(f"[signal check] top-{a.keep_m} rows hold {st.mean(conc):.3f} of a tile's score mass "
          f"(uniform would be {a.keep_m}/{GROUP} = {a.keep_m/GROUP:.3f})")
    print(f"[floor] Jaccard(target_attn, random): mean {st.mean(jac_rand):.3f} "
          f"(pure chance = {a.keep_m/(2*GROUP-a.keep_m):.3f})")


if __name__ == "__main__":
    main()
