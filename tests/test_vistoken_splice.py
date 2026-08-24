"""Sanity checks for the Idea-1 row compressor and its sequence splice."""
import torch

from angelslim.compressor.vistoken.row_compressor import (
    VisRowCompressor,
    VisRowCompressorConfig,
)
from angelslim.compressor.vistoken.splice import compress_image_rows

IMG = 49190
D, NS, TILE = 8, 9, 64


def build_batch(n_tiles=3, n_text=7, k_cfg=None):
    """[text | (marker, 64 img) * n_tiles | text] -- SmolVLM's layout."""
    ids, is_img = [], []
    for _ in range(n_text):
        ids.append(1000 + len(ids))
        is_img.append(False)
    for t in range(n_tiles):
        ids.append(2000 + t)  # <row_i_col_j> marker
        is_img.append(False)
        ids.extend([IMG] * TILE)
        is_img.extend([True] * TILE)
    for _ in range(n_text):
        ids.append(3000 + len(ids))
        is_img.append(False)
    seq = len(ids)
    input_ids = torch.tensor(ids).unsqueeze(0)
    hidden = torch.randn(1, seq, NS * D)
    loss_mask = torch.zeros(1, seq)
    loss_mask[0, -n_text:] = 1.0
    attn = torch.ones(1, seq, dtype=torch.long)
    logits = torch.randn(1, seq, 5)
    return input_ids, hidden, loss_mask, attn, logits, torch.tensor(is_img)


def run(k, routing="shared", query_mode="learned"):
    cfg = VisRowCompressorConfig(
        hidden_size=D, num_streams=NS, num_queries=k, routing=routing,
        query_mode=query_mode,
        stream_bands=((0, 1, 2, 3), (4, 5, 6), (7, 8)) if routing == "per_band" else None,
    )
    comp = VisRowCompressor(cfg)
    ids, hidden, loss_mask, attn, logits, is_img = build_batch()
    out = compress_image_rows(
        comp, hidden_states=hidden, input_ids=ids, attention_mask=attn,
        loss_mask=loss_mask, target_logits=logits, position_ids=None,
        image_token_id=IMG,
    )
    return cfg, comp, (ids, hidden, loss_mask, is_img), out


def main():
    torch.manual_seed(0)
    n_tiles, n_text = 3, 7

    for k in (1, 4, 16):
        cfg, comp, src, out = run(k)
        ids, hidden, loss_mask, is_img = src
        seq = ids.shape[1]
        exp_len = seq - n_tiles * (TILE - k)
        assert out["input_ids"].shape == (1, exp_len), out["input_ids"].shape
        assert out["hidden_states"].shape == (1, exp_len, NS * D)
        ins = out["compressed_rows"][0]
        assert ins.sum().item() == n_tiles * k
        # every compressed row keeps the <image> id, so embed_input_ids finds it
        assert (out["input_ids"][0][ins] == IMG).all()
        assert not (out["input_ids"][0][~ins] == IMG).any()
        # text rows survive byte-for-byte, hidden and ids alike
        text_src = (~is_img).nonzero().flatten()
        assert torch.equal(out["input_ids"][0][~ins], ids[0][text_src])
        assert torch.equal(out["hidden_states"][0][~ins], hidden[0][text_src])
        assert torch.equal(out["loss_mask"][0][~ins], loss_mask[0][text_src])
        # text positions are the ORIGINAL absolute positions (gaps kept)
        assert torch.equal(out["position_ids"][0][~ins], text_src)
        assert (out["loss_mask"][0][ins] == 0).all()
        # each summary keeps the REAL position of the tile slot it occupies
        pos = out["position_ids"][0][ins].view(n_tiles, k)
        tiles = is_img.nonzero().flatten().view(n_tiles, TILE)
        offs = torch.as_tensor(VisRowCompressor.slot_offsets(k, TILE))
        assert torch.equal(pos, tiles.index_select(1, offs)), (pos, offs)
        print(f"k={k}: {seq} -> {exp_len} rows, {n_tiles * k} compressed. ok")

    # mean-pool null baseline reproduces a plain average over each tile's rows
    cfg, comp, src, out = run(1, query_mode="mean")
    ids, hidden, loss_mask, is_img = src
    tiles = is_img.nonzero().flatten().view(n_tiles, TILE)
    ins = out["compressed_rows"][0]
    got = out["hidden_states"][0][ins]
    want = torch.stack([hidden[0][tiles[t]].mean(0) for t in range(n_tiles)])
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
    print("mean query_mode == uniform tile average. ok")

    # identity value path: output rows are convex combinations of real rows
    cfg, comp, src, out = run(4)
    ids, hidden, loss_mask, is_img = src
    tiles = is_img.nonzero().flatten().view(n_tiles, TILE)
    ins = out["compressed_rows"][0]
    got = out["hidden_states"][0][ins].view(n_tiles, 4, NS * D)
    for t in range(n_tiles):
        lo = hidden[0][tiles[t]].min(0).values
        hi = hidden[0][tiles[t]].max(0).values
        assert (got[t] >= lo - 1e-4).all() and (got[t] <= hi + 1e-4).all()
    print("outputs stay in the convex hull of the tile's real rows. ok")

    # shared routing sends every aux stream through ONE decision
    cfg = VisRowCompressorConfig(hidden_size=D, num_streams=NS, num_queries=2)
    comp = VisRowCompressor(cfg)
    tiles_hs = torch.randn(1, n_tiles, TILE, NS * D)
    out2, w = comp(tiles_hs, n_tiles)
    assert w.shape == (1, n_tiles, 2, TILE)
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)
    streams = tiles_hs.view(1, n_tiles, TILE, NS, D)
    want = torch.einsum("btkn,btnsd->btksd", w, streams).reshape(1, n_tiles, 2, NS * D)
    assert torch.allclose(out2, want, atol=1e-5)
    print("shared routing: one w applied to all 9 streams. ok")

    # per_band routing: 3 k_projs, one query set
    _, comp, _, out3 = run(2, routing="per_band")
    assert len(comp.k_proj) == 3 and comp.queries.shape == (2, D)
    print("per_band routing: 3 k_projs, one query set. ok")

    n_params = sum(p.numel() for p in VisRowCompressor(
        VisRowCompressorConfig(hidden_size=576, num_streams=9, num_queries=1)
    ).parameters())
    print(f"params at d=576, k=1, 32 tiles: {n_params}")


main()
