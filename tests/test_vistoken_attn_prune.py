"""Sanity checks for target-attention row pruning in front of the compressor."""
import torch
import torch.nn as nn

from angelslim.compressor.vistoken.row_compressor import (
    VisRowCompressor,
    VisRowCompressorConfig,
)
from angelslim.compressor.vistoken.splice import compress_image_rows, image_tiles
from angelslim.compressor.vistoken.target_attn_prune import (
    TargetQKCapture,
    prune_sample_image_rows,
    prune_tiles_by_score,
)

IMG = 49190
D, NS, TILE = 8, 9, 64
HEAD_DIM = 4
N_HEADS = 2


def build_batch(n_tiles=2, n_text=5):
    ids, is_img = [], []
    for _ in range(n_text):
        ids.append(1000 + len(ids))
        is_img.append(False)
    for t in range(n_tiles):
        ids.append(2000 + t)
        is_img.append(False)
        ids.extend([IMG] * TILE)
        is_img.extend([True] * TILE)
    for _ in range(n_text):
        ids.append(3000 + len(ids))
        is_img.append(False)
    input_ids = torch.tensor(ids).unsqueeze(0)
    hidden = torch.randn(1, len(ids), NS * D)
    loss_mask = torch.zeros(1, len(ids))
    loss_mask[0, -n_text:] = 1.0
    attn = torch.ones(1, len(ids), dtype=torch.long)
    return input_ids, hidden, loss_mask, attn, torch.tensor(is_img)


def test_prune_tiles_by_score_picks_top_m():
    tiles = torch.arange(2 * 8).view(2, 8) + 100  # 2 tiles of 8 "rows"
    scores = torch.zeros(1000)
    # tile 0: rows 100..107, groups [100..103] (max at 102) / [104..107] (max at 107)
    scores[100:104] = torch.tensor([0.1, 0.2, 0.9, 0.3])
    scores[104:108] = torch.tensor([0.1, 0.4, 0.2, 0.8])
    kept = prune_tiles_by_score(tiles, scores, group_size=4, keep_m=1)
    assert kept.shape == (2, 2)
    assert kept[0].tolist() == [102, 107]
    print("prune_tiles_by_score: top-1-per-group-of-4 matches hand-picked scores. ok")


def test_qk_capture_and_end_to_end_prune_compress():
    torch.manual_seed(0)
    n_tiles, n_text = 2, 5
    input_ids, hidden, loss_mask, attn, is_img = build_batch(n_tiles, n_text)
    seq_len = input_ids.shape[1]

    # A single fake decoder layer whose q_proj/k_proj TargetQKCapture hooks.
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(D, N_HEADS * HEAD_DIM, bias=False)
    layer.self_attn.k_proj = nn.Linear(D, N_HEADS * HEAD_DIM, bias=False)
    layers = [layer]

    x = torch.randn(1, seq_len, D)
    with TargetQKCapture(layers, [0]) as cap:
        _ = layer.self_attn.q_proj(x)
        _ = layer.self_attn.k_proj(x)
        q, k = cap.captured[0]
        assert q.shape == (1, seq_len, N_HEADS * HEAD_DIM)
        assert k.shape == (1, seq_len, N_HEADS * HEAD_DIM)
    assert cap.captured != {} and cap._handles == []
    print("TargetQKCapture: hooks fire during the forward and clean up on exit. ok")

    tiles = prune_sample_image_rows(
        cap, [0], HEAD_DIM, input_ids[0], attn[0].bool(), loss_mask[0],
        IMG, TILE, group_size=16, keep_m=4, mode="target_attn",
    )
    assert tiles is not None
    assert tiles.shape == (n_tiles, TILE // 16 * 4)  # 4 groups * keep 4 = 16 rows/tile
    real_tiles = is_img.nonzero().flatten().view(n_tiles, TILE)
    for t in range(n_tiles):
        assert set(tiles[t].tolist()) <= set(real_tiles[t].tolist())
        assert tiles[t].tolist() == sorted(tiles[t].tolist())
    print("prune_sample_image_rows: kept rows are a real subset, ascending order. ok")

    # Feed the pruned tiles into compress_image_rows via tiles_per_sample.
    cfg = VisRowCompressorConfig(hidden_size=D, num_streams=NS, num_queries=1)
    comp = VisRowCompressor(cfg)
    out = compress_image_rows(
        comp, hidden_states=hidden, input_ids=input_ids, attention_mask=attn,
        loss_mask=loss_mask, target_logits=torch.randn(1, seq_len, 5), position_ids=None,
        image_token_id=IMG, compressor_rows_per_sample=[tiles],
    )
    # k=1: every tile (all TILE original rows, pruned or not) collapses to 1 slot.
    exp_len = seq_len - n_tiles * (TILE - 1)
    assert out["input_ids"].shape == (1, exp_len), (out["input_ids"].shape, exp_len)
    ins = out["compressed_rows"][0]
    assert ins.sum().item() == n_tiles  # k=1 summary row per tile
    # summary position is the middle of the PRUNED M rows, not the original 64
    offs = torch.as_tensor(VisRowCompressor.slot_offsets(1, tiles.shape[1]))
    want_pos = tiles.index_select(1, offs)
    assert torch.equal(out["position_ids"][0][ins].view(n_tiles, 1), want_pos)
    print("compress_image_rows(tiles_per_sample=...): slot offsets match the pruned tile size. ok")

    # mode="none" must reproduce image_tiles()'s own unpruned tiles exactly.
    unpruned = prune_sample_image_rows(
        None, [0], HEAD_DIM, input_ids[0], attn[0].bool(), loss_mask[0],
        IMG, TILE, group_size=16, keep_m=4, mode="none",
    )
    assert torch.equal(unpruned, image_tiles(input_ids[0], attn[0].bool(), IMG, TILE))
    print("mode=none reproduces image_tiles() unchanged. ok")


if __name__ == "__main__":
    test_prune_tiles_by_score_picks_top_m()
    test_qk_capture_and_end_to_end_prune_compress()
