"""Training splice vs the vLLM path must keep the same rows with the same values.

The vLLM side (third_party/patches/20-vllm-v0.25.0-eagle3-vistoken.patch) writes
each tile's summaries into fixed slots of the raw aux concat and hands the
compact pass a keep mask. This reproduces that arithmetic and checks it lands
on the training splice, row for row.
"""
import torch

from angelslim.compressor.vistoken.row_compressor import (
    VisRowCompressor, VisRowCompressorConfig,
)
from angelslim.compressor.vistoken.splice import compress_image_rows

IMG, TILE, D, NS = 49190, 64, 8, 9


def build(n_tiles=3, n_text=9):
    ids = list(range(100, 100 + n_text))
    for t in range(n_tiles):
        ids.append(300 + t)
        ids += [IMG] * TILE
    ids += list(range(400, 400 + n_text))
    seq = len(ids)
    return torch.tensor(ids).unsqueeze(0), torch.randn(1, seq, NS * D), seq


def vllm_side(comp, ids, aux, num_tokens):
    """What speculator._maybe_compress_visual_rows does, in plain torch."""
    aux = aux[0].clone()
    img_pos = ids[0].eq(IMG).nonzero().flatten()
    tiles = img_pos.view(-1, TILE)
    out, _ = comp(aux[tiles].unsqueeze(0), tiles.shape[0])
    slots = tiles.index_select(1, torch.as_tensor(comp.offsets)).reshape(-1)
    aux[slots] = out[0].reshape(slots.numel(), -1)
    keep = torch.ones(num_tokens, dtype=torch.bool)
    keep[tiles.reshape(-1)] = False
    keep[slots] = True
    return aux[keep], keep


def main():
    torch.manual_seed(0)
    for k in (1, 4):
        comp = VisRowCompressor(
            VisRowCompressorConfig(hidden_size=D, num_streams=NS, num_queries=k)
        )
        with torch.no_grad():
            comp.queries.normal_(std=0.3)
            comp.tile_embed.normal_(std=0.3)
            comp.ref_mix_logits.normal_(std=0.5)
        ids, aux, seq = build()
        loss_mask = torch.zeros(1, seq)
        loss_mask[0, -9:] = 1
        attn = torch.ones(1, seq, dtype=torch.long)

        train = compress_image_rows(
            comp, hidden_states=aux, input_ids=ids, attention_mask=attn,
            loss_mask=loss_mask, target_logits=torch.randn(1, seq, 5),
            position_ids=None, image_token_id=IMG,
        )
        vl, keep = vllm_side(comp, ids, aux, seq)

        assert train["hidden_states"].shape[1] == int(keep.sum()), (
            train["hidden_states"].shape, int(keep.sum())
        )
        assert torch.equal(train["input_ids"][0], ids[0][keep])
        assert torch.equal(train["position_ids"][0], keep.nonzero().flatten())
        assert torch.allclose(train["hidden_states"][0], vl, atol=1e-6), (
            (train["hidden_states"][0] - vl).abs().max()
        )
        # and the summary rows are exactly the ones both sides call compressed
        assert torch.equal(
            train["compressed_rows"][0].nonzero().flatten(),
            (ids[0][keep] == IMG).nonzero().flatten(),
        )
        print(f"k={k}: {seq} -> {int(keep.sum())} rows; train and vLLM paths "
              "agree row-for-row. ok")


main()
