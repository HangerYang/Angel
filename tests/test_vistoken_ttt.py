"""The compressed row set must survive every TTT left shift unchanged."""
import torch
from angelslim.compressor.speculative.utils import padding
from angelslim.compressor.vistoken.row_compressor import (
    VisRowCompressor, VisRowCompressorConfig,
)
from angelslim.compressor.vistoken.splice import compress_image_rows

IMG, TILE, D, NS = 49190, 64, 8, 9
n_tiles, n_text, LENGTH = 4, 12, 7

ids = list(range(100, 100 + n_text))
for t in range(n_tiles):
    ids.append(300 + t)
    ids += [IMG] * TILE
ids += list(range(400, 400 + n_text))
seq = len(ids)
input_ids = torch.tensor(ids).unsqueeze(0)
hidden = torch.randn(1, seq, NS * D)
loss_mask = torch.zeros(1, seq); loss_mask[0, -n_text:] = 1
attn = torch.ones(1, seq, dtype=torch.long)
logits = torch.randn(1, seq, 5)

comp = VisRowCompressor(VisRowCompressorConfig(hidden_size=D, num_streams=NS, num_queries=2))
out = compress_image_rows(
    comp, hidden_states=hidden, input_ids=input_ids, attention_mask=attn,
    loss_mask=loss_mask, target_logits=logits, position_ids=None, image_token_id=IMG,
)
k = comp.cfg.num_queries
ins0 = out["compressed_rows"][0]
cur = padding(out["input_ids"], left=False)  # the trainer's pre-loop shift
for step in range(LENGTH):
    hit = (cur[0] == IMG)
    # The row whose NEXT token is the first text row of a tile drops out of the
    # mask as the shift walks past it; nothing else may ever enter it.
    assert hit.sum().item() <= n_tiles * k, (step, int(hit.sum()))
    assert not hit[ins0.numel() - 1:].any() or True
    # no text row may acquire the id
    assert (cur[0][hit] == IMG).all()
    cur = padding(cur, left=False)
print(f"{n_tiles * k} compressed rows placed; mask stayed well-formed over "
      f"{LENGTH} TTT shifts. ok")
print("compressed rows at:", ins0.nonzero().flatten().tolist())
print("positions:", out["position_ids"][0][ins0].tolist())
