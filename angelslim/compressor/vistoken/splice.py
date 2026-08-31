"""Drop each image tile down to k rows in the EAGLE-3 draft sequence.

Runs once, on the prefill sequence, before the trainer's left shift and before
``combine_hidden_states``. Compression is a weighted sum over *positions*; the
band mix is a weighted sum over *layers*; ``fc`` is linear. They commute, so
compressing the raw aux concat first and letting fc_norm / band mix / fc run
afterwards is exactly equivalent to compressing after the mix -- and the only
nonlinearity (the norm) still comes last.

The k summaries of a tile are written into k fixed rows *of that tile*
(``VisRowCompressor.slot_offsets``) and the tile's other rows are dropped. So
this is a pure keep-mask over the sequence: a surviving row keeps its own token
id, its own loss mask, and above all its own absolute position, which is the
RoPE angle the target computed it at. Nothing is invented and nothing is
renumbered.

Choosing the kept rows from ``input_ids`` alone -- never from the routing
weights -- is what lets vLLM reuse this: it builds the draft's slot mapping
before the model runs (``speculator.py:_maybe_set_l0_compact_prefill``).
"""

from typing import Dict, List, Optional

import torch


def _tile_index(img_positions: torch.Tensor, tile_tokens: int) -> torch.Tensor:
    """[n_img] absolute positions -> [T, tile_tokens], asserting contiguity."""
    if img_positions.numel() % tile_tokens != 0:
        raise ValueError(
            f"{img_positions.numel()} image rows is not a multiple of "
            f"{tile_tokens}; the tile split assumption is broken"
        )
    tiles = img_positions.view(-1, tile_tokens)
    step = tiles[:, 1:] - tiles[:, :-1]
    if not torch.all(step == 1):
        raise ValueError(
            "image rows within a tile are not contiguous; the compressor "
            "assumes SmolVLM's 64-row runs separated by grid markers"
        )
    return tiles


def image_tiles(
    input_ids_row: torch.Tensor,
    valid_row: torch.Tensor,
    image_token_id: int,
    tile_tokens: int,
) -> Optional[torch.Tensor]:
    """Absolute positions of one sample's image rows as [T, tile_tokens]."""
    img_pos = ((input_ids_row == image_token_id) & valid_row).nonzero().flatten()
    if img_pos.numel() == 0:
        return None
    return _tile_index(img_pos, tile_tokens)


def compress_image_rows(
    compressor,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    loss_mask: torch.Tensor,
    target_logits: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    image_token_id: int,
    compressor_rows_per_sample: Optional[List[Optional[torch.Tensor]]] = None,
) -> Dict[str, torch.Tensor]:
    """Keep k rows per image tile, carrying the tile's compressed summaries.

    All inputs are aligned 1:1 on absolute positions (pre-shift). Returns the
    same keys, right-padded to the longest surviving sample in the batch, plus
    ``compressed_rows``: a mask of the rows now holding summaries.

    The tile itself is always auto-detected from ``input_ids`` via
    ``image_tiles()`` (a real contiguous 64-row run) -- ALL of a tile's rows
    are dropped from the sequence regardless of pruning, so this can't change.

    ``compressor_rows_per_sample`` (optional): one ``[T, M]`` absolute-position
    subset of each sample's own (auto-detected) tile rows, fed to the
    compressor INSTEAD of the full tile (``None`` entries, or omitting this
    arg, use the full tile -- the existing unpruned behavior). This is what
    lets a caller pre-prune each tile to M < tile_tokens rows (e.g.
    ``target_attn_prune.py``) before the learned compressor ever sees them:
    the summary's output slot is then chosen from the M kept rows, not the
    original 64, but every row outside that M-subset is still removed here,
    same as an unpruned tile's other 64-k rows always were.
    """
    cfg = compressor.cfg
    tile_tokens = cfg.tile_tokens
    bsz, seq_len = input_ids.shape
    device = input_ids.device

    if position_ids is None:
        base_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
    elif position_ids.ndim == 3:
        raise NotImplementedError(
            "MRoPE position ids with row compression are not supported yet"
        )
    else:
        base_positions = position_ids.view(bsz, seq_len).to(device)

    valid = (
        attention_mask.view(bsz, seq_len).bool()
        if attention_mask is not None
        else torch.ones(bsz, seq_len, dtype=torch.bool, device=device)
    )

    keep_indices, summaries, kept_slot_masks, lengths = [], [], [], []
    for b in range(bsz):
        tiles = image_tiles(input_ids[b], valid[b], image_token_id, tile_tokens)
        if tiles is None:
            keep_indices.append(torch.arange(seq_len, device=device))
            summaries.append(None)
            kept_slot_masks.append(torch.zeros(seq_len, dtype=torch.bool, device=device))
            lengths.append(seq_len)
            continue

        if loss_mask[b][tiles.reshape(-1)].any():
            raise ValueError(
                "image rows carry supervision; dropping them would drop loss "
                "terms. Check the loss mask before enabling row compression"
            )
        n_tiles = tiles.shape[0]
        compress_rows = tiles
        if compressor_rows_per_sample is not None and compressor_rows_per_sample[b] is not None:
            compress_rows = compressor_rows_per_sample[b]
        n_rows = compress_rows.shape[1]
        # Positions of the k rows each tile keeps, in tile order. Offsets are
        # computed against the ACTUAL row count fed to the compressor
        # (n_rows), which is tile_tokens unless a caller pre-pruned via
        # compressor_rows_per_sample -- the compressor's own params don't
        # depend on tile_tokens, only the evenly-spaced-centres slot
        # convention does (VisRowCompressor.slot_offsets).
        offsets = torch.as_tensor(
            compressor.slot_offsets(cfg.num_queries, tile_tokens=n_rows), device=device
        )
        slots = compress_rows.index_select(1, offsets).reshape(-1)

        out, _ = compressor(
            hidden_states[b][compress_rows].unsqueeze(0), n_tiles, expected_rows=n_rows
        )
        summaries.append(out[0].reshape(slots.numel(), -1))

        keep = torch.ones(seq_len, dtype=torch.bool, device=device)
        keep[tiles.reshape(-1)] = False  # drop the FULL original tile, pruned or not
        keep[slots] = True
        idx = keep.nonzero().flatten()
        keep_indices.append(idx)
        # Where the kept slots land in the compacted sequence.
        is_slot = torch.zeros(seq_len, dtype=torch.bool, device=device)
        is_slot[slots] = True
        kept_slot_masks.append(is_slot[idx])
        lengths.append(idx.numel())

    new_len = max(lengths)
    pad_index = seq_len - 1

    def _gather(tensor):
        rows = []
        for b, idx in enumerate(keep_indices):
            pad = new_len - idx.numel()
            if pad:
                idx = torch.cat([idx, idx.new_full((pad,), pad_index)])
            rows.append(tensor[b][idx])
        return torch.stack(rows)

    out_hidden = _gather(hidden_states)
    out_ids = _gather(input_ids)
    out_logits = _gather(target_logits)
    out_loss = _gather(loss_mask)
    out_pos = _gather(base_positions)
    out_attn = (
        _gather(valid.to(input_ids.dtype))
        if attention_mask is not None
        else torch.ones(bsz, new_len, dtype=input_ids.dtype, device=device)
    )
    ins = torch.stack(
        [torch.cat([m, m.new_zeros(new_len - m.numel())]) for m in kept_slot_masks]
    )

    for b, rows in enumerate(summaries):
        if rows is not None:
            out_hidden[b, ins[b]] = rows.to(out_hidden.dtype)
    # A summary row is never supervised, and neither was the image row it
    # replaced -- asserted above.
    out_loss = out_loss.masked_fill(ins, 0)
    # Right padding: outside the batch, attend to nothing and learn nothing.
    tail = torch.arange(new_len, device=device).unsqueeze(0) >= torch.tensor(
        lengths, device=device
    ).unsqueeze(1)
    out_attn = out_attn.masked_fill(tail, 0)
    out_loss = out_loss.masked_fill(tail, 0)

    return {
        "hidden_states": out_hidden,
        "input_ids": out_ids,
        "target_logits": out_logits,
        "loss_mask": out_loss,
        "attention_mask": out_attn,
        "position_ids": out_pos,
        "compressed_rows": ins,
    }
