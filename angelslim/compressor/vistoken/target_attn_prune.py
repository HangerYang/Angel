"""Attention-guided image-row pruning, in front of the learned compressor.

vistoken's row compressor (``row_compressor.py``) routes over a tile's rows
with a *content-blind* learned query -- at k=1 it has to average almost all
64 rows regardless of what's actually relevant to the question being asked.
This module adds a cheap, content-aware pre-filter: use the TARGET model's
own attention (which rows it actually looked at while answering) to drop
clearly-irrelevant rows first, so the existing compressor only has to route
over a smaller, curated set.

Runs at the same point ``compress_image_rows`` already does -- after the
target forward, before the left shift -- and produces the same kind of
output: an explicit ``[T, M]`` absolute-position tile index tensor that
``compress_image_rows`` can consume directly (bypassing its own
contiguous-64-run auto-detection, which real pruned rows would fail).

Getting attention weights the obvious way (``output_attentions=True``) does
not work here: training loads the target with ``attn_implementation=
"flash_attention_2"`` (``target_model_wrapper.py``), which does not support
returning attention matrices, and switching the whole target to eager
attention would change memory/perf for the entire forward just for this.
Instead, ``TargetQKCapture`` hooks the raw ``q_proj``/``k_proj`` outputs of
the relevant decoder layers during the SAME forward that already produces
the aux hidden states, and a small side matmul (restricted to the query and
image-row positions we actually care about) stands in for real attention --
never a full-sequence attention matrix, never a second target forward.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .splice import image_tiles


class TargetQKCapture:
    """Forward hooks capturing q_proj/k_proj OUTPUT on a set of decoder
    layers, for the single forward this is used as a context manager around.

    Captures the raw projection output (pre-RoPE, pre-head-split) -- the
    scores built from it are a ranking signal for row pruning, not a
    reproduction of the model's real attention output, so exact RoPE/GQA
    numerics are not required.
    """

    def __init__(self, layers: Sequence[nn.Module], layer_ids: Sequence[int]):
        self.layers = layers
        self.layer_ids = list(layer_ids)
        self.captured: Dict[int, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "TargetQKCapture":
        for layer_id in self.layer_ids:
            attn = self.layers[layer_id].self_attn
            self._handles.append(
                attn.q_proj.register_forward_hook(self._make_hook(layer_id, "q"))
            )
            self._handles.append(
                attn.k_proj.register_forward_hook(self._make_hook(layer_id, "k"))
            )
        return self

    def _make_hook(self, layer_id: int, which: str):
        def hook(_module, _inputs, output):
            q, k = self.captured.get(layer_id, (None, None))
            if which == "q":
                q = output.detach()
            else:
                k = output.detach()
            self.captured[layer_id] = (q, k)

        return hook

    def __exit__(self, *exc) -> bool:
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


@torch.no_grad()
def image_row_scores(
    qk_capture: TargetQKCapture,
    layer_ids: Sequence[int],
    query_positions: torch.Tensor,
    image_positions: torch.Tensor,
    head_dim: int,
    sample_idx: int = 0,
) -> torch.Tensor:
    """[n_image_rows] relevance score for one sample, mean over heads, over
    the 9 aux layers, and over every loss_mask==1 query position (training
    has no "round" concept -- ``compress_image_rows`` runs once on the whole
    teacher-forced sequence -- so this aggregates over the whole supervised
    span rather than a single "latest verified token").

    K is mean-pooled over whatever (possibly fewer, under GQA) KV heads it
    has before the dot product, then broadcast against every query head --
    a ranking signal only, so exact query/kv head grouping isn't needed.

    ``sample_idx`` selects the row of the captured ``[B, S, H*d]`` projection
    output to score. The hooks fire once for the whole batch, so this must be
    the same ``b`` the caller took ``input_ids[b]`` / ``loss_mask[b]`` from.
    """
    per_layer = []
    for layer_id in layer_ids:
        q, k = qk_capture.captured[layer_id]
        if q is None or k is None:
            raise RuntimeError(f"no captured q/k for layer {layer_id}; was the hook active?")
        seq_len = q.shape[1]
        # Select the sample FIRST: these are [B, S, H*d] for the whole batch,
        # and folding B into the reshape silently scrambles positions into the
        # head axis instead of failing.
        q = q[sample_idx].view(seq_len, -1, head_dim).index_select(0, query_positions)  # [Nq, Hq, d]
        k = k[sample_idx].view(seq_len, -1, head_dim).index_select(0, image_positions)  # [Ni, Hk, d]
        k_mean = k.mean(dim=1)  # [Ni, d] -- collapse KV heads
        scores = torch.einsum("qhd,id->qhi", q.float(), k_mean.float()) / (head_dim**0.5)
        scores = scores.mean(dim=1)  # mean over query heads -> [Nq, Ni]
        scores = torch.softmax(scores, dim=-1)  # normalize over image rows, per query pos
        per_layer.append(scores.mean(dim=0))  # mean over query positions -> [Ni]
    return torch.stack(per_layer, dim=0).mean(dim=0)  # mean over layers -> [Ni]


def prune_tiles_by_score(
    tiles: torch.Tensor,
    scores_by_position: torch.Tensor,
    group_size: int,
    keep_m: int,
    mode: str = "target_attn",
) -> torch.Tensor:
    """``tiles``: ``[T, tile_tokens]`` absolute positions from ``image_tiles()``.
    ``scores_by_position``: full-sequence-length tensor; only entries at
    ``tiles``' own positions are read.

    Groups each tile's rows into fixed ``group_size`` chunks and keeps the
    top-``keep_m`` (by score) absolute positions per chunk, in ascending
    absolute-position order (matching what ``image_tiles()`` itself returns,
    so the result is a drop-in ``tiles`` arg for ``compress_image_rows``).

    ``mode="random"``: same grouping/keep_m, but a random draw instead of the
    attention score -- the "does pruning at all help, independent of the
    signal used to choose rows" ablation.
    """
    n_tiles, tile_tokens = tiles.shape
    if tile_tokens % group_size != 0:
        raise ValueError(f"tile_tokens={tile_tokens} not divisible by group_size={group_size}")
    if not 0 < keep_m <= group_size:
        raise ValueError(f"keep_m={keep_m} must be in 1..group_size={group_size}")
    n_groups = tile_tokens // group_size

    groups = tiles.view(n_tiles, n_groups, group_size)
    if mode == "random":
        rank = torch.rand(n_tiles, n_groups, group_size, device=tiles.device)
    elif mode == "target_attn":
        rank = scores_by_position[groups]
    else:
        raise ValueError(f"mode must be target_attn|random, got {mode!r}")

    topk_idx = rank.topk(keep_m, dim=-1).indices  # [T, n_groups, M]
    kept = torch.gather(groups, -1, topk_idx)
    kept, _ = kept.sort(dim=-1)  # keep ascending absolute-position order within each group
    return kept.reshape(n_tiles, n_groups * keep_m)


def prune_sample_image_rows(
    qk_capture: Optional[TargetQKCapture],
    layer_ids: Sequence[int],
    head_dim: int,
    input_ids_row: torch.Tensor,
    valid_row: torch.Tensor,
    loss_mask_row: torch.Tensor,
    image_token_id: int,
    tile_tokens: int,
    group_size: int,
    keep_m: int,
    mode: str,
    sample_idx: int = 0,
) -> Optional[torch.Tensor]:
    """One sample's pruned ``[T, n_groups*keep_m]`` absolute-position tile
    index tensor, or ``None`` if the sample has no image rows (mirrors
    ``image_tiles()``'s own contract). ``mode="none"`` returns the
    unpruned ``[T, tile_tokens]`` tiles unchanged (the existing vistoken-k1
    behavior, for a common code path across all four eval variants).
    """
    tiles = image_tiles(input_ids_row, valid_row, image_token_id, tile_tokens)
    if tiles is None or mode == "none":
        return tiles

    seq_len = input_ids_row.shape[0]
    device = input_ids_row.device
    if mode == "random":
        scores = torch.zeros(seq_len, device=device)  # unused by prune_tiles_by_score
    else:
        query_positions = loss_mask_row.nonzero().flatten()
        if query_positions.numel() == 0:
            # No supervised positions in this sample (e.g. an all-image pad
            # row) -- nothing to rank by; fall back to keeping everything.
            return tiles
        image_positions = tiles.reshape(-1)
        assert qk_capture is not None, "target_attn mode requires a TargetQKCapture"
        row_scores = image_row_scores(
            qk_capture, layer_ids, query_positions, image_positions, head_dim, sample_idx
        )
        scores = torch.zeros(seq_len, device=device, dtype=row_scores.dtype)
        scores[image_positions] = row_scores

    return prune_tiles_by_score(tiles, scores, group_size, keep_m, mode=mode)
