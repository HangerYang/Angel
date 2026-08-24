# Copyright 2025 Tencent Inc. All Rights Reserved.
"""HiViS helpers: hide visual tokens from the drafter.

Matches the official prune rule from
https://github.com/lnn-ops/HiViS (Qwen vision-span / LLaVA image-token drop).

The target still sees the full multimodal sequence. After the target forward,
the draft sequence is compacted by dropping visual tokens (and, when start/end
ids are set, the whole vision span).
"""

from __future__ import annotations

from typing import Optional

import torch


def build_hivis_keep_mask(
    input_ids: torch.Tensor,
    image_token_id: Optional[int] = None,
    vision_start_token_id: Optional[int] = None,
    vision_end_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Boolean keep mask for one 1-D token row (True = keep for the drafter)."""
    if input_ids.ndim != 1:
        raise ValueError(f"expected 1-D input_ids, got shape {tuple(input_ids.shape)}")
    keep = torch.ones_like(input_ids, dtype=torch.bool)

    if vision_start_token_id is not None:
        starts = (input_ids == int(vision_start_token_id)).nonzero(as_tuple=False)
        if starts.numel() > 0:
            start = int(starts[0].item())
            if vision_end_token_id is not None:
                ends = (input_ids[start:] == int(vision_end_token_id)).nonzero(
                    as_tuple=False
                )
                if ends.numel() > 0:
                    end = start + int(ends[0].item())
                elif image_token_id is not None:
                    count = int((input_ids == int(image_token_id)).sum().item())
                    end = min(start + count + 2, input_ids.numel() - 1)
                else:
                    end = start
            elif image_token_id is not None:
                count = int((input_ids == int(image_token_id)).sum().item())
                end = min(start + count + 2, input_ids.numel() - 1)
            else:
                end = start
            keep[start : end + 1] = False
            return keep

    if image_token_id is not None:
        keep = input_ids != int(image_token_id)
    return keep


def prune_hidden_and_ids(
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_token_id: Optional[int] = None,
    vision_start_token_id: Optional[int] = None,
    vision_end_token_id: Optional[int] = None,
    pad_token_id: int = 0,
):
    """Drop visual tokens from a batch of draft tensors (right-pad to max keep).

    Returns compact ``input_ids``, ``hidden_states``, ``attention_mask``, and
    the number of dropped tokens per row.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"expected [B,S] input_ids, got {tuple(input_ids.shape)}")
    keep_masks = [
        build_hivis_keep_mask(
            row,
            image_token_id=image_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
        )
        for row in input_ids
    ]
    dropped = [int((~mask).sum().item()) for mask in keep_masks]
    if all(mask.all().item() for mask in keep_masks):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids, hidden_states, attention_mask, dropped

    max_len = max(int(mask.sum().item()) for mask in keep_masks)
    max_len = max(max_len, 1)
    compact_ids = input_ids.new_full((input_ids.shape[0], max_len), int(pad_token_id))
    compact_hs = hidden_states.new_zeros(
        (hidden_states.shape[0], max_len, hidden_states.shape[-1])
    )
    compact_mask = input_ids.new_zeros((input_ids.shape[0], max_len))
    src_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
    for idx, mask in enumerate(keep_masks):
        n = int(mask.sum().item())
        if n == 0:
            continue
        compact_ids[idx, :n] = input_ids[idx][mask]
        compact_hs[idx, :n] = hidden_states[idx][mask]
        compact_mask[idx, :n] = src_mask[idx][mask]
    return compact_ids, compact_hs, compact_mask, dropped
