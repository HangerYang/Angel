# Copyright 2025 Tencent Inc. All Rights Reserved.
"""Recompute Llama self-attention probs for selected target layers.

Used for attention-matching losses during Eagle training. Avoids requiring
``attn_implementation=eager`` / ``output_attentions`` on the full target
forward (which is often flash/SDPA and does not return weights).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import torch


def _resolve_text_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "text_model"):
        return model.model.text_model
    if hasattr(model, "text_model"):
        return model.text_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise ValueError("Could not locate Llama text tower (text_model / model.layers)")


@torch.no_grad()
def compute_llama_layer_attention_probs(
    model: torch.nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention probs for one Llama decoder layer.

    Args:
        model: HF causal LM / VLM containing a Llama text tower.
        layer_idx: Zero-based decoder layer index.
        hidden_states: Input residual to that layer ``[B, S, H]``
            (HF ``outputs.hidden_states[layer_idx]``).
        attention_mask: Optional ``[B, S]`` (1 = keep) padding mask.
        position_ids: Optional ``[B, S]`` positions for RoPE.

    Returns:
        Softmax attention ``[B, num_heads, S, S]`` (float32).
    """
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    text_model = _resolve_text_model(model)
    layer = text_model.layers[layer_idx]
    attn = layer.self_attn
    bsz, q_len, _ = hidden_states.shape
    device = hidden_states.device

    h = layer.input_layernorm(hidden_states)
    num_heads = attn.config.num_attention_heads
    num_kv_heads = attn.config.num_key_value_heads
    head_dim = attn.head_dim

    q = attn.q_proj(h).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    k = attn.k_proj(h).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    if position_ids is None:
        position_ids = torch.arange(q_len, device=device, dtype=torch.long)[None, :].expand(
            bsz, -1
        )
    elif position_ids.dim() == 1:
        position_ids = position_ids[None, :].expand(bsz, -1)
    elif position_ids.dim() == 3:
        # Some VLMs pass [3, B, S] or [B, 3, S]; take first plane.
        if position_ids.shape[0] == 3:
            position_ids = position_ids[0]
        else:
            position_ids = position_ids[:, 0]

    cos, sin = text_model.rotary_emb(h, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = repeat_kv(k, attn.num_key_value_groups)

    scores = torch.matmul(q.float(), k.float().transpose(2, 3)) / math.sqrt(head_dim)
    causal = torch.ones(q_len, q_len, device=device, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
    if attention_mask is not None:
        # attention_mask: 1 = keep. Broadcast over heads/queries.
        key_keep = attention_mask.to(device=device, dtype=torch.bool)
        scores = scores.masked_fill(~key_keep[:, None, None, :], torch.finfo(scores.dtype).min)

    return torch.softmax(scores, dim=-1)


@torch.no_grad()
def compute_target_layer_attentions(
    model: torch.nn.Module,
    hidden_states_tuple: Sequence[torch.Tensor],
    layer_ids: Iterable[int],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    """Map ``layer_id -> [B, H, S, S]`` attention probs for the requested layers."""
    out: Dict[int, torch.Tensor] = {}
    for lid in layer_ids:
        lid = int(lid)
        if lid < 0 or lid + 1 >= len(hidden_states_tuple):
            raise IndexError(
                f"attn_match target layer {lid} out of range for "
                f"hidden_states len={len(hidden_states_tuple)}"
            )
        # HF: hidden_states[i] is the input to decoder layer i.
        out[lid] = compute_llama_layer_attention_probs(
            model,
            lid,
            hidden_states_tuple[lid],
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    return out


def attention_kl_loss(
    draft_attn: torch.Tensor,
    target_attn: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    key_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL(draft || target) over attention maps.

    Args:
        draft_attn / target_attn: ``[B, H, S, S]`` (or draft ``[B, H, S, S_k]``;
            only the leading ``S`` key positions are matched).
        query_mask: optional ``[B, S]`` or ``[B, S, 1]`` (1 = include query).
        key_mask: optional ``[B, S]`` (1 = include key). When set (e.g. image
            token positions), both maps are restricted to those keys and
            **renormalized** before KL — i.e. match attention *on images*,
            not the full causal distribution.
    """
    if draft_attn.shape[:3] != target_attn.shape[:3]:
        raise ValueError(
            f"attn shape mismatch draft={tuple(draft_attn.shape)} "
            f"target={tuple(target_attn.shape)}"
        )
    s = target_attn.shape[-1]
    d = draft_attn[..., :s].float().clamp_min(eps)
    t = target_attn.float().clamp_min(eps)

    if key_mask is not None:
        km = key_mask
        while km.ndim > 2:
            km = km.squeeze(-1)
        km = km.to(device=d.device, dtype=torch.bool)
        if not km.any():
            return d.new_zeros(())
        km4 = km[:, None, None, :]  # [B,1,1,S]
        d_img = torch.where(km4, d, torch.zeros_like(d))
        t_img = torch.where(km4, t, torch.zeros_like(t))
        d_sum = d_img.sum(dim=-1, keepdim=True)
        t_sum = t_img.sum(dim=-1, keepdim=True)
        # Queries with no mass on selected keys (e.g. before first image) are dropped.
        valid = (t_sum.squeeze(-1) > eps) & (d_sum.squeeze(-1) > eps)  # [B,H,S]
        d_img = d_img / d_sum.clamp_min(eps)
        t_img = t_img / t_sum.clamp_min(eps)
        # Avoid 0*log(0) on non-image keys: only evaluate log on selected keys.
        d_log = torch.where(km4, d_img.clamp_min(eps).log(), torch.zeros_like(d_img))
        t_log = torch.where(km4, t_img.clamp_min(eps).log(), torch.zeros_like(t_img))
        kl = torch.where(km4, d_img * (d_log - t_log), torch.zeros_like(d_img))
        kl = kl.sum(dim=-1)  # [B,H,S]
        kl = kl.masked_fill(~valid, 0.0)
        kl = kl.sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0)  # [B,S]
        q_valid = valid.any(dim=1)  # [B,S]
    else:
        d = d / d.sum(dim=-1, keepdim=True).clamp_min(eps)
        kl = torch.nn.functional.kl_div(t.log(), d, reduction="none", log_target=False)
        kl = kl.sum(dim=-1).mean(dim=1).clamp_min(0.0)  # [B, S]
        q_valid = None
    if query_mask is None and q_valid is None:
        return kl.mean()

    mask = None
    if query_mask is not None:
        mask = query_mask
        while mask.ndim > 2:
            mask = mask.squeeze(-1)
        mask = mask.to(dtype=kl.dtype, device=kl.device)
    if q_valid is not None:
        qv = q_valid.to(dtype=kl.dtype, device=kl.device)
        mask = qv if mask is None else mask * qv
    denom = mask.sum().clamp_min(1.0)
    return (kl * mask).sum() / denom
