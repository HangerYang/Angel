#!/usr/bin/env python3
"""Simulate HiViS-style early visual-token skipping for progressive SmolVLM Eagle.

This does not patch vLLM. It runs the HF draft layer stack in teacher-forced
mode and compares:

  full:       normal progressive Eagle over image + text tokens
  l0_text:    layer 0 runs only non-image tokens, then image positions are
              reinserted before layer 1 using aux0 target HS

The goal is a quick quality/speed/acceptance proxy before touching the real
decode path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from angelslim.compressor.speculative.train.data.chat_templates import (  # noqa: E402
    ChatTemplateType,
)
from angelslim.compressor.speculative.train.data.dataset_builder.online_dataset_builder import (  # noqa: E402
    OnlineSmolVLMDatasetBuilder,
)
from angelslim.compressor.speculative.train.models.draft import (  # noqa: E402
    DraftModelConfig,
    create_draft_model,
)
from angelslim.compressor.speculative.utils import padding  # noqa: E402


def _load_state_dict(ckpt: Path) -> Dict[str, torch.Tensor]:
    safes = sorted(ckpt.glob("*.safetensors"))
    if safes:
        from safetensors.torch import load_file

        out: Dict[str, torch.Tensor] = {}
        for path in safes:
            out.update(load_file(str(path)))
        return out
    bin_path = ckpt / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No safetensors or pytorch_model.bin under {ckpt}")


def _load_draft(ckpt: Path, device: torch.device, dtype: torch.dtype):
    cfg = DraftModelConfig.from_file(ckpt / "config.json")
    draft = create_draft_model(cfg)
    missing, unexpected = draft.load_state_dict(_load_state_dict(ckpt), strict=False)
    missing = [k for k in missing if not (k.startswith("t2d") or k.startswith("d2t"))]
    if missing:
        print(f"WARNING missing draft keys: {len(missing)} e.g. {missing[:5]}")
    if unexpected:
        print(f"WARNING unexpected draft keys: {len(unexpected)} e.g. {unexpected[:5]}")
    draft.to(device=device, dtype=dtype)
    draft.eval()
    for p in draft.parameters():
        p.requires_grad_(False)
    return draft, cfg


def _prepare_mask(
    draft,
    attention_mask_2d: torch.Tensor,
    seq_len: int,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    return draft.prepare_decoder_attention_mask(
        attention_mask_2d,
        (attention_mask_2d.shape[0], seq_len),
        hidden_states,
        0,
    )


def _target_distribution_loss(
    draft,
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    target_max_token = target_logits.argmax(-1)
    valid_target = (target_max_token >= 0) & (target_max_token < draft.t2d.shape[0])
    safe_target = target_max_token.clamp(0, draft.t2d.shape[0] - 1)
    target_mask = (draft.t2d[safe_target] & valid_target)[..., None].int()
    position_mask = target_mask * loss_mask
    target_head = target_logits[..., draft.t2d].float()
    target_p = torch.softmax(target_head, dim=-1).detach()
    out_logp = torch.log_softmax(logits.float(), dim=-1)
    denom = position_mask.sum().clamp_min(1.0)
    loss = -(position_mask * target_p * out_logp).sum() / denom
    correct = (logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)
    acc = float(correct.sum().item() / float(denom.item()))
    return loss, acc


def _target_agreement_mask(
    draft,
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return draft-vocab argmax agreement and valid CE positions."""

    target_max_token = target_logits.argmax(-1)
    valid_target = (target_max_token >= 0) & (target_max_token < draft.t2d.shape[0])
    safe_target = target_max_token.clamp(0, draft.t2d.shape[0] - 1)
    target_mask = draft.t2d[safe_target] & valid_target
    position_mask = target_mask & (loss_mask.squeeze(-1) > 0)
    target_head = target_logits[..., draft.t2d].float()
    target_idx = target_head.argmax(dim=-1)
    draft_idx = logits.argmax(dim=-1)
    return draft_idx.eq(target_idx) & position_mask, position_mask


def _summarize_acceptance_proxy(
    matches: List[torch.Tensor],
    valids: List[torch.Tensor],
) -> Dict[str, float]:
    if not matches:
        return {}
    base = valids[0].bool()
    alive = base.clone()
    accepted = torch.zeros_like(base, dtype=torch.float32)
    step_agreements: Dict[str, float] = {}
    for idx, (match, valid) in enumerate(zip(matches, valids)):
        valid = valid.bool()
        match = match.bool()
        denom = valid.float().sum().clamp_min(1.0)
        step_agreements[f"step{idx}_argmax_agreement"] = float(
            (match & valid).float().sum().item() / float(denom.item())
        )
        ok = alive & valid & match
        accepted += ok.float()
        alive = ok
    denom = base.float().sum().clamp_min(1.0)
    mean_accepted = float(accepted.sum().item() / float(denom.item()))
    out = {
        "mean_accepted_tokens_proxy": mean_accepted,
        "mean_acceptance_length_proxy": 1.0 + mean_accepted,
        "valid_positions": float(denom.item()),
    }
    out.update(step_agreements)
    return out


@torch.no_grad()
def _full_forward(
    draft,
    aux_concat: torch.Tensor,
    input_ids_shifted: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    out = draft(
        hidden_states=aux_concat,
        input_ids=input_ids_shifted,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        output_hidden_states=False,
        output_attentions=False,
    )
    return out["logits"] if isinstance(out, dict) else out[1]


@torch.no_grad()
def _compact_l0_forward(
    draft,
    aux_concat: torch.Tensor,
    input_ids_shifted: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    image_mask: torch.Tensor,
) -> torch.Tensor:
    if input_ids_shifted.shape[0] != 1:
        raise ValueError("Simulation currently expects batch_size=1.")
    if not getattr(draft, "progressive_staged", False):
        raise ValueError("compact_l0 simulation is for progressive_staged Eagle.")

    dtype = next(draft.parameters()).dtype
    aux_concat = aux_concat.to(dtype=dtype)
    if aux_concat.shape[-1] != draft.hidden_size:
        h0 = draft.combine_hidden_states(aux_concat)
    else:
        h0 = aux_concat
    aux = draft._aux_inject
    if aux is None:
        raise RuntimeError("draft.combine_hidden_states did not populate aux injects.")

    embeds = draft.embed_input_ids(input_ids_shifted).to(dtype)
    text_idx = (~image_mask[0]).nonzero(as_tuple=False).flatten()
    if text_idx.numel() == 0:
        raise ValueError("No text tokens remain after image-token removal.")

    h0_text = h0[:, text_idx, :]
    emb_text = embeds[:, text_idx, :]
    attn_text_2d = attention_mask[:, text_idx]
    # HiViS-style compact prefill: the drafter sees a text-only sequence, so
    # positions are contiguous in that compact sequence rather than sparse
    # original prompt positions that include hidden visual tokens.
    pos_text = torch.arange(
        int(text_idx.numel()), device=input_ids_shifted.device, dtype=torch.long
    ).view(1, -1)

    attn_text = _prepare_mask(draft, attn_text_2d, int(text_idx.numel()), h0_text)
    cache = draft.init_cache_hidden()
    layer0_out, _ = draft.layers[0](
        emb_text,
        h0_text,
        cache[0],
        attn_text,
        pos_text,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        inject=None,
    )
    h0_text_out = layer0_out[0]

    full_h = torch.empty_like(h0)
    full_h[:, text_idx, :] = h0_text_out
    full_h[:, image_mask[0], :] = aux[0][:, image_mask[0], :]

    full_attn = _prepare_mask(draft, attention_mask, input_ids_shifted.shape[1], full_h)
    if position_ids is None:
        full_position_ids = torch.arange(
            input_ids_shifted.shape[1], device=input_ids_shifted.device, dtype=torch.long
        ).view(1, -1)
    else:
        full_position_ids = position_ids
    cache = draft.init_cache_hidden()
    hidden = full_h
    layer_outs = [full_h]
    for layer_idx in range(1, len(draft.layers)):
        layer_out, _ = draft.layers[layer_idx](
            None,
            hidden,
            cache[layer_idx],
            full_attn,
            full_position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            inject=aux[layer_idx],
        )
        hidden = layer_out[0]
        layer_outs.append(hidden)
    draft._last_layer_outs = layer_outs
    return draft.compute_logits(hidden)


@torch.no_grad()
def _feedback_acceptance_proxy(
    draft,
    aux_concat: torch.Tensor,
    input_ids_shifted: torch.Tensor,
    target_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    image_mask: torch.Tensor,
    compact_l0: bool,
    steps: int,
) -> Dict[str, float]:
    matches: List[torch.Tensor] = []
    valids: List[torch.Tensor] = []
    cur_hidden = aux_concat
    cur_input_ids = input_ids_shifted
    cur_target_logits = target_logits
    cur_loss_mask = loss_mask

    for idx in range(steps):
        if idx == 0 and compact_l0:
            logits = _compact_l0_forward(
                draft,
                cur_hidden,
                cur_input_ids,
                attention_mask,
                position_ids,
                image_mask,
            )
        else:
            logits = _full_forward(
                draft,
                cur_hidden,
                cur_input_ids,
                attention_mask,
                position_ids,
            )
        match, valid = _target_agreement_mask(draft, logits, cur_target_logits, cur_loss_mask)
        matches.append(match.detach())
        valids.append(valid.detach())

        if idx < steps - 1:
            cur_input_ids = padding(cur_input_ids, left=False)
            cur_target_logits = padding(cur_target_logits, left=False)
            cur_loss_mask = padding(cur_loss_mask, left=False)
            seed = draft.take_progressive_draft_feedback()
            if seed is None:
                raise RuntimeError("progressive feedback returned None during proxy loop")
            cur_hidden = seed

    return _summarize_acceptance_proxy(matches, valids)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    p.add_argument(
        "--draft-checkpoint",
        default=str(
            _REPO_ROOT
            / "output/progressive_layer_group_tests/layers_1_15_23_attn_match_img/checkpoint-66466"
        ),
    )
    p.add_argument(
        "--data-path",
        default=str(_REPO_ROOT / "dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl"),
    )
    p.add_argument("--output", default=str(Path(__file__).resolve().parent / "outputs" / "hivis_progressive_sim.json"))
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--sample-pool-size", type=int, default=None)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--acceptance-steps", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Loading target {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    target = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    print(f"Loading draft {args.draft_checkpoint}")
    draft, cfg = _load_draft(Path(args.draft_checkpoint), device, dtype)
    aux_ids = tuple(int(x) for x in getattr(cfg, "aux_hidden_states_layer_ids", [1, 15, 23]))
    image_token_id = int(getattr(target.config, "image_token_id", 49190))

    builder = OnlineSmolVLMDatasetBuilder(
        tokenizer=processor,
        max_length=args.max_length,
        shuffle_seed=args.seed,
        chat_template_type=ChatTemplateType.SMOLVLM,
        display=False,
    )
    pool = args.sample_pool_size or max(args.max_samples * 4, args.max_samples)
    ds = builder.build_dataset(
        args.data_path,
        num_proc=args.num_proc,
        shuffle=True,
        sample_num=pool,
        load_from_cache_file=False,
    )
    ds = ds.filter(
        lambda batch: [bool(p) and p != "[]" for p in batch["image_paths"]],
        batched=True,
        num_proc=args.num_proc,
        load_from_cache_file=False,
        desc="Filtering image-only samples",
    )
    if len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=builder.get_data_collator())

    metrics: Dict[str, List[float]] = {
        "full_loss": [],
        "drop_loss": [],
        "full_acc": [],
        "drop_acc": [],
        "drop_kl_vs_full": [],
        "full_ms": [],
        "drop_ms": [],
        "seq_len": [],
        "image_tokens": [],
        "text_tokens": [],
        "l0_attention_work_ratio": [],
        "three_layer_attention_work_ratio": [],
    }

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)[..., None]
        image_mask = input_ids.eq(image_token_id)
        if not image_mask.any():
            continue

        fwd: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }
        if "pixel_values" in batch and batch["pixel_values"] is not None:
            fwd["pixel_values"] = batch["pixel_values"].to(device)
        if "pixel_attention_mask" in batch and batch["pixel_attention_mask"] is not None:
            fwd["pixel_attention_mask"] = batch["pixel_attention_mask"].to(device)
        with torch.no_grad():
            target_out = target(**fwd)

        aux_concat = torch.cat([target_out.hidden_states[i + 1] for i in aux_ids], dim=-1)
        target_logits = padding(target_out.logits, left=False).to(device)
        input_ids_shifted = padding(input_ids, left=False)
        position_ids = None
        if hasattr(target_out, "position_ids"):
            position_ids = getattr(target_out, "position_ids")

        _sync(device)
        t0 = time.perf_counter()
        full_logits = _full_forward(draft, aux_concat, input_ids_shifted, attention_mask, position_ids)
        _sync(device)
        metrics["full_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        drop_logits = _compact_l0_forward(
            draft, aux_concat, input_ids_shifted, attention_mask, position_ids, image_mask
        )
        _sync(device)
        metrics["drop_ms"].append((time.perf_counter() - t0) * 1000)

        full_loss, full_acc = _target_distribution_loss(draft, full_logits, target_logits, loss_mask)
        drop_loss, drop_acc = _target_distribution_loss(draft, drop_logits, target_logits, loss_mask)
        metrics["full_loss"].append(float(full_loss.item()))
        metrics["drop_loss"].append(float(drop_loss.item()))
        metrics["full_acc"].append(full_acc)
        metrics["drop_acc"].append(drop_acc)

        q = torch.softmax(full_logits.float(), dim=-1).clamp_min(1e-8)
        p_drop = torch.softmax(drop_logits.float(), dim=-1).clamp_min(1e-8)
        kl = (q * (q.log() - p_drop.log())).sum(dim=-1, keepdim=True)
        metrics["drop_kl_vs_full"].append(
            float((kl * loss_mask).sum().item() / loss_mask.sum().clamp_min(1).item())
        )

        full_proxy = _feedback_acceptance_proxy(
            draft,
            aux_concat,
            input_ids_shifted,
            target_logits,
            loss_mask,
            attention_mask,
            position_ids,
            image_mask,
            compact_l0=False,
            steps=args.acceptance_steps,
        )
        drop_proxy = _feedback_acceptance_proxy(
            draft,
            aux_concat,
            input_ids_shifted,
            target_logits,
            loss_mask,
            attention_mask,
            position_ids,
            image_mask,
            compact_l0=True,
            steps=args.acceptance_steps,
        )
        for key, value in full_proxy.items():
            metrics.setdefault(f"full_{key}", []).append(float(value))
        for key, value in drop_proxy.items():
            metrics.setdefault(f"drop_{key}", []).append(float(value))

        s = int(attention_mask.sum().item())
        v = int((image_mask & attention_mask.bool()).sum().item())
        t = max(s - v, 1)
        metrics["seq_len"].append(float(s))
        metrics["image_tokens"].append(float(v))
        metrics["text_tokens"].append(float(t))
        metrics["l0_attention_work_ratio"].append(float((t * t) / max(s * s, 1)))
        metrics["three_layer_attention_work_ratio"].append(float((t * t + 2 * s * s) / max(3 * s * s, 1)))

        if (step + 1) % 10 == 0 or (step + 1) == len(ds):
            print(f"processed {step + 1}/{len(ds)}")

    summary = {k: _mean(v) for k, v in metrics.items()}
    summary.update(
        {
            "n_samples": len(metrics["full_loss"]),
            "acceptance_steps": args.acceptance_steps,
            "model_path": args.model_path,
            "draft_checkpoint": str(args.draft_checkpoint),
            "aux_hidden_states_layer_ids": list(aux_ids),
            "image_token_id": image_token_id,
            "note": (
                "Timing is HF eager simulation, not vLLM. attention_work_ratio estimates "
                "self-attention score-matrix work only: lower is better. "
                "drop mode fills removed image positions with target HS from the first aux layer before layer1. "
                "Acceptance length is a teacher-forced proxy, not recursive vLLM verification."
            ),
        }
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw": metrics}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
