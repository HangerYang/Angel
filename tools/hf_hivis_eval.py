#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""HF HiViS speculative eval: hide visual tokens from the drafter.

The target prefills the full multimodal prompt. The draft sees only the compact
text sequence (image tokens removed), matching
``hivis_remove_image_tokens`` training and the official HiViS prune rule.

vLLM Eagle still shares the full prompt with the draft, so this HF loop is the
matching decode path until a vLLM patch exists.

Usage:
  python tools/hf_hivis_eval.py \\
    --draft_model output/progressive_default_tests/progressive_hivis/checkpoint-66466 \\
    --dataset lmms-lab/textvqa --num_prompts 8 --num_spec_tokens 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from angelslim.compressor.speculative.hivis import prune_hidden_and_ids
from angelslim.compressor.speculative.train.models.draft import (
    DraftModelConfig,
    create_draft_model,
)


def _load_draft_state(draft_dir: Path) -> Dict[str, torch.Tensor]:
    st: Dict[str, torch.Tensor] = {}
    safes = sorted(draft_dir.glob("*.safetensors"))
    if safes:
        from safetensors.torch import load_file

        for path in safes:
            st.update(load_file(str(path)))
        return st
    bin_path = draft_dir / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    model_st = draft_dir / "model.safetensors"
    if model_st.is_file():
        from safetensors.torch import load_file

        return load_file(str(model_st))
    raise FileNotFoundError(f"No weights under {draft_dir}")


def load_draft(draft_dir: Path, config_path: Path, device: torch.device, dtype: torch.dtype):
    cfg = DraftModelConfig.from_file(str(config_path if config_path.is_file() else draft_dir / "config.json"))
    draft = create_draft_model(cfg)
    missing, unexpected = draft.load_state_dict(_load_draft_state(draft_dir), strict=False)
    missing = [k for k in missing if not (k.startswith("t2d") or k.startswith("d2t"))]
    if missing:
        print(f"WARNING missing draft keys: {len(missing)} e.g. {missing[:5]}")
    if unexpected:
        print(f"WARNING unexpected draft keys: {len(unexpected)} e.g. {unexpected[:5]}")
    draft.to(device=device, dtype=dtype)
    draft.eval()
    if hasattr(draft, "gradient_checkpointing"):
        draft.gradient_checkpointing = False
    for p in draft.parameters():
        p.requires_grad_(False)
    if hasattr(draft, "d2t"):
        print(f"  d2t range=[{int(draft.d2t.min())}, {int(draft.d2t.max())}] n={draft.d2t.numel()}")
    return draft, cfg


def _draft_token_to_target(draft, draft_tok: int) -> int:
    if hasattr(draft, "d2t") and draft.d2t is not None:
        return int(draft.d2t[draft_tok].item())
    return draft_tok


@torch.no_grad()
def draft_propose(
    draft,
    aux_concat: torch.Tensor,
    draft_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_spec: int,
) -> List[int]:
    """Propose tokens over the compact (image-hidden) draft sequence."""
    cur_ids = draft_ids
    cur_hidden = aux_concat
    cur_mask = attention_mask
    proposed: List[int] = []
    for step in range(num_spec):
        seq_len = cur_ids.shape[1]
        position_ids = torch.arange(seq_len, device=cur_ids.device, dtype=torch.long).view(1, -1)
        out = draft(
            hidden_states=cur_hidden,
            input_ids=cur_ids,
            attention_mask=cur_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
        )
        logits = out["logits"] if isinstance(out, dict) else out[1]
        draft_tok = int(logits[:, -1, :].argmax(dim=-1).item())
        proposed.append(_draft_token_to_target(draft, draft_tok))
        if step == num_spec - 1:
            break
        nxt = torch.tensor([[proposed[-1]]], device=cur_ids.device, dtype=cur_ids.dtype)
        cur_ids = torch.cat([cur_ids, nxt], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(nxt, dtype=cur_mask.dtype)], dim=1)
        if getattr(draft, "progressive_staged", False) or getattr(draft, "hawk", False):
            if getattr(draft, "take_progressive_draft_feedback", None) is not None:
                draft.take_progressive_draft_feedback()
            injects = getattr(draft, "_aux_inject", None)
            if injects:
                padded = []
                for t in injects:
                    if t.shape[1] < cur_ids.shape[1]:
                        pad = t[:, -1:, :].expand(-1, cur_ids.shape[1] - t.shape[1], -1)
                        t = torch.cat([t, pad], dim=1)
                    padded.append(t[:, : cur_ids.shape[1]])
                draft._aux_inject = None
                cur_hidden = torch.cat(padded, dim=-1)
        else:
            last_h = out["last_hidden_state"] if isinstance(out, dict) else out[0]
            if last_h.shape[1] < cur_ids.shape[1]:
                pad = last_h[:, -1:, :].expand(-1, cur_ids.shape[1] - last_h.shape[1], -1)
                last_h = torch.cat([last_h, pad], dim=1)
            cur_hidden = last_h[:, : cur_ids.shape[1]]
    return proposed


@torch.no_grad()
def target_prefill(model, batch: Dict[str, torch.Tensor], aux_ids: Sequence[int]):
    out = model(**batch, output_hidden_states=True, use_cache=True)
    hs = out.hidden_states
    aux = torch.cat([hs[i + 1] for i in aux_ids], dim=-1)
    return out.logits, aux, out.past_key_values


@torch.no_grad()
def target_decode_tokens(model, input_ids, attention_mask, past_key_values, aux_ids):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    hs = out.hidden_states
    aux = torch.cat([hs[i + 1] for i in aux_ids], dim=-1)
    return out.logits, aux, out.past_key_values


@torch.no_grad()
def generate_hivis(
    model,
    draft,
    processor,
    messages: List[dict],
    max_new_tokens: int,
    num_spec: int,
    device: torch.device,
    aux_ids: Sequence[int],
    image_token_id: int,
    hide_visual: bool,
    vision_start_token_id: Optional[int],
    vision_end_token_id: Optional[int],
) -> Tuple[List[int], float, Dict[str, float]]:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    pixel_values = inputs.get("pixel_values")
    pixel_attention_mask = inputs.get("pixel_attention_mask")

    eos_ids = getattr(model.config, "eos_token_id", None)
    if eos_ids is None:
        text_cfg = getattr(model.config, "text_config", None)
        eos_ids = getattr(text_cfg, "eos_token_id", None) if text_cfg is not None else None
    if eos_ids is None:
        eos_ids = getattr(processor.tokenizer, "eos_token_id", None)
    if eos_ids is None:
        eos_ids = []
    elif isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    else:
        eos_ids = list(eos_ids)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    prefill = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if pixel_values is not None:
        prefill["pixel_values"] = pixel_values
    if pixel_attention_mask is not None:
        prefill["pixel_attention_mask"] = pixel_attention_mask
    logits, aux, past = target_prefill(model, prefill, aux_ids)

    draft_ids, draft_aux, draft_mask, dropped = (
        prune_hidden_and_ids(
            input_ids,
            aux,
            attention_mask,
            image_token_id=image_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            pad_token_id=int(
                getattr(getattr(model.config, "text_config", None), "pad_token_id", None)
                or getattr(model.config, "pad_token_id", 0)
                or 0
            ),
        )
        if hide_visual
        else (input_ids, aux, attention_mask, [0])
    )

    generated: List[int] = []
    cur_ids = input_ids
    cur_mask = attention_mask
    stats = {
        "drafts": 0,
        "accepted": 0,
        "pos_accept": [0] * num_spec,
        "dropped_image_tokens": int(dropped[0]),
        "full_len": int(input_ids.shape[1]),
        "draft_len": int(draft_ids.shape[1]),
    }

    def _append_and_step(token_id: int) -> None:
        nonlocal logits, aux, past, cur_ids, cur_mask, generated
        nonlocal draft_ids, draft_aux, draft_mask
        generated.append(token_id)
        tok = torch.tensor([[token_id]], device=device, dtype=cur_ids.dtype)
        cur_ids = torch.cat([cur_ids, tok], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(tok, dtype=cur_mask.dtype)], dim=1)
        logits, aux, past = target_decode_tokens(model, tok, cur_mask, past, aux_ids)
        draft_ids = torch.cat([draft_ids, tok], dim=1)
        draft_mask = torch.cat([draft_mask, torch.ones_like(tok, dtype=draft_mask.dtype)], dim=1)
        # New target aux is only the last position; keep previous compact aux and append.
        draft_aux = torch.cat([draft_aux, aux[:, -1:, :]], dim=1)

    while len(generated) < max_new_tokens:
        proposals = draft_propose(draft, draft_aux, draft_ids, draft_mask, num_spec)
        stats["drafts"] += 1
        n_acc = 0
        rejected = False
        for i, prop in enumerate(proposals):
            if len(generated) >= max_new_tokens:
                break
            target_pred = int(logits[:, -1, :].argmax(dim=-1).item())
            if target_pred != prop:
                _append_and_step(target_pred)
                rejected = True
                break
            n_acc += 1
            stats["pos_accept"][i] += 1
            _append_and_step(prop)
            if generated[-1] in eos_ids:
                rejected = True
                break
        stats["accepted"] += n_acc
        if not rejected and len(generated) < max_new_tokens:
            bonus = int(logits[:, -1, :].argmax(dim=-1).item())
            _append_and_step(bonus)
        if generated and generated[-1] in eos_ids:
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    return generated[:max_new_tokens], dt, stats


def load_prompts(dataset: str, num_prompts: int) -> List[List[dict]]:
    if os.path.exists(dataset):
        ds = load_dataset("json", data_files=dataset, split="train")
    elif dataset == "lmms-lab/textvqa":
        ds = load_dataset(dataset, split="validation")
    elif dataset == "MMMU/MMMU":
        ds = load_dataset(dataset, "History", split="test")
    elif dataset == "Lin-Chen/MMStar":
        ds = load_dataset(dataset, split="val")
    else:
        ds = load_dataset(dataset, split="test")
    ds = ds.select(range(min(num_prompts, len(ds))))
    prompts: List[List[dict]] = []
    for item in ds:
        if dataset == "lmms-lab/textvqa" or "image" in item:
            img = item.get("image") or item.get("image_1")
            q = item.get("question") or item.get("problem") or "Describe the image."
            if isinstance(q, str):
                q = q.replace("<image>", "").replace("<image 1>", "")
            content: List[Any] = []
            if img is not None:
                if not isinstance(img, Image.Image):
                    img = Image.open(img).convert("RGB")
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": q})
            prompts.append([{"role": "user", "content": content}])
        else:
            prompts.append(
                [{"role": "user", "content": item.get("problem") or item["question"]}]
            )
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target_model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument(
        "--draft_model",
        type=Path,
        default=Path("output/progressive_default_tests/progressive_hivis/checkpoint-66466"),
    )
    ap.add_argument(
        "--draft_model_config_path",
        type=Path,
        default=Path(
            "angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-hivis.json"
        ),
    )
    ap.add_argument("--dataset", default="lmms-lab/textvqa")
    ap.add_argument("--num_prompts", type=int, default=8)
    ap.add_argument("--num_spec_tokens", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--output_file", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--no_hivis",
        action="store_true",
        help="Do not prune image tokens from the draft (full-sequence ablation).",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    hide_visual = not args.no_hivis

    print(f"Loading target {args.target_model}")
    processor = AutoProcessor.from_pretrained(args.target_model, trust_remote_code=True)
    target = AutoModelForImageTextToText.from_pretrained(
        args.target_model, dtype=dtype, trust_remote_code=True
    ).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    print(f"Loading draft {args.draft_model}")
    draft, cfg = load_draft(args.draft_model, args.draft_model_config_path, device, dtype)
    aux_ids = tuple(int(x) for x in getattr(cfg, "aux_hidden_states_layer_ids", [1, 14, 26]))
    image_token_id = int(getattr(cfg, "image_token_id", getattr(target.config, "image_token_id", 49190)))
    vision_start = getattr(cfg, "vision_start_token_id", None)
    vision_end = getattr(cfg, "vision_end_token_id", None)
    print(
        f"HiViS hide_visual={hide_visual} image_token_id={image_token_id} "
        f"aux={list(aux_ids)} mode={getattr(cfg, 'eagle_aux_injection_mode', None)}"
    )

    prompts = load_prompts(args.dataset, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts from {args.dataset}")

    spec_tokens = 0
    spec_time = 0.0
    total_drafts = 0
    total_accepted = 0
    pos_accept = [0] * args.num_spec_tokens
    dropped_sum = 0
    rows = []
    for i, messages in enumerate(prompts):
        toks, dt, st = generate_hivis(
            target,
            draft,
            processor,
            messages,
            args.max_new_tokens,
            args.num_spec_tokens,
            device,
            aux_ids,
            image_token_id,
            hide_visual,
            vision_start,
            vision_end,
        )
        spec_tokens += len(toks)
        spec_time += dt
        total_drafts += st["drafts"]
        total_accepted += st["accepted"]
        dropped_sum += st["dropped_image_tokens"]
        for j in range(args.num_spec_tokens):
            pos_accept[j] += st["pos_accept"][j]
        print(
            f"  [{i}] tokens={len(toks)} time={dt:.2f}s drafts={st['drafts']} "
            f"accepted={st['accepted']} dropped_img={st['dropped_image_tokens']} "
            f"full_len={st['full_len']} draft_len={st['draft_len']}"
        )
        rows.append({"tokens": len(toks), "time": dt, **st})

    mean_accept = 1.0 + (total_accepted / total_drafts if total_drafts else 0.0)
    spec_tps = spec_tokens / max(spec_time, 1e-6)
    summary = {
        "mode": "hivis" if hide_visual else "full_sequence_ablation",
        "draft_model": str(args.draft_model),
        "num_prompts": len(prompts),
        "num_spec_tokens": args.num_spec_tokens,
        "mean_acceptance_length": mean_accept,
        "output_throughput_tok_s": spec_tps,
        "total_output_tokens": spec_tokens,
        "total_time_s": spec_time,
        "mean_dropped_image_tokens": dropped_sum / max(len(prompts), 1),
        "acceptance_rate_per_pos": [
            (pos_accept[j] / total_drafts if total_drafts else 0.0)
            for j in range(args.num_spec_tokens)
        ],
    }
    print(f"Mean acceptance length: {mean_accept:.3f}")
    print(f"output_throughput: {spec_tps:.2f} tokens/s")
    print(f"mean dropped image tokens: {summary['mean_dropped_image_tokens']:.1f}")

    out = args.output_file
    if out is None:
        tag = "hivis" if hide_visual else "no_hivis"
        out = args.draft_model / "eval_hf_hivis" / f"{tag}_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_prompt": rows}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
