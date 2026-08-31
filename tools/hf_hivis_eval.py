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


def load_draft(draft_dir: Path, config_path: Path, device: torch.device, dtype: torch.dtype,
                target_model_name_or_path: Optional[str] = None):
    cfg = DraftModelConfig.from_file(str(config_path if config_path.is_file() else draft_dir / "config.json"))
    draft = create_draft_model(cfg)
    missing, unexpected = draft.load_state_dict(_load_draft_state(draft_dir), strict=False)
    missing = [k for k in missing if not (k.startswith("t2d") or k.startswith("d2t"))]
    if missing:
        print(f"WARNING missing draft keys: {len(missing)} e.g. {missing[:5]}")
    if "embed_tokens.weight" in missing:
        # Some published checkpoints (e.g. AngelSlim/Qwen3-VL-4B-Instruct_eagle3)
        # omit embed_tokens.weight deliberately -- it is meant to be copied from
        # the TARGET model at load time, not trained. Without this the draft's
        # token embeddings stay randomly initialised and every prediction is
        # noise (mean_acceptance_length pins at 1.0 / exactly-0% acceptance
        # regardless of aux layers or positions -- confirmed empirically).
        from angelslim.compressor.speculative.train.models.model_utils import MODEL_TYPE_PARAM_MAP

        target_type = getattr(cfg, "target_model_type", None)
        entry = MODEL_TYPE_PARAM_MAP.get(target_type)
        if entry is None or target_model_name_or_path is None:
            raise RuntimeError(
                f"draft checkpoint has no embed_tokens.weight and cannot source one: "
                f"target_model_type={target_type!r} in MODEL_TYPE_PARAM_MAP={entry is not None}, "
                f"target_model_name_or_path={target_model_name_or_path!r}"
            )
        _, embed_weight_key, _ = entry
        print(f"Loading draft embed_tokens from {target_model_name_or_path}:{embed_weight_key}")
        draft.load_embed_weights(target_model_name_or_path, embed_weight_key)
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


def mrope_prefix_positions(model, input_ids, attention_mask, mm_kwargs: Dict[str, Any]):
    """Real (T,H,W) mrope position ids for a Qwen-VL-family target's full
    prompt, via the model's own get_rope_index -- None for non-mrope targets
    (image_grid_thw absent) so callers fall back to plain arange positions.

    Only meaningful when the draft still sees the image span (the "with
    images" arm): once image tokens are pruned from the draft's sequence the
    remaining rows are plain text, for which arange positions already match
    Qwen's own mrope convention (T=H=W=running index), so no caller needs
    this in the pruned arm.
    """
    image_grid_thw = mm_kwargs.get("image_grid_thw")
    if image_grid_thw is None:
        return None
    inner = getattr(model, "model", model)
    fn = getattr(inner, "get_rope_index", None)
    if fn is None:
        return None
    kw = dict(
        image_grid_thw=image_grid_thw,
        video_grid_thw=mm_kwargs.get("video_grid_thw"),
        attention_mask=attention_mask,
    )
    try:
        # transformers>=5: mm_token_type_ids is a required positional arg.
        pos, delta = fn(input_ids, mm_kwargs["mm_token_type_ids"], **kw)
    except (TypeError, KeyError):
        pos, delta = fn(input_ids, **kw)
    return pos, delta


def mrope_positions_for_len(prefix_pos: torch.Tensor, rope_delta: torch.Tensor, seq_len: int,
                             device: torch.device) -> torch.Tensor:
    """Extend a prefix's real [3,1,prefix_len] mrope positions to a longer
    (image-inclusive) draft sequence: prompt positions are copied verbatim,
    every appended generation token continues the scalar convention Qwen's
    own get_rope_index uses past the vision span (index + mrope_position_delta,
    same value broadcast across T/H/W).
    """
    prefix_len = prefix_pos.shape[-1]
    if seq_len <= prefix_len:
        return prefix_pos[:, :, :seq_len].to(device)
    extra_n = seq_len - prefix_len
    idx = torch.arange(prefix_len, prefix_len + extra_n, device=device, dtype=prefix_pos.dtype).view(1, 1, -1)
    delta = rope_delta.reshape(1, 1, 1).to(device=device, dtype=prefix_pos.dtype)
    extra = (idx + delta).expand(3, prefix_pos.shape[1], -1)
    return torch.cat([prefix_pos.to(device), extra], dim=-1)


@torch.no_grad()
def draft_propose(
    draft,
    aux_concat: torch.Tensor,
    draft_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_spec: int,
    mrope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> List[int]:
    """Propose tokens over the compact (image-hidden) draft sequence."""
    cur_ids = draft_ids
    cur_hidden = aux_concat
    cur_mask = attention_mask
    proposed: List[int] = []
    for step in range(num_spec):
        seq_len = cur_ids.shape[1]
        if mrope is not None:
            position_ids = mrope_positions_for_len(mrope[0], mrope[1], seq_len, cur_ids.device)
        else:
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
def target_prefill(model, batch: Dict[str, torch.Tensor], aux_ids: Sequence[int],
                    position_ids: Optional[torch.Tensor] = None):
    out = model(**batch, position_ids=position_ids, output_hidden_states=True, use_cache=True)
    hs = out.hidden_states
    aux = torch.cat([hs[i + 1] for i in aux_ids], dim=-1)
    return out.logits, aux, out.past_key_values


@torch.no_grad()
def target_decode_tokens(model, input_ids, attention_mask, past_key_values, aux_ids,
                          position_ids: Optional[torch.Tensor] = None):
    # A Qwen-VL-family target's own position_ids=None auto-derivation is only
    # exercised (and only reliable) inside its own generate() loop; called
    # here directly with a manually-grown past_key_values, it can mis-derive
    # cache_position and corrupt the attention output shape. Passing the real
    # mrope positions explicitly (mrope_positions_for_len) sidesteps this
    # entirely -- confirmed via a minimal reproduction against the target
    # alone. Non-mrope targets (position_ids stays None) are unaffected.
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
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
    mm_kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}

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

    # Real mrope (T/H/W) positions for the target's full, image-inclusive
    # prompt -- None for non-mrope targets (no image_grid_thw). The TARGET
    # always sees every image row regardless of hide_visual, so this is
    # computed unconditionally and used for every target call (both prefill
    # and decode -- see target_decode_tokens's docstring for why leaving
    # position_ids=None breaks manual incremental decode for these models).
    target_mrope = mrope_prefix_positions(model, input_ids, attention_mask, mm_kwargs)
    target_pos = target_mrope[0] if target_mrope is not None else None

    prefill = {"input_ids": input_ids, "attention_mask": attention_mask, **mm_kwargs}
    logits, aux, past = target_prefill(model, prefill, aux_ids, position_ids=target_pos)

    # The DRAFT only gets real mrope positions while it still sees the image
    # span (hide_visual=False); once images are pruned the remaining rows are
    # plain text, for which arange positions already match Qwen's own mrope
    # convention (T=H=W=running index) -- see mrope_prefix_positions.
    draft_mrope = None if hide_visual else target_mrope

    # EAGLE3's defining shift: the draft's input token at position i is the
    # REAL token at i+1 (`shifted_ids`), paired with aux[i] -- the target
    # hidden state that PREDICTED it. Without this the draft is trained/run
    # on a token/hidden-state pairing one position off from what it was ever
    # trained on, and its output is unrelated noise (confirmed empirically:
    # this was silently missing here and produced exactly-0% acceptance on
    # both the SmolVLM banded-mix-fc-3.1 and Qwen3-VL-4B checkpoints, despite
    # both loading and running without error). `first` is the target's own
    # greedy prediction for the token right after the prompt -- the "free"
    # first token EAGLE always gets from the prefill forward, matching
    # draft_bench.py's eagle3_target_prefill/hivis_target_inputs.
    first_tok = int(logits[:, -1, :].argmax(dim=-1).item())
    shifted_ids = torch.cat(
        [input_ids[:, 1:], torch.full((1, 1), first_tok, device=device, dtype=input_ids.dtype)], dim=1
    )

    draft_ids, draft_aux, draft_mask, dropped = (
        prune_hidden_and_ids(
            shifted_ids,
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
        else (shifted_ids, aux, attention_mask, [0])
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
        # Same shift as the prompt's shifted_ids: this token pairs with the aux
        # from the PREVIOUS target forward (the hidden state that predicted
        # it), not the aux this token's own forward is about to produce.
        prev_aux_last = aux[:, -1:, :]
        draft_ids = torch.cat([draft_ids, tok], dim=1)
        draft_mask = torch.cat([draft_mask, torch.ones_like(tok, dtype=draft_mask.dtype)], dim=1)
        draft_aux = torch.cat([draft_aux, prev_aux_last], dim=1)
        cur_ids = torch.cat([cur_ids, tok], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(tok, dtype=cur_mask.dtype)], dim=1)
        step_pos = (
            mrope_positions_for_len(target_mrope[0], target_mrope[1], cur_ids.shape[1], device)[:, :, -1:]
            if target_mrope is not None
            else None
        )
        logits, aux, past = target_decode_tokens(model, tok, cur_mask, past, aux_ids, position_ids=step_pos)

    while len(generated) < max_new_tokens:
        proposals = draft_propose(draft, draft_aux, draft_ids, draft_mask, num_spec, mrope=draft_mrope)
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


# Lengthens raw VQA-style questions so there is enough decode length for a
# speculative-decoding comparison to show anything -- matches the
# answer_then_describe variant in tools/vllm_offline_eagle3_vlm_batch.py.
ANSWER_THEN_DESCRIBE = (
    "Answer this question: {q} Then describe the image in detail to justify your answer."
)


def load_prompts(dataset: str, num_prompts: int, prompt_style: str = "raw") -> List[List[dict]]:
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
                if prompt_style == "answer_then_describe":
                    q = ANSWER_THEN_DESCRIBE.format(q=q)
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
    ap.add_argument(
        "--prompt_style",
        choices=("raw", "answer_then_describe"),
        default="raw",
        help="answer_then_describe lengthens VQA-style questions so there are enough "
        "decode rounds for the with/without-images comparison to show anything.",
    )
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
    draft, cfg = load_draft(args.draft_model, args.draft_model_config_path, device, dtype,
                             target_model_name_or_path=args.target_model)
    aux_ids = tuple(int(x) for x in getattr(cfg, "aux_hidden_states_layer_ids", [1, 14, 26]))
    image_token_id = int(getattr(cfg, "image_token_id", getattr(target.config, "image_token_id", 49190)))
    vision_start = getattr(cfg, "vision_start_token_id", None)
    vision_end = getattr(cfg, "vision_end_token_id", None)
    print(
        f"HiViS hide_visual={hide_visual} image_token_id={image_token_id} "
        f"aux={list(aux_ids)} mode={getattr(cfg, 'eagle_aux_injection_mode', None)}"
    )

    prompts = load_prompts(args.dataset, args.num_prompts, args.prompt_style)
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
