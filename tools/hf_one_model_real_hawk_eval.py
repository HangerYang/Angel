#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""One-model real_hawk eval (HF): shared target W, LoRA only on draft steps.

Measures the speedup story you actually want to claim:
  VERIFY  = full SmolVLM, base weights only (no LoRA)
  DRAFT   = hawk fuse + layers[1,14,26] with LoRA on the *same* W tensors

Unlike merged 2-model vLLM eval, draft ``LoRALinear.base.weight`` is the
target Parameter object (no duplicated W, LoRA GEMMs paid every draft step).

Usage:
  python tools/hf_one_model_real_hawk_eval.py \\
    --draft_model output/smolvlm_256m_real_hawk_nccl/checkpoint-XXXX \\
    --dataset lmms-lab/textvqa --num_prompts 20 --num_spec_tokens 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from angelslim.compressor.speculative.train.models.draft import (
    DraftModelConfig,
    create_draft_model,
)
from angelslim.compressor.speculative.train.models.draft.lora_utils import (
    LoRALinear,
    apply_real_hawk_training_setup,
)

AUX_TRAIN_IDS = (0, 13, 25)
DRAFT_LAYER_IDS = (1, 14, 26)
LORA_PROJ_PATHS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _attr(root: nn.Module, path: str) -> nn.Module:
    cur: nn.Module = root
    for p in path.split("."):
        cur = getattr(cur, p)
    return cur


def _load_draft_state(draft_dir: Path) -> Dict[str, torch.Tensor]:
    st: Dict[str, torch.Tensor] = {}
    safes = sorted(draft_dir.glob("*.safetensors"))
    if safes:
        from safetensors.torch import load_file

        for p in safes:
            st.update(load_file(str(p)))
        return st
    bin_path = draft_dir / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    model_st = draft_dir / "model.safetensors"
    if model_st.is_file():
        from safetensors.torch import load_file

        return load_file(str(model_st))
    raise FileNotFoundError(f"No weights under {draft_dir}")


def share_parameter(dst_module: nn.Module, name: str, src_param: nn.Parameter) -> None:
    """Point ``dst_module``'s Parameter ``name`` at ``src_param`` (same storage)."""
    if name not in dst_module._parameters and not hasattr(dst_module, name):
        raise AttributeError(f"{dst_module} has no parameter {name}")
    dst_module._parameters[name] = src_param


def share_real_hawk_bases(
    draft: nn.Module,
    target_text: nn.Module,
    layer_ids: Sequence[int] = DRAFT_LAYER_IDS,
) -> int:
    """Share draft LoRA base W (+ layer norms, embed) with target text tower."""
    n = 0
    # Embeddings
    share_parameter(
        draft.embed_tokens, "weight", target_text.embed_tokens.weight
    )
    n += 1
    for di, tid in enumerate(layer_ids):
        t_layer = target_text.layers[tid]
        d_layer = draft.layers[di]
        share_parameter(
            d_layer.input_layernorm, "weight", t_layer.input_layernorm.weight
        )
        share_parameter(
            d_layer.post_attention_layernorm,
            "weight",
            t_layer.post_attention_layernorm.weight,
        )
        n += 2
        for path in LORA_PROJ_PATHS:
            d_lora = _attr(d_layer, path)
            t_lin = _attr(t_layer, path)
            if not isinstance(d_lora, LoRALinear):
                raise TypeError(f"expected LoRALinear at draft {path}, got {type(d_lora)}")
            if not isinstance(t_lin, nn.Linear):
                raise TypeError(f"expected Linear at target {path}, got {type(t_lin)}")
            share_parameter(d_lora.base, "weight", t_lin.weight)
            n += 1
    return n


def build_draft_from_ckpt(
    draft_dir: Path,
    config_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    train_cfg = DraftModelConfig.from_file(str(config_path))
    mode = getattr(train_cfg, "eagle_aux_injection_mode", "")
    if mode not in ("real_hawk", "layer_skip_lora"):
        print(
            f"WARNING: config mode={mode!r}; expected real_hawk/layer_skip_lora. "
            "Continuing with LoRA inject anyway."
        )
    draft = create_draft_model(train_cfg)
    apply_real_hawk_training_setup(
        draft,
        r=int(getattr(train_cfg, "lora_r", 16)),
        alpha=float(getattr(train_cfg, "lora_alpha", 32)),
        dropout=float(getattr(train_cfg, "lora_dropout", 0.0)),
        target_modules=list(
            getattr(
                train_cfg,
                "lora_target_modules",
                list(p.split(".")[-1] for p in LORA_PROJ_PATHS),
            )
        ),
    )
    state = _load_draft_state(draft_dir)
    missing, unexpected = draft.load_state_dict(state, strict=False)
    print(
        f"draft load: missing={len(missing)} unexpected={len(unexpected)} "
        f"(ok if target-tied bases get replaced by share)"
    )
    draft.to(device=device, dtype=dtype)
    draft.eval()
    return draft


def causal_mask(batch: int, q_len: int, kv_len: int, device, dtype) -> torch.Tensor:
    # [B,1,Q,KV] additive mask: 0 keep, -inf mask
    mask = torch.zeros(batch, 1, q_len, kv_len, device=device, dtype=dtype)
    if q_len > 1:
        # standard causal for full prefill-style draft calls
        tri = torch.triu(
            torch.ones(q_len, kv_len, device=device, dtype=torch.bool),
            diagonal=kv_len - q_len + 1,
        )
        mask = mask.masked_fill(tri, torch.finfo(dtype).min)
    return mask


@torch.no_grad()
def target_prefill(
    model,
    batch: Dict[str, torch.Tensor],
    aux_ids: Sequence[int] = AUX_TRAIN_IDS,
):
    out = model(**batch, output_hidden_states=True, use_cache=True)
    hs = out.hidden_states  # (embed, after_l0, ..., after_l{n-1})
    aux = torch.cat([hs[i + 1] for i in aux_ids], dim=-1)
    return out.logits, aux, out.past_key_values


@torch.no_grad()
def target_decode_tokens(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_attention_mask: Optional[torch.Tensor] = None,
    aux_ids: Sequence[int] = AUX_TRAIN_IDS,
):
    kwargs: Dict[str, Any] = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    # Only pass pixels on prefill; decode steps typically omit
    if pixel_values is not None and past_key_values is None:
        kwargs["pixel_values"] = pixel_values
        if pixel_attention_mask is not None:
            kwargs["pixel_attention_mask"] = pixel_attention_mask
    out = model(**kwargs)
    hs = out.hidden_states
    aux = torch.cat([hs[i + 1] for i in aux_ids], dim=-1)
    return out.logits, aux, out.past_key_values


def draft_propose(
    draft: nn.Module,
    aux_3h: torch.Tensor,
    context_ids: torch.Tensor,
    num_spec: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[int], int]:
    """Propose ``num_spec`` tokens with hawk fuse + LoRA (shared W).

    Recomputes the draft stack each micro-step (no draft KV) so LoRA matmuls
    are paid every time — conservative vs a cached one-model engine.
    """
    last = aux_3h[:, -1:, :]  # [B,1,3H]
    pos0 = context_ids.shape[1] - 1
    proposed: List[int] = []
    local_ids = context_ids[:, -1:].clone()

    # Seed injects from target aux (single position, expand per step)
    seed_chunks = last.split(draft.hidden_size, dim=-1)
    draft._aux_inject = seed_chunks

    for step in range(num_spec):
        bsz, seqlen = local_ids.shape
        embeds = draft.embed_input_ids(local_ids)
        # Expand current injects to seq length
        assert draft._aux_inject is not None
        draft._aux_inject = tuple(
            t[:, -1:, :].expand(bsz, seqlen, -1) for t in draft._aux_inject
        )
        attn = causal_mask(bsz, seqlen, seqlen, device, dtype)
        position_ids = torch.arange(
            pos0, pos0 + seqlen, device=device, dtype=torch.long
        ).unsqueeze(0)
        # Hawk encode ignores the passed hidden_states seed (re-fuses).
        hidden, _ = draft.encode_layers(
            inputs_embeds=embeds,
            hidden_states=draft._aux_inject[0],
            cache_hidden=draft.init_cache_hidden(),
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = draft.compute_logits(hidden)[:, -1, :]
        draft_tok = int(logits.argmax(dim=-1).item())
        target_tok = int(draft.d2t[draft_tok].item())
        proposed.append(target_tok)
        draft.take_progressive_draft_feedback()
        local_ids = torch.cat(
            [
                local_ids,
                torch.tensor([[target_tok]], device=device, dtype=local_ids.dtype),
            ],
            dim=1,
        )
    return proposed, len(proposed)


@torch.no_grad()
def generate_baseline(
    model,
    processor,
    messages: List[dict],
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[List[int], float]:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    dt = time.perf_counter() - t0
    new_tokens = out[0, prompt_len:].tolist()
    return new_tokens, dt


@torch.no_grad()
def generate_one_model_spec(
    model,
    draft: nn.Module,
    processor,
    messages: List[dict],
    max_new_tokens: int,
    num_spec: int,
    device: torch.device,
    dtype: torch.dtype,
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
    pixel_values = inputs.get("pixel_values")
    pixel_attention_mask = inputs.get("pixel_attention_mask")

    eos_ids = model.config.eos_token_id
    if eos_ids is None:
        eos_ids = []
    elif isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    t0 = time.perf_counter()

    logits, aux, past = target_prefill(
        model,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **({"pixel_values": pixel_values} if pixel_values is not None else {}),
            **(
                {"pixel_attention_mask": pixel_attention_mask}
                if pixel_attention_mask is not None
                else {}
            ),
        },
    )
    generated: List[int] = []
    cur_ids = input_ids
    cur_mask = attention_mask
    stats = {"drafts": 0, "accepted": 0, "pos_accept": [0] * num_spec}

    def _append_and_step(token_id: int) -> None:
        nonlocal logits, aux, past, cur_ids, cur_mask, generated
        generated.append(token_id)
        tok = torch.tensor([[token_id]], device=device, dtype=cur_ids.dtype)
        cur_ids = torch.cat([cur_ids, tok], dim=1)
        if cur_mask is not None:
            cur_mask = torch.cat(
                [cur_mask, torch.ones_like(tok, dtype=cur_mask.dtype)], dim=1
            )
        logits, aux, past = target_decode_tokens(
            model, tok, cur_mask, past, aux_ids=AUX_TRAIN_IDS
        )

    while len(generated) < max_new_tokens:
        # Draft K proposals from current aux (LoRA on, shared W)
        proposals, _ = draft_propose(
            draft, aux, cur_ids, num_spec, device, dtype
        )
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
            # All K accepted → bonus token from target
            bonus = int(logits[:, -1, :].argmax(dim=-1).item())
            _append_and_step(bonus)

        if generated and generated[-1] in eos_ids:
            break

    torch.cuda.synchronize(device) if device.type == "cuda" else None
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--draft_model", type=Path, required=True)
    ap.add_argument(
        "--draft_model_config_path",
        type=Path,
        default=Path(
            "angelslim/compressor/speculative/train/configs/smolvlm-256m-real-hawk.json"
        ),
    )
    ap.add_argument("--dataset", default="lmms-lab/textvqa")
    ap.add_argument("--num_prompts", type=int, default=20)
    ap.add_argument("--num_spec_tokens", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--output_file", type=Path, default=None)
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    print("Loading target (one model)...")
    processor = AutoProcessor.from_pretrained(args.target_model, trust_remote_code=True)
    target = AutoModelForImageTextToText.from_pretrained(
        args.target_model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    target.eval()
    text = target.model.text_model

    print("Loading real_hawk draft adapters...")
    draft = build_draft_from_ckpt(
        args.draft_model, args.draft_model_config_path, device, dtype
    )
    n_shared = share_real_hawk_bases(draft, text, DRAFT_LAYER_IDS)
    print(
        f"ONE-MODEL: shared {n_shared} Parameter objects "
        f"(embed + norms + LoRA bases ↔ target layers {list(DRAFT_LAYER_IDS)})"
    )
    # Sanity: base storage identity
    d0 = draft.layers[0].self_attn.q_proj
    t0 = text.layers[DRAFT_LAYER_IDS[0]].self_attn.q_proj
    assert isinstance(d0, LoRALinear)
    assert d0.base.weight.data_ptr() == t0.weight.data_ptr(), (
        "base W not shared — refusing to report fake one-model speedup"
    )
    print("  verified: draft.layers[0].q_proj.base.weight is target.layers[1].q_proj.weight")

    prompts = load_prompts(args.dataset, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts from {args.dataset}")

    base_tokens = 0
    base_time = 0.0
    if not args.skip_baseline:
        print("=== Baseline (target AR, no draft) ===")
        for i, messages in enumerate(prompts):
            toks, dt = generate_baseline(
                target, processor, messages, args.max_new_tokens, device
            )
            base_tokens += len(toks)
            base_time += dt
            print(f"  [{i}] tokens={len(toks)} time={dt:.2f}s")
        print(
            f"baseline: {base_tokens} toks in {base_time:.2f}s → "
            f"{base_tokens / max(base_time, 1e-6):.2f} tok/s"
        )

    print("=== One-model real_hawk (shared W + LoRA on draft) ===")
    spec_tokens = 0
    spec_time = 0.0
    total_drafts = 0
    total_accepted = 0
    pos_accept = [0] * args.num_spec_tokens
    rows = []
    for i, messages in enumerate(prompts):
        toks, dt, st = generate_one_model_spec(
            target,
            draft,
            processor,
            messages,
            args.max_new_tokens,
            args.num_spec_tokens,
            device,
            dtype,
        )
        spec_tokens += len(toks)
        spec_time += dt
        total_drafts += st["drafts"]
        total_accepted += st["accepted"]
        for j in range(args.num_spec_tokens):
            pos_accept[j] += st["pos_accept"][j]
        print(
            f"  [{i}] tokens={len(toks)} time={dt:.2f}s "
            f"drafts={st['drafts']} accepted={st['accepted']}"
        )
        rows.append({"tokens": len(toks), "time": dt, **st})

    mean_accept = 1.0 + (
        total_accepted / total_drafts if total_drafts > 0 else 0.0
    )
    spec_tps = spec_tokens / max(spec_time, 1e-6)
    summary = {
        "mode": "one_model_real_hawk",
        "draft_model": str(args.draft_model),
        "num_prompts": len(prompts),
        "num_spec_tokens": args.num_spec_tokens,
        "mean_acceptance_length": mean_accept,
        "output_throughput_tok_s": spec_tps,
        "total_output_tokens": spec_tokens,
        "total_time_s": spec_time,
        "acceptance_rate_per_pos": [
            (pos_accept[j] / total_drafts if total_drafts else 0.0)
            for j in range(args.num_spec_tokens)
        ],
    }
    if not args.skip_baseline and base_time > 0:
        base_tps = base_tokens / base_time
        summary["baseline_throughput_tok_s"] = base_tps
        summary["speedup_vs_baseline"] = spec_tps / base_tps
        print(
            f"SPEEDUP (one-model): {summary['speedup_vs_baseline']:.3f}x "
            f"({spec_tps:.2f} / {base_tps:.2f} tok/s)"
        )
    print(f"Mean acceptance length: {mean_accept:.3f}")
    print(f"output_throughput: {spec_tps:.2f} tokens/s")
    print(
        "acceptance rates: "
        + str(
            {
                f"acceptance_rate_pos_{j}": round(summary["acceptance_rate_per_pos"][j], 4)
                for j in range(args.num_spec_tokens)
            }
        )
    )

    out = args.output_file
    if out is None:
        out = args.draft_model / "eval_one_model" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_prompt": rows}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
