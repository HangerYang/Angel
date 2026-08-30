#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""HF (pure PyTorch, no vLLM) TREE-based speculative eval, HiViS-style.

Same "hide visual tokens from the drafter" idea as tools/hf_hivis_eval.py
(which is chain/linear-only), extended to real tree-based speculative
decoding with actual target verification (accept/reject), not the
no-verification timing-only tree in tools/draft_bench.py.

Why this needed a new script instead of reusing HiViS's own tree code
(hivis/model/utils_hivis.py's initialize_tree/tree_decoding/
update_inference_inputs) directly: that code is tightly coupled to HiViS's
own EaModel wrapper (model.ea_layer.topK_genrate(..., head=...), a custom
static-buffer KVCache, and a target model class with a hand-patched
_update_causal_mask that special-cases self.tree_mask). Our checkpoint is
angelslim's own Eagle3LlamaForCausalLM (create_draft_model factory;
banded_mix_fc injection), whose topK_genrate has its own lm_head baked in
and a different signature (no `head` kwarg). And SmolVLM's text backbone
(model.model.text_model) turns out to be a *stock*, unmodified
transformers.models.llama.modeling_llama.LlamaModel -- confirmed empirically
-- and stock HF LlamaModel's _update_causal_mask already passes a 4D
attention_mask through unchanged (verified against the "Copied from
transformers...LlamaModel._update_causal_mask" comment in HiViS's own
modeling_qwen2_5_kv.py), so no monkeypatching is needed: we can build the
tree-shaped 4D mask ourselves and call the stock text_model directly.

KV-cache accept/compact: after each tree round we keep only the winning
path's positions. transformers' DynamicCache (this env: 5.16.1) stores each
layer as a DynamicLayer with plain `.keys`/`.values` tensors
[batch, heads, seq, head_dim] -- we gather+recat those directly rather than
going through HiViS's own legacy KVCache class (which expects a different,
older cache protocol that stock LlamaModel does not use here).

Usage:
  PYTHONPATH=/home/hyang/Angel/HiViS python3 tools/hf_hivis_eval_tree.py \
    --draft_model dataset/angelslim-smolvlm-eagle3-artifacts/weight/branch-distill-top1-w01/checkpoint-66466 \
    --draft_model_config_path dataset/angelslim-smolvlm-eagle3-artifacts/weight/branch-distill-top1-w01/checkpoint-66466/config.json \
    --dataset lmms-lab/textvqa --num_prompts 8 --total_token 60 --depth 5 --top_k 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_hivis_eval import (  # noqa: E402
    _draft_token_to_target,
    load_draft,
    load_prompts,
)

from angelslim.compressor.speculative.hivis import prune_hidden_and_ids  # noqa: E402

try:
    from hivis.model.utils_hivis import evaluate_posterior
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Could not import hivis.model.utils_hivis.evaluate_posterior. "
        "Set PYTHONPATH to include /home/hyang/Angel/HiViS, e.g.:\n"
        "  PYTHONPATH=/home/hyang/Angel/HiViS python3 tools/hf_hivis_eval_tree.py ..."
    ) from e


NEG_INF = torch.finfo(torch.bfloat16).min


def build_tree_attention_mask(
    tree_mask: torch.Tensor, prev_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """[1,1,tree_len, prev_len+tree_len] additive mask: full attention to the
    confirmed past (columns [:prev_len]), tree-structured attention within
    the speculative block (columns [prev_len:])."""
    tree_len = tree_mask.shape[-1]
    total_len = prev_len + tree_len
    mask = torch.zeros((1, 1, tree_len, total_len), dtype=dtype, device=device)
    tree_block = (tree_mask.to(device=device) == 0).to(dtype) * NEG_INF
    mask[:, :, :, prev_len:] = tree_block
    return mask


@torch.no_grad()
def target_prefill_tree(model, batch: Dict[str, torch.Tensor], aux_ids: Sequence[int], cache: DynamicCache):
    out = model(**batch, output_hidden_states=True, use_cache=True, past_key_values=cache)
    hs = out.hidden_states
    aux = torch.cat([hs[i + 1] for i in aux_ids], dim=-1)
    return out.logits, aux, out.past_key_values


@torch.no_grad()
def tree_verify_step(
    text_model,
    lm_head,
    draft_tokens: torch.Tensor,
    tree_mask: torch.Tensor,
    tree_position_ids: torch.Tensor,
    cache: DynamicCache,
    prev_len: int,
    aux_ids: Sequence[int],
):
    """One batched target forward over the whole speculative tree."""
    inputs_embeds = text_model.get_input_embeddings()(draft_tokens)
    attn_mask = build_tree_attention_mask(
        tree_mask, prev_len, inputs_embeds.dtype, inputs_embeds.device
    )
    position_ids = (tree_position_ids + prev_len).unsqueeze(0).to(inputs_embeds.device)
    out = text_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    tree_logits = lm_head(out.last_hidden_state)
    hs = out.hidden_states
    aux = torch.cat([hs[i + 1] for i in aux_ids], dim=-1)
    return tree_logits, aux, out.past_key_values


def compact_cache(cache: DynamicCache, prev_len: int, select_indices: torch.Tensor) -> None:
    """Keep [:prev_len] (confirmed) + gathered accepted tree positions; drop
    the rest of this round's speculative (rejected) KV entries."""
    for layer in cache.layers:
        acc_k = layer.keys[..., select_indices, :]
        acc_v = layer.values[..., select_indices, :]
        layer.keys = torch.cat([layer.keys[..., :prev_len, :], acc_k], dim=-2)
        layer.values = torch.cat([layer.values[..., :prev_len, :], acc_v], dim=-2)


@torch.no_grad()
def generate_hivis_tree(
    model,
    draft,
    processor,
    messages: List[dict],
    max_new_tokens: int,
    total_token: int,
    depth: int,
    top_k: int,
    device: torch.device,
    aux_ids: Sequence[int],
    image_token_id: int,
    hide_visual: bool,
    vision_start_token_id: Any,
    vision_end_token_id: Any,
) -> Tuple[List[int], float, Dict[str, float]]:
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
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
    eos_ids = [] if eos_ids is None else ([eos_ids] if isinstance(eos_ids, int) else list(eos_ids))

    text_model = model.model.text_model
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    prefill = {"input_ids": input_ids, "attention_mask": attention_mask}
    if pixel_values is not None:
        prefill["pixel_values"] = pixel_values
    if pixel_attention_mask is not None:
        prefill["pixel_attention_mask"] = pixel_attention_mask
    cache = DynamicCache()
    logits, aux, cache = target_prefill_tree(model, prefill, aux_ids, cache)

    sample_token = torch.argmax(logits[:, -1, :], dim=-1)[None, None]

    draft_ids, draft_aux, _draft_mask, dropped = (
        prune_hidden_and_ids(
            input_ids, aux, attention_mask,
            image_token_id=image_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            pad_token_id=int(
                getattr(getattr(model.config, "text_config", None), "pad_token_id", None)
                or getattr(model.config, "pad_token_id", 0) or 0
            ),
        )
        if hide_visual
        else (input_ids, aux, attention_mask, [0])
    )

    draft_ids_with_token = torch.cat((draft_ids, sample_token), dim=1)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = draft.topK_genrate(
        hidden_states=draft_aux, input_ids=draft_ids_with_token, inputs_embeds=None, logits_processor=None
    )
    draft_tokens = draft_tokens.to(device)
    tree_mask = tree_mask.to(device)
    tree_position_ids = tree_position_ids.to(device)
    retrieve_indices = retrieve_indices.to(device)

    padding = torch.zeros((1, 1), dtype=draft_tokens.dtype, device=device) - 1

    generated: List[int] = []
    prev_len = input_ids.shape[1]
    stats = {"rounds": 0, "accepted": 0, "dropped_image_tokens": int(dropped[0]), "accept_lengths": []}

    while len(generated) < max_new_tokens:
        stats["rounds"] += 1
        tree_logits, aux_new, cache = tree_verify_step(
            text_model, lm_head, draft_tokens, tree_mask, tree_position_ids, cache, prev_len, aux_ids
        )
        logits_paths = tree_logits[0, retrieve_indices]
        draft_tokens_padded = torch.cat((draft_tokens, padding), dim=1)
        candidates = draft_tokens_padded[0, retrieve_indices]

        best_candidate, accept_length, _sample_p = evaluate_posterior(logits_paths, candidates, None)
        accept_length = int(accept_length)
        best_candidate = int(best_candidate)
        stats["accepted"] += accept_length
        stats["accept_lengths"].append(accept_length + 1)

        accepted_tokens = candidates[best_candidate, : accept_length + 1].tolist()
        accepted_tokens = [_draft_token_to_target(draft, t) if t >= 0 else t for t in accepted_tokens]
        take = min(len(accepted_tokens), max_new_tokens - len(generated))
        generated.extend(accepted_tokens[:take])

        select_indices = retrieve_indices[best_candidate, : accept_length + 1] + prev_len
        compact_cache(cache, prev_len, select_indices)
        new_len = prev_len + accept_length + 1
        prev_len = new_len

        aux_paths = aux_new[:, retrieve_indices]
        accept_hidden = aux_paths[:, best_candidate, : accept_length + 1]

        if any(t in eos_ids for t in generated[len(generated) - take:]) or len(generated) >= max_new_tokens:
            break

        next_token = torch.tensor([[generated[-1]]], device=device, dtype=input_ids.dtype)
        draft_ids = torch.cat(
            (draft_ids, torch.tensor(accepted_tokens[:take], device=device, dtype=input_ids.dtype)[None]), dim=1
        )
        draft_ids_with_token = torch.cat((draft_ids, next_token), dim=1)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = draft.topK_genrate(
            hidden_states=accept_hidden, input_ids=draft_ids_with_token, inputs_embeds=None, logits_processor=None
        )
        draft_tokens = draft_tokens.to(device)
        tree_mask = tree_mask.to(device)
        tree_position_ids = tree_position_ids.to(device)
        retrieve_indices = retrieve_indices.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    mean_accept = sum(stats["accept_lengths"]) / max(len(stats["accept_lengths"]), 1)
    stats["mean_accept_length_this_prompt"] = mean_accept
    return generated[:max_new_tokens], dt, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target_model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--draft_model", type=Path, required=True)
    ap.add_argument("--draft_model_config_path", type=Path, required=True)
    ap.add_argument("--dataset", default="lmms-lab/textvqa")
    ap.add_argument("--num_prompts", type=int, default=8)
    ap.add_argument("--total_token", type=int, default=60)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--output_file", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_hivis", action="store_true")
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
    print(f"  target text backbone: {type(target.model.text_model)}")

    print(f"Loading draft {args.draft_model}")
    draft, cfg = load_draft(args.draft_model, args.draft_model_config_path, device, dtype)
    draft.total_tokens = args.total_token - 1
    draft.depth = args.depth
    draft.top_k = args.top_k
    draft.init_tree()

    aux_ids = tuple(int(x) for x in getattr(cfg, "aux_hidden_states_layer_ids", [1, 14, 26]))
    image_token_id = int(getattr(cfg, "image_token_id", getattr(target.config, "image_token_id", 49190)))
    vision_start = getattr(cfg, "vision_start_token_id", None)
    vision_end = getattr(cfg, "vision_end_token_id", None)
    print(
        f"HiViS-tree hide_visual={hide_visual} image_token_id={image_token_id} aux={list(aux_ids)} "
        f"total_token={args.total_token} depth={args.depth} top_k={args.top_k} "
        f"mode={getattr(cfg, 'eagle_aux_injection_mode', None)}"
    )

    prompts = load_prompts(args.dataset, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts from {args.dataset}")

    total_tokens_gen = 0
    total_time = 0.0
    total_rounds = 0
    total_accepted = 0
    dropped_sum = 0
    rows = []
    for i, messages in enumerate(prompts):
        toks, dt, st = generate_hivis_tree(
            target, draft, processor, messages, args.max_new_tokens,
            args.total_token, args.depth, args.top_k, device, aux_ids,
            image_token_id, hide_visual, vision_start, vision_end,
        )
        total_tokens_gen += len(toks)
        total_time += dt
        total_rounds += st["rounds"]
        total_accepted += st["accepted"]
        dropped_sum += st["dropped_image_tokens"]
        print(
            f"  [{i}] tokens={len(toks)} time={dt:.2f}s rounds={st['rounds']} "
            f"mean_accept_len={st['mean_accept_length_this_prompt']:.2f} "
            f"dropped_img={st['dropped_image_tokens']}"
        )
        rows.append({"tokens": len(toks), "time": dt, **st})

    mean_accept = 1.0 + (total_accepted / total_rounds if total_rounds else 0.0)
    tps = total_tokens_gen / max(total_time, 1e-6)
    summary = {
        "mode": "hivis_tree" if hide_visual else "full_sequence_ablation_tree",
        "draft_model": str(args.draft_model),
        "num_prompts": len(prompts),
        "total_token": args.total_token,
        "depth": args.depth,
        "top_k": args.top_k,
        "mean_acceptance_length": mean_accept,
        "output_throughput_tok_s": tps,
        "total_output_tokens": total_tokens_gen,
        "total_time_s": total_time,
        "mean_dropped_image_tokens": dropped_sum / max(len(prompts), 1),
    }
    print(f"Mean acceptance length: {mean_accept:.3f}")
    print(f"output_throughput: {tps:.2f} tokens/s")

    out = args.output_file
    if out is None:
        tag = "hivis_tree" if hide_visual else "no_hivis_tree"
        out = args.draft_model / "eval_hf_hivis_tree" / f"{tag}_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_prompt": rows}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
