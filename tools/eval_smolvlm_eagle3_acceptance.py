#!/usr/bin/env python3
"""Greedy accepted-token smoke eval for SmolVLM EAGLE-3 drafts.

This computes the temperature-0 rejection-sampling acceptance prefix length on
real SmolVLM target logits along the dataset target trajectory. It is intended
for quick baseline-vs-oracle draft comparisons, including oracle gist
conditioning produced by the online SmolVLM dataset builder (both
``gist_mode='remaining'`` and ``gist_mode='whole'``).

The gist mode is read from the draft config so the oracle supplied at eval
time matches the one the draft was trained on; ``--gist_mode`` overrides it
only for deliberate mismatch ablations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader

from angelslim.compressor.speculative.train.data import DatasetManager
from angelslim.compressor.speculative.train.models import (
    DraftModelConfig,
    create_draft_model,
    create_target_model,
)
from angelslim.compressor.speculative.utils import padding
from angelslim.compressor.vistoken.splice import compress_image_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target_model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    p.add_argument("--draft_model", required=True, help="Draft checkpoint directory")
    p.add_argument("--draft_model_config_path", required=True)
    p.add_argument("--data_path", default="dataset/smolvlm_256m_target_gen/eval.jsonl")
    p.add_argument("--output_file", default=None)
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--num_spec_tokens", type=int, default=7)
    p.add_argument("--model_max_length", type=int, default=4096)
    p.add_argument("--num_proc", type=int, default=1)
    p.add_argument("--load_from_cache_file", action="store_true")
    p.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--chat_template_type", default="smolvlm")
    p.add_argument("--target_backend", default="hf")
    p.add_argument(
        "--gist_mode",
        default=None,
        choices=["remaining", "whole"],
        help=(
            "Override the oracle gist mode. Default: take it from the draft "
            "config, which is what keeps eval consistent with training. Set "
            "explicitly only to measure a deliberate train/eval mismatch."
        ),
    )
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_draft(args: argparse.Namespace, cfg: DraftModelConfig) -> torch.nn.Module:
    draft = create_draft_model(cfg)
    state_path = Path(args.draft_model) / "model.safetensors"
    if state_path.is_file():
        from safetensors.torch import load_file

        state = load_file(str(state_path))
    else:
        bin_path = Path(args.draft_model) / "pytorch_model.bin"
        if not bin_path.is_file():
            raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {args.draft_model}")
        state = torch.load(bin_path, map_location="cpu")
    missing, unexpected = draft.load_state_dict(state, strict=False)
    if unexpected:
        print(f"WARNING unexpected draft keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    if missing:
        print(f"WARNING missing draft keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    draft.eval()
    return draft


def make_dataset(args: argparse.Namespace, target_model: Any, cfg: DraftModelConfig):
    data_args = SimpleNamespace(
        training_mode="online",
        modal_type="VLM",
        train_data_path=None,
        eval_data_path=args.data_path,
        num_proc=args.num_proc,
        sample_num=args.num_samples,
        shuffle_seed=42,
        load_from_cache_file=args.load_from_cache_file,
        target_model_name_or_path=args.target_model,
        output_dir=str(Path(args.output_file).parent) if args.output_file else "output/acceptance_eval",
        gist_conditioning=bool(getattr(cfg, "gist_conditioning", False)),
        gist_encoder_model_name_or_path=getattr(cfg, "gist_encoder_model_name_or_path", "Qwen/Qwen3-Embedding-0.6B"),
        gist_refresh_every=getattr(cfg, "gist_refresh_every", 4),
        gist_encoder_device=getattr(cfg, "gist_encoder_device", "cuda:0"),
        gist_batch_size=getattr(cfg, "gist_batch_size", 32),
        gist_embedding_dim=getattr(cfg, "gist_embedding_dim", 0),
        gist_mode=args.gist_mode or getattr(cfg, "gist_mode", "remaining"),
    )
    manager = DatasetManager(
        data_args,
        tokenizer=target_model.tokenizer,
        model_max_length=args.model_max_length,
        chat_template_type=args.chat_template_type,
        target_model_type=getattr(cfg, "target_model_type", "smolvlm"),
    )
    _, eval_ds, collator = manager.create_online_datasets()
    if eval_ds is None:
        raise RuntimeError("Failed to build eval dataset")
    return eval_ds, collator


def prepare_batch_for_draft(
    batch: dict[str, torch.Tensor],
    target_model: Any,
    cfg: DraftModelConfig,
    draft: torch.nn.Module | None = None,
):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    loss_mask = batch["loss_mask"]
    gist_embeddings = batch.get("gist_embeddings")
    kwargs = {
        k: v
        for k, v in batch.items()
        if k not in {"input_ids", "attention_mask", "loss_mask", "gist_embeddings"}
    }
    hidden_states, target_logits, _, position_ids = target_model.get_hidden_states_and_logits(
        input_ids=input_ids,
        attention_mask=attention_mask,
        aux_hidden_states_layer_ids=getattr(cfg, "aux_hidden_states_layer_ids", None),
        **kwargs,
    )
    # Row compression, exactly where training does it: after the target
    # forward, before the left shift, while every tensor is still aligned 1:1
    # on absolute positions.
    compressor = getattr(draft, "vistoken", None) if draft is not None else None
    if compressor is not None:
        spliced = compress_image_rows(
            compressor,
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target_logits=target_logits,
            position_ids=position_ids,
            image_token_id=draft.vistoken_image_token_id,
        )
        hidden_states = spliced["hidden_states"]
        input_ids = spliced["input_ids"]
        target_logits = spliced["target_logits"]
        loss_mask = spliced["loss_mask"]
        attention_mask = spliced["attention_mask"]
        position_ids = spliced["position_ids"]
        if gist_embeddings is not None:
            raise NotImplementedError(
                "gist conditioning and row compression both rewrite the draft "
                "sequence; they have not been reconciled"
            )

    result = {
        "input_ids": padding(input_ids, left=False),
        "target_logits": padding(target_logits, left=False).to(input_ids.device),
        "loss_mask": loss_mask[..., None].to(input_ids.device),
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    if gist_embeddings is not None:
        result["gist_embeddings"] = padding(gist_embeddings, left=False).to(input_ids.device)
    return result


@torch.no_grad()
def accepted_lengths_for_batch(
    batch: dict[str, torch.Tensor],
    target_model: Any,
    draft: torch.nn.Module,
    cfg: DraftModelConfig,
    num_spec_tokens: int,
):
    data = prepare_batch_for_draft(batch, target_model, cfg, draft)
    input_ids = data["input_ids"]
    target_logits = data["target_logits"]
    loss_mask = data["loss_mask"]
    hidden_states = data["hidden_states"]
    gist_embeddings = data.get("gist_embeddings")

    hidden_states = draft.combine_hidden_states(hidden_states, gist_embeddings)
    bsz, seq_len, _ = hidden_states.shape
    position_ids = data["position_ids"]
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
    elif position_ids.ndim != 3:
        position_ids = position_ids.view(bsz, seq_len).long()

    attn = draft.prepare_decoder_attention_mask(
        data["attention_mask"], (bsz, seq_len), hidden_states, 0
    )
    cache_hidden = draft.init_cache_hidden() if hasattr(draft, "init_cache_hidden") else [[], []]
    alive = loss_mask.squeeze(-1).bool()
    accepted = torch.zeros_like(input_ids, dtype=torch.long)

    for step in range(num_spec_tokens):
        embeds = draft.embed_input_ids(input_ids)
        hidden_states, cache_hidden = draft.encode_layers(
            inputs_embeds=embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=True,
            gist_embeddings=gist_embeddings,
        )
        logits = draft.compute_logits(hidden_states)
        target_max = target_logits.argmax(-1)
        target_in_draft_vocab = draft.t2d[target_max]
        target_head = target_logits[..., draft.t2d].argmax(-1)
        pred = logits.argmax(-1)
        correct = pred.eq(target_head) & target_in_draft_vocab & alive
        accepted += correct.long()
        alive = alive & correct
        if step < num_spec_tokens - 1:
            input_ids = padding(input_ids, left=False)
            target_logits = padding(target_logits, left=False)
            loss_mask = padding(loss_mask, left=False)
            alive = alive & loss_mask.squeeze(-1).bool()
            if gist_embeddings is not None:
                gist_embeddings = padding(gist_embeddings, left=False)
            if hasattr(draft, "next_hidden_from_encode"):
                hidden_states = draft.next_hidden_from_encode(hidden_states)
            if hasattr(draft, "shift_aux_inject"):
                draft.shift_aux_inject(left=False)
    mask0 = data["loss_mask"].squeeze(-1).bool()
    return accepted[mask0].float()


def main() -> None:
    args = parse_args()
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    torch_dtype = dtype_from_name(args.torch_dtype)
    cfg = DraftModelConfig.from_file(args.draft_model_config_path)
    target = create_target_model(
        backend=args.target_backend,
        model_path=args.target_model,
        modal_type="VLM",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        target_model_type=getattr(cfg, "target_model_type", "smolvlm"),
    )
    draft = load_draft(args, cfg)
    target_model_obj = target.backend.model
    device = args.device or next(target_model_obj.parameters()).device
    draft.to(device=device, dtype=next(target_model_obj.parameters()).dtype)
    draft.freeze_embed_weights()

    gist_on = bool(getattr(cfg, "gist_conditioning", False))
    gist_mode = (args.gist_mode or getattr(cfg, "gist_mode", "remaining")) if gist_on else None
    if gist_on:
        detail = (
            f" refresh_every={getattr(cfg, 'gist_refresh_every', 4)}"
            if gist_mode == "remaining"
            else " (one vector per example)"
        )
        print(f"Oracle gist conditioning: mode={gist_mode}{detail}")
        if args.gist_mode and args.gist_mode != getattr(cfg, "gist_mode", "remaining"):
            print(
                f"WARNING deliberate train/eval gist mismatch: config says "
                f"{getattr(cfg, 'gist_mode', 'remaining')!r}, evaluating with "
                f"{args.gist_mode!r}"
            )

    ds, collator = make_dataset(args, target, cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator)

    all_lengths = []
    for i, batch in enumerate(loader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        lengths = accepted_lengths_for_batch(batch, target, draft, cfg, args.num_spec_tokens)
        if lengths.numel() > 0:
            all_lengths.append(lengths.cpu())
        if args.num_samples and i + 1 >= args.num_samples:
            break

    if not all_lengths:
        raise RuntimeError("No answer/loss tokens evaluated")
    lengths = torch.cat(all_lengths)
    hist = torch.bincount(lengths.long(), minlength=args.num_spec_tokens + 1)[: args.num_spec_tokens + 1]
    result = {
        "draft_model": args.draft_model,
        "draft_model_config_path": args.draft_model_config_path,
        "target_model": args.target_model,
        "data_path": args.data_path,
        "num_positions": int(lengths.numel()),
        "num_spec_tokens": args.num_spec_tokens,
        "mean_accepted_length": float(lengths.mean().item()),
        "median_accepted_length": float(lengths.median().item()),
        "accept_histogram": hist.tolist(),
        "gist_conditioning": gist_on,
        "gist_encoder_model_name_or_path": getattr(cfg, "gist_encoder_model_name_or_path", None),
        "gist_mode": gist_mode,
        # Effective value, not the raw config key: in "remaining" mode a config
        # without gist_refresh_every still runs at the builder default (4), and
        # recording null there would misreport what was actually evaluated.
        "gist_refresh_every": (
            int(getattr(cfg, "gist_refresh_every", 4)) if gist_mode == "remaining" else None
        ),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output_file:
        Path(args.output_file).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
