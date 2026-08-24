#!/usr/bin/env python
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""Generate SmolVLM HiViS offline hidden states for Eagle3 training.

This converts the existing SmolVLM online JSONL data into .ckpt samples accepted
by tools/train_eagle3_offline.py. The frozen target runs on the full multimodal
sequence; the saved draft sequence can be compacted with the HiViS visual-token
prune rule when the draft config sets hivis_remove_image_tokens=true.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from angelslim.compressor.speculative import (
    DatasetManager,
    DraftModelConfig,
    create_target_model,
    infer_model_params,
)
from angelslim.compressor.speculative.hivis import build_hivis_keep_mask
from angelslim.compressor.speculative.train.data.data_utils import (
    process_token_dict_to_mappings,
)
from angelslim.utils import rank0_print


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SmolVLM HiViS offline hidden-state .ckpt data"
    )
    parser.add_argument(
        "--target_model_name_or_path",
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Frozen target VLM used to produce auxiliary/final hidden states.",
    )
    parser.add_argument(
        "--draft_model_config_path",
        default=(
            "angelslim/compressor/speculative/train/configs/"
            "smolvlm-256m-eagle3-progressive-hivis.json"
        ),
        help="Draft config; supplies aux layers, vocab sizes, and HiViS IDs.",
    )
    parser.add_argument(
        "--train_data_path",
        default="dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl",
        help="Training JSONL/raw data path accepted by the SmolVLM online builder.",
    )
    parser.add_argument(
        "--eval_data_path",
        default="dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl",
        help="Eval JSONL/raw data path. Empty string disables eval generation.",
    )
    parser.add_argument(
        "--output_dir",
        default="dataset/smolvlm_256m_hivis_offline",
        help="Output root; split data is written under output_dir/{train,eval}.",
    )
    parser.add_argument("--split", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument(
        "--load_from_cache_file",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Reuse HF datasets map cache from the online SmolVLM builder.",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--chat_template_type", default=None)
    parser.add_argument("--target_backend", default="hf", choices=["hf"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _to_device(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _compact_sample(
    row: dict[str, torch.Tensor | None],
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    prune_visual_tokens: bool,
    image_token_id: int | None,
    vision_start_token_id: int | None,
    vision_end_token_id: int | None,
) -> tuple[dict[str, torch.Tensor], int]:
    seq_len = int(attention_mask.long().sum().item())
    seq_len = max(seq_len, 1)
    keep = torch.zeros_like(input_ids, dtype=torch.bool)
    keep[:seq_len] = True
    if prune_visual_tokens:
        hivis_keep = build_hivis_keep_mask(
            input_ids[:seq_len],
            image_token_id=image_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
        )
        keep[:seq_len] = hivis_keep

    kept = int(keep.sum().item())
    if kept == 0:
        keep[0] = True
        kept = 1
    dropped = seq_len - kept

    out: dict[str, torch.Tensor] = {}
    for key, tensor in row.items():
        if tensor is None:
            continue
        if tensor.ndim >= 1 and tensor.shape[0] == input_ids.shape[0]:
            out[key] = tensor[keep].unsqueeze(0).detach().cpu()
        else:
            out[key] = tensor.detach().cpu()
    return out, dropped


def _save_vocab_mapping(
    token_dict: Counter,
    split_dir: Path,
    draft_config: DraftModelConfig,
) -> None:
    d2t, t2d = process_token_dict_to_mappings(
        token_dict,
        int(draft_config.draft_vocab_size),
        int(draft_config.vocab_size),
    )
    torch.save({"d2t": d2t, "t2d": t2d}, split_dir / "vocab_mapping.pt")


def _generate_split(
    *,
    split_name: str,
    dataset,
    collator,
    target_model,
    draft_config: DraftModelConfig,
    output_root: Path,
    batch_size: int,
    overwrite: bool,
) -> None:
    if dataset is None:
        rank0_print(f"Skipping {split_name}: no dataset")
        return

    split_dir = output_root / split_name
    if split_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{split_dir} already exists. Pass --overwrite to regenerate it."
            )
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    device = next(target_model.model.parameters()).device
    aux_ids = list(getattr(draft_config, "aux_hidden_states_layer_ids", []))
    prune_visual_tokens = bool(getattr(draft_config, "hivis_remove_image_tokens", False))
    image_token_id = getattr(draft_config, "image_token_id", None)
    vision_start_token_id = getattr(draft_config, "vision_start_token_id", None)
    vision_end_token_id = getattr(draft_config, "vision_end_token_id", None)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    token_dict: Counter = Counter()
    saved = 0
    dropped_total = 0

    for batch in tqdm(loader, desc=f"generate {split_name}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        forward_kwargs = {
            k: _to_device(v, device)
            for k, v in batch.items()
            if k
            not in {
                "input_ids",
                "attention_mask",
                "loss_mask",
                "hidden_states",
                "target_hiddens",
                "inputs_embeds",
                "position_ids",
            }
            and v is not None
        }
        outputs = target_model.get_aux_and_target_hiddens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aux_hidden_states_layer_ids=aux_ids,
            **forward_kwargs,
        )

        batch_rows = input_ids.shape[0]
        for row_idx in range(batch_rows):
            row = {
                "input_ids": input_ids[row_idx],
                "hidden_states": outputs["hidden_states"][row_idx],
                "target_hiddens": outputs["target_hiddens"][row_idx],
                "loss_mask": batch["loss_mask"].to(device)[row_idx],
                "inputs_embeds": (
                    outputs.get("inputs_embeds")[row_idx]
                    if outputs.get("inputs_embeds") is not None
                    and outputs.get("inputs_embeds").shape[0] == batch_rows
                    else None
                ),
                "position_ids": (
                    outputs.get("position_ids")[row_idx]
                    if outputs.get("position_ids") is not None
                    and outputs.get("position_ids").ndim >= 2
                    and outputs.get("position_ids").shape[0] == batch_rows
                    else None
                ),
            }
            sample, dropped = _compact_sample(
                row,
                attention_mask[row_idx],
                input_ids[row_idx],
                prune_visual_tokens=prune_visual_tokens,
                image_token_id=image_token_id,
                vision_start_token_id=vision_start_token_id,
                vision_end_token_id=vision_end_token_id,
            )
            ids = sample["input_ids"].view(-1).tolist()
            token_dict.update(int(i) for i in ids)
            torch.save(sample, split_dir / f"{saved:08d}.ckpt")
            saved += 1
            dropped_total += dropped

    if split_name == "train":
        _save_vocab_mapping(token_dict, split_dir, draft_config)
    stats = {
        "split": split_name,
        "saved": saved,
        "hivis_remove_image_tokens": prune_visual_tokens,
        "dropped_visual_tokens": dropped_total,
        "output_dir": str(split_dir),
    }
    (split_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    rank0_print(f"Saved {saved} {split_name} samples to {split_dir}")


def main() -> None:
    args = parse_args()
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    draft_config = DraftModelConfig.from_file(args.draft_model_config_path)
    target_model_type = getattr(draft_config, "target_model_type", "smolvlm")
    _, _, inferred_chat_template_type = infer_model_params(
        model_name_or_path=args.target_model_name_or_path,
        model_type=target_model_type,
    )
    chat_template_type = args.chat_template_type or inferred_chat_template_type or target_model_type

    rank0_print(f"Loading target model: {args.target_model_name_or_path}")
    target_model = create_target_model(
        backend=args.target_backend,
        model_path=args.target_model_name_or_path,
        modal_type="VLM",
        torch_dtype=dtype_mapping[args.torch_dtype],
        trust_remote_code=args.trust_remote_code,
        target_model_type=target_model_type,
    )

    data_args = SimpleNamespace(
        modal_type="VLM",
        training_mode="online",
        target_model_name_or_path=args.target_model_name_or_path,
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path or None,
        num_proc=args.num_proc,
        sample_num=args.sample_num,
        shuffle_seed=args.shuffle_seed,
        load_from_cache_file=args.load_from_cache_file == "true",
        output_dir=args.output_dir,
    )
    manager = DatasetManager(
        data_args=data_args,
        tokenizer=target_model.tokenizer,
        model_max_length=args.model_max_length,
        chat_template_type=chat_template_type,
        target_model_type=target_model_type,
    )
    train_dataset, eval_dataset, collator = manager.create_online_datasets()
    output_root = Path(args.output_dir)

    if args.split in ("train", "all"):
        _generate_split(
            split_name="train",
            dataset=train_dataset,
            collator=collator,
            target_model=target_model,
            draft_config=draft_config,
            output_root=output_root,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    if args.split in ("eval", "all"):
        _generate_split(
            split_name="eval",
            dataset=eval_dataset,
            collator=collator,
            target_model=target_model,
            draft_config=draft_config,
            output_root=output_root,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
