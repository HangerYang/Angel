#!/usr/bin/env python
"""Recover HF-trainer acc_0..acc_k for a SmolVLM Eagle/Hawk checkpoint.

This intentionally reuses the AngelSlim training objects instead of
reimplementing the forward/loss path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from torch.utils.data import DataLoader
from torch import nn

from angelslim.compressor.speculative.utils import padding
from angelslim.compressor.speculative import (
    DatasetManager,
    DraftModelConfig,
    Eagle3TrainerFactory,
    create_draft_model,
    create_target_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="output/smolvlm_256m_hawk_nccl/checkpoint-66466",
    )
    p.add_argument(
        "--target-model",
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
    )
    p.add_argument(
        "--train-data",
        default="dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl",
    )
    p.add_argument("--output-dir", default="/tmp/recover_hf_train_acc_smolvlm_hawk")
    p.add_argument("--sample-num", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--model-max-length", type=int, default=4096)
    p.add_argument("--num-proc", type=int, default=1)
    p.add_argument("--length", type=int, default=7)
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument("--load-from-cache-file", action="store_true")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--rollout", choices=["teacher", "free", "both"], default="teacher")
    p.add_argument("--print-every", type=int, default=1)
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]



def compute_free_rollout_metrics(trainer, batch, length: int) -> dict[str, float]:
    """Run the trainer preparation path, then feed draft argmax tokens back."""
    data = trainer.prepare_data_for_draft_model(batch)
    attention_mask = data["attention_mask"]
    position_ids = data["position_ids"]
    input_ids = data["input_ids"]
    target_logits = data["target_logits"]
    loss_mask = data["loss_mask"]
    hidden_states = data["hidden_states"]

    hidden_states = trainer.down_project_hidden_states(hidden_states)
    attention_mask, position_ids = trainer.prepare_attention_mask_and_position_ids(
        hidden_states, attention_mask, position_ids
    )

    draft = trainer.draft_model
    cache_hidden = draft.init_cache_hidden() if hasattr(draft, "init_cache_hidden") else [[], []]
    use_draft_feedback = bool(
        getattr(draft, "progressive_staged", False) or getattr(draft, "hawk", False)
    )

    correct_cols = []
    valid_cols = []
    with torch.no_grad():
        for idx in range(length):
            inputs_embeds = draft.embed_input_ids(input_ids)
            hidden_states, cache_hidden = draft.encode_layers(
                inputs_embeds=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            logits = draft.compute_logits(hidden_states)
            target_max_token = target_logits.argmax(-1)
            target_mask = draft.t2d[target_max_token][..., None].int()
            position_mask = target_mask * loss_mask
            target_head = target_logits[..., draft.t2d].float()
            target_p = nn.Softmax(dim=2)(target_head)
            pred = logits.argmax(-1)
            tgt = target_p.argmax(-1)
            valid = position_mask.squeeze(-1).bool()
            correct = (pred == tgt) & valid
            correct_cols.append(correct)
            valid_cols.append(valid)

            if idx < length - 1:
                # Strip teacher forcing: next draft input is its own argmax token.
                pred_target_ids = pred + draft.d2t[pred]
                input_ids = pred_target_ids
                target_logits = padding(target_logits, left=False)
                loss_mask = padding(loss_mask, left=False)
                if use_draft_feedback:
                    seed = draft.take_progressive_draft_feedback()
                    if seed is None:
                        raise RuntimeError("draft feedback returned None in free rollout")
                    hidden_states = seed
                elif hasattr(draft, "shift_aux_inject"):
                    draft.shift_aux_inject(left=False)

    metrics: dict[str, float] = {}
    for i, (correct, valid) in enumerate(zip(correct_cols, valid_cols)):
        denom = valid.sum().item()
        metrics[f"free_acc_{i}"] = correct.sum().item() / (denom + 1e-6)

    prefix_alive = None
    accepted = torch.zeros_like(valid_cols[0], dtype=torch.long)
    for correct, valid in zip(correct_cols, valid_cols):
        cur = (correct & valid) if prefix_alive is None else (prefix_alive & correct & valid)
        accepted += cur.long()
        prefix_alive = cur
    draft_windows = valid_cols[0].sum().item()
    metrics["free_mean_acceptance_length"] = 1.0 + accepted.sum().item() / (draft_windows + 1e-6)
    for i, valid in enumerate(valid_cols):
        alive = correct_cols[0] & valid_cols[0]
        for j in range(1, i + 1):
            alive = alive & correct_cols[j] & valid_cols[j]
        metrics[f"free_acceptance_rate_pos_{i}"] = alive.sum().item() / (draft_windows + 1e-6)
    return metrics

def main() -> None:
    args = parse_args()
    ckpt = Path(args.checkpoint)
    config_path = ckpt / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    torch_dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    draft_config = DraftModelConfig.from_file(str(config_path))
    target_model_type = getattr(draft_config, "target_model_type", None)
    print(f"checkpoint={ckpt}")
    print(f"device={device} dtype={torch_dtype}")
    print(
        "draft config:",
        json.dumps(
            {
                "mode": getattr(draft_config, "eagle_aux_injection_mode", "fused_fc"),
                "num_hidden_layers": getattr(draft_config, "num_hidden_layers", None),
                "aux_hidden_states_layer_ids": getattr(
                    draft_config, "aux_hidden_states_layer_ids", None
                ),
                "eagle_aux_hidden_state_layer_ids": getattr(
                    draft_config, "eagle_aux_hidden_state_layer_ids", None
                ),
                "draft_layer_init_from_target": getattr(
                    draft_config, "draft_layer_init_from_target", None
                ),
            },
            sort_keys=True,
        ),
    )

    print("loading target model...")
    target_model = create_target_model(
        backend="hf",
        model_path=args.target_model,
        modal_type="VLM",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        target_model_type=target_model_type,
    )

    print("loading draft checkpoint through training class...")
    draft_model = create_draft_model(draft_config)
    loaded = draft_model.__class__.from_pretrained(
        str(ckpt),
        config=draft_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    missing, unexpected = draft_model.load_state_dict(loaded.state_dict(), strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing[:10]=", missing[:10])
    if unexpected:
        print("  unexpected[:10]=", unexpected[:10])
    del loaded
    draft_model.to(device)
    draft_model.eval()

    data_args = SimpleNamespace(
        modal_type="VLM",
        training_mode="online",
        train_data_path=[args.train_data],
        eval_data_path=None,
        num_proc=args.num_proc,
        sample_num=args.sample_num,
        load_from_cache_file=args.load_from_cache_file,
        shuffle_seed=args.shuffle_seed,
        display=False,
        target_model_name_or_path=args.target_model,
        output_dir=args.output_dir,
    )
    print("building dataset...")
    dataset_manager = DatasetManager(
        data_args=data_args,
        tokenizer=target_model.tokenizer,
        model_max_length=args.model_max_length,
        chat_template_type="smolvlm",
        display=False,
        target_model_type=target_model_type,
    )
    train_dataset, _, data_collator = dataset_manager.create_online_datasets()
    print(f"dataset size={len(train_dataset)}")

    cache_path = Path(args.checkpoint).parent / "vocab_mapping_cache.pt"
    print(f"loading vocab mapping from {cache_path}")
    draft_model.build_vocab_mapping(dataset=train_dataset, cache_path=str(cache_path))

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        remove_unused_columns=False,
        report_to=[],
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        dataloader_num_workers=0,
    )
    trainer = Eagle3TrainerFactory.create(
        training_mode="online",
        modal_type="VLM",
        draft_model=draft_model,
        target_model=target_model,
        length=args.length,
        draft_model_config=draft_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        target_hs_warmup_steps=0,
    )
    if trainer.state is not None:
        trainer.state.global_step = 66466

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    print(f"running rollout={args.rollout}...")
    free_sum: dict[str, float] = {}
    free_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            if args.rollout in ("teacher", "both"):
                _ = trainer.compute_loss(draft_model, batch, return_outputs=False)
                n = trainer._train_pending_log_count
                acc = {
                    f"acc_{i}": trainer._train_pending_log.get(f"train/acc_{i}", 0.0) / max(n, 1)
                    for i in range(args.length)
                }
                if args.print_every > 0 and (batch_idx + 1) % args.print_every == 0:
                    print(f"after batch {batch_idx + 1} teacher: " + json.dumps(acc, sort_keys=True))
            if args.rollout in ("free", "both"):
                m = compute_free_rollout_metrics(trainer, batch, args.length)
                for k, v in m.items():
                    free_sum[k] = free_sum.get(k, 0.0) + float(v)
                free_count += 1
                avg = {k: v / free_count for k, v in sorted(free_sum.items())}
                if args.print_every > 0 and (batch_idx + 1) % args.print_every == 0:
                    print(f"after batch {batch_idx + 1} free: " + json.dumps(avg, sort_keys=True))

    if args.rollout in ("teacher", "both"):
        n = max(trainer._train_pending_log_count, 1)
        final = {
            k.removeprefix("train/"): v / n
            for k, v in sorted(trainer._train_pending_log.items())
            if k.startswith("train/acc_")
        }
        print("FINAL_TEACHER_RAW", json.dumps(final, sort_keys=True))
        print("FINAL_TEACHER_ROUNDED", json.dumps({k: round(v, 6) for k, v in final.items()}, sort_keys=True))
    if args.rollout in ("free", "both"):
        final_free = {k: v / max(free_count, 1) for k, v in sorted(free_sum.items())}
        print("FINAL_FREE_RAW", json.dumps(final_free, sort_keys=True))
        print("FINAL_FREE_ROUNDED", json.dumps({k: round(v, 6) for k, v in final_free.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
