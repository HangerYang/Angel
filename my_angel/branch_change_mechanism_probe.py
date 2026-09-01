#!/usr/bin/env python3
"""Mechanism probes for branch-change distillation.

Compares the EAGLE 3.1 banded-mix baseline against the branch-change run on
held-out online-training batches. The probes are intentionally about logits,
not acceptance length:

1. Delta alignment on baseline near-miss branches.
2. Repair of baseline near-miss ranking errors.
3. Agreement changes by teacher ambiguity bucket.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from angelslim.compressor.speculative import (
    DatasetManager,
    DraftModelConfig,
    create_draft_model,
    create_target_model,
)
from angelslim.compressor.speculative.utils import padding


BENCH_KEYS = ("low", "medium", "high")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    p.add_argument(
        "--eval-data",
        default="dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl",
    )
    p.add_argument(
        "--baseline",
        default="my_angel/eagle/smolvlm-256m-eagle3-banded-mix-fc-3.1/checkpoint-66466",
    )
    p.add_argument(
        "--branch-change",
        default="my_angel/eagle/branch-change-top1-w01/checkpoint-66466",
    )
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="my_angel/branch_change_mechanism")
    return p.parse_args()


def center(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x - x.mean(dim=-1, keepdim=True)


def shift_left(x: torch.Tensor, n: int = 1) -> torch.Tensor:
    for _ in range(n):
        x = padding(x, left=False)
    return x


def fork_cache(cache: list[Any]) -> list[Any]:
    out = []
    for item in cache:
        if torch.is_tensor(item):
            out.append(item.clone())
        elif isinstance(item, list):
            out.append(fork_cache(item))
        elif isinstance(item, tuple):
            out.append(tuple(fork_cache(list(item))))
        else:
            out.append(item)
    return out


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def load_draft(ckpt: str, device: torch.device):
    ckpt_path = Path(ckpt)
    cfg = DraftModelConfig.from_file(ckpt_path / "config.json")
    model = create_draft_model(cfg)
    state = load_file(str(ckpt_path / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"{ckpt}: unexpected keys: {unexpected[:8]}")
    if missing:
        print(f"{ckpt}: missing keys: {missing[:8]}")
    vocab_cache = ckpt_path.parent / "vocab_mapping_cache.pt"
    vocab = torch.load(vocab_cache, map_location="cpu")
    model.t2d.copy_(vocab["t2d"])
    model.d2t.copy_(vocab["d2t"])
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def prepare_target(target, cfg, batch: dict[str, Any], device: torch.device):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    loss_mask = batch["loss_mask"]
    kwargs = {
        k: v
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
    }
    aux_ids = getattr(cfg, "aux_hidden_states_layer_ids", None)
    hidden, logits, _, position_ids = target.get_hidden_states_and_logits(
        input_ids=input_ids,
        attention_mask=attention_mask,
        aux_hidden_states_layer_ids=aux_ids,
        **kwargs,
    )
    return {
        "orig_ids": input_ids,
        "input_ids": padding(input_ids, left=False),
        "attention_mask": attention_mask,
        "loss_mask": loss_mask[..., None].to(device),
        "hidden": hidden,
        "target_logits": padding(logits, left=False).to(device),
        "position_ids": position_ids,
        "kwargs": kwargs,
    }


@torch.no_grad()
def target_logits_for_ids(target, cfg, data: dict[str, Any], ids: torch.Tensor):
    aux_ids = getattr(cfg, "aux_hidden_states_layer_ids", None)
    _, logits, _, _ = target.get_hidden_states_and_logits(
        input_ids=ids,
        attention_mask=data["attention_mask"],
        aux_hidden_states_layer_ids=aux_ids,
        **data["kwargs"],
    )
    return logits.detach().to(ids.device)


@torch.no_grad()
def draft_two_steps(draft, data: dict[str, Any]):
    hidden = draft.combine_hidden_states(data["hidden"])
    bsz, seq_len, _ = hidden.shape
    attn = draft.prepare_decoder_attention_mask(
        data["attention_mask"], (bsz, seq_len), hidden, 0
    )
    pos = data["position_ids"]
    if pos is None:
        pos = torch.arange(seq_len, device=hidden.device, dtype=torch.long).view(1, -1)
    cache = draft.init_cache_hidden() if hasattr(draft, "init_cache_hidden") else None

    embeds0 = draft.embed_input_ids(data["input_ids"])
    h0, cache = draft.encode_layers(
        inputs_embeds=embeds0,
        hidden_states=hidden,
        cache_hidden=cache,
        attention_mask=attn,
        position_ids=pos,
        use_cache=True,
    )
    logits0 = draft.compute_logits(h0)

    pre1_hidden = draft.next_hidden_from_encode(h0)
    if hasattr(draft, "shift_aux_inject"):
        draft.shift_aux_inject(left=False)
    pre1_cache = fork_cache(cache)
    input_ids1 = padding(data["input_ids"], left=False)

    embeds1 = draft.embed_input_ids(input_ids1)
    h1, _ = draft.encode_layers(
        inputs_embeds=embeds1,
        hidden_states=pre1_hidden,
        cache_hidden=cache,
        attention_mask=attn,
        position_ids=pos,
        use_cache=True,
    )
    base_logits1 = draft.compute_logits(h1)
    return logits0, input_ids1, pre1_hidden, pre1_cache, attn, pos, base_logits1


@torch.no_grad()
def branch_logits(draft, input_ids1, pre1_hidden, pre1_cache, attn, pos, mask, tokens):
    branch_ids = torch.where(mask, tokens.to(input_ids1.dtype), input_ids1)
    embeds = draft.embed_input_ids(branch_ids)
    h, _ = draft.encode_layers(
        inputs_embeds=embeds,
        hidden_states=pre1_hidden,
        cache_hidden=fork_cache(pre1_cache),
        attention_mask=attn,
        position_ids=pos,
        use_cache=True,
    )
    return draft.compute_logits(h)


def update_running(s: dict[str, float], key: str, value: torch.Tensor):
    if value.numel() == 0:
        return
    s[key + "_sum"] = s.get(key + "_sum", 0.0) + float(value.sum().item())
    s[key + "_n"] = s.get(key + "_n", 0.0) + float(value.numel())


def mean(s: dict[str, float], key: str) -> float:
    n = s.get(key + "_n", 0.0)
    return s.get(key + "_sum", 0.0) / n if n else 0.0


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    target = create_target_model(
        backend="hf",
        model_path=args.target,
        modal_type="VLM",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        target_model_type="smolvlm",
    )
    target.model.to(device).eval()

    baseline, cfg = load_draft(args.baseline, device)
    branch_change, _ = load_draft(args.branch_change, device)

    data_args = SimpleNamespace(
        modal_type="VLM",
        training_mode="online",
        train_data_path=None,
        eval_data_path=args.eval_data,
        num_proc=1,
        sample_num=args.samples,
        load_from_cache_file=True,
        shuffle_seed=42,
        target_model_name_or_path=args.target,
        output_dir=str(out_dir / "dataset_cache"),
    )
    dm = DatasetManager(
        data_args=data_args,
        tokenizer=target.tokenizer,
        model_max_length=args.max_length,
        chat_template_type="smolvlm",
        display=False,
        target_model_type="smolvlm",
    )
    _, dataset, collator = dm.create_online_datasets()

    stats: dict[str, float] = {}
    buckets = {k: {"total": 0.0, "base_correct": 0.0, "chg_correct": 0.0} for k in BENCH_KEYS}

    total_positions = 0
    total_near_miss = 0
    for i in range(len(dataset)):
        batch = move_batch(collator([dataset[i]]), device)
        data = prepare_target(target, cfg, batch, device)
        tlog = data["target_logits"]
        loss_mask = data["loss_mask"][..., 0].bool()

        b0, b_ids1, b_h1, b_cache1, b_attn, b_pos, b_base1 = draft_two_steps(
            baseline, data
        )
        c0, c_ids1, c_h1, c_cache1, c_attn, c_pos, c_base1 = draft_two_steps(
            branch_change, data
        )

        b_draft_top_d = b0.argmax(-1)
        b_draft_top = b_draft_top_d + baseline.d2t[b_draft_top_d]
        c_draft_top_d = c0.argmax(-1)
        c_draft_top = c_draft_top_d + branch_change.d2t[c_draft_top_d]
        teacher_top = tlog.argmax(-1)
        teacher_top3 = tlog.topk(3, dim=-1).indices
        teacher_top2 = tlog.topk(2, dim=-1).indices

        cols = torch.arange(tlog.shape[1], device=device)
        in_range = (cols + 2 < data["orig_ids"].shape[1])[None, :]
        scorable = loss_mask & in_range
        near = (
            scorable
            & (b_draft_top != teacher_top)
            & (teacher_top3 == b_draft_top[..., None]).any(-1)
        )
        total_positions += int(scorable.sum().item())
        total_near_miss += int(near.sum().item())

        probs = F.softmax(tlog.float(), dim=-1)
        ratio2 = probs.gather(-1, teacher_top2[..., 1:2]).squeeze(-1) / probs.gather(
            -1, teacher_top2[..., 0:1]
        ).squeeze(-1).clamp_min(1e-12)
        for name, m in {
            "low": scorable & (ratio2 < 0.05),
            "medium": scorable & (ratio2 >= 0.05) & (ratio2 < 0.2),
            "high": scorable & (ratio2 >= 0.2),
        }.items():
            if bool(m.any()):
                buckets[name]["total"] += float(m.sum().item())
                buckets[name]["base_correct"] += float((b_draft_top[m] == teacher_top[m]).sum().item())
                buckets[name]["chg_correct"] += float((c_draft_top[m] == teacher_top[m]).sum().item())

        if not bool(near.any()):
            continue

        rows, cols_near = torch.nonzero(near, as_tuple=True)
        sub_ids = data["orig_ids"].clone()
        sub_ids[rows, cols_near + 2] = b_draft_top[rows, cols_near].to(sub_ids.dtype)
        branch_tlog = shift_left(target_logits_for_ids(target, cfg, data, sub_ids), 2)
        real_next_tlog = shift_left(tlog, 1)
        target_delta = (
            center(branch_tlog[..., baseline.t2d]) - center(real_next_tlog[..., baseline.t2d])
        )

        b_branch1 = branch_logits(
            baseline, b_ids1, b_h1, b_cache1, b_attn, b_pos, near, b_draft_top
        )
        c_branch1 = branch_logits(
            branch_change, c_ids1, c_h1, c_cache1, c_attn, c_pos, near, b_draft_top
        )
        b_delta = center(b_branch1) - center(b_base1)
        c_delta = center(c_branch1) - center(c_base1)

        td = target_delta[near]
        bd = b_delta[near]
        cd = c_delta[near]
        update_running(stats, "delta_mse_baseline", (bd - td).pow(2).mean(-1))
        update_running(stats, "delta_mse_change", (cd - td).pow(2).mean(-1))
        update_running(stats, "delta_cos_baseline", F.cosine_similarity(bd, td, dim=-1))
        update_running(stats, "delta_cos_change", F.cosine_similarity(cd, td, dim=-1))

        t_top_delta = td.abs().topk(10, dim=-1).indices
        b_top_delta = bd.abs().topk(10, dim=-1).indices
        c_top_delta = cd.abs().topk(10, dim=-1).indices
        b_overlap = (t_top_delta[..., None] == b_top_delta[:, None, :]).any(-1).float().mean(-1)
        c_overlap = (t_top_delta[..., None] == c_top_delta[:, None, :]).any(-1).float().mean(-1)
        update_running(stats, "top10_delta_overlap_baseline", b_overlap)
        update_running(stats, "top10_delta_overlap_change", c_overlap)

        teacher_top_d = None
        # Convert teacher target ids to draft ids by looking up equality in the
        # selected target-vocab view. This is small because it only runs on
        # near-miss positions.
        draft_target_ids = torch.nonzero(baseline.t2d, as_tuple=False).flatten()
        local = []
        for tok in teacher_top[near].detach().cpu().tolist():
            hit = (draft_target_ids.cpu() == tok).nonzero(as_tuple=False)
            local.append(int(hit[0].item()) if hit.numel() else -1)
        teacher_top_d = torch.tensor(local, device=device)
        valid = teacher_top_d >= 0
        if bool(valid.any()):
            bd0 = b0[near][valid]
            cd0 = c0[near][valid]
            tdid = teacher_top_d[valid]
            b_rank = (bd0 > bd0.gather(-1, tdid[:, None])).sum(-1) + 1
            c_rank = (cd0 > cd0.gather(-1, tdid[:, None])).sum(-1) + 1
            update_running(stats, "teacher_top1_rank_baseline", b_rank.float())
            update_running(stats, "teacher_top1_rank_change", c_rank.float())
            update_running(stats, "teacher_top1_logit_baseline", bd0.gather(-1, tdid[:, None]).squeeze(-1))
            update_running(stats, "teacher_top1_logit_change", cd0.gather(-1, tdid[:, None]).squeeze(-1))
            update_running(stats, "change_corrects_baseline_nearmiss", (c_draft_top[near][valid] == teacher_top[near][valid]).float())

    result = {
        "samples": len(dataset),
        "positions": total_positions,
        "baseline_near_miss_positions": total_near_miss,
        "baseline_near_miss_rate": total_near_miss / max(total_positions, 1),
        "delta_alignment": {
            "mse_baseline": mean(stats, "delta_mse_baseline"),
            "mse_branch_change": mean(stats, "delta_mse_change"),
            "cos_baseline": mean(stats, "delta_cos_baseline"),
            "cos_branch_change": mean(stats, "delta_cos_change"),
            "top10_overlap_baseline": mean(stats, "top10_delta_overlap_baseline"),
            "top10_overlap_branch_change": mean(stats, "top10_delta_overlap_change"),
        },
        "near_miss_repair": {
            "teacher_top1_rank_baseline": mean(stats, "teacher_top1_rank_baseline"),
            "teacher_top1_rank_branch_change": mean(stats, "teacher_top1_rank_change"),
            "teacher_top1_logit_baseline": mean(stats, "teacher_top1_logit_baseline"),
            "teacher_top1_logit_branch_change": mean(stats, "teacher_top1_logit_change"),
            "branch_change_top1_correct_on_baseline_near_miss": mean(
                stats, "change_corrects_baseline_nearmiss"
            ),
        },
        "ambiguity_buckets": {
            k: {
                "positions": v["total"],
                "top1_agreement_baseline": v["base_correct"] / v["total"]
                if v["total"]
                else 0.0,
                "top1_agreement_branch_change": v["chg_correct"] / v["total"]
                if v["total"]
                else 0.0,
            }
            for k, v in buckets.items()
        },
    }
    (out_dir / "mechanism_probe.json").write_text(json.dumps(result, indent=2))

    md = ["# Branch-Change Mechanism Probe", ""]
    md += [
        f"- samples: {result['samples']}",
        f"- scorable positions: {result['positions']}",
        f"- baseline near-miss positions: {result['baseline_near_miss_positions']} "
        f"({100*result['baseline_near_miss_rate']:.2f}%)",
        "",
        "## Delta Alignment",
        "",
        "| metric | baseline | branch-change |",
        "|---|---:|---:|",
    ]
    da = result["delta_alignment"]
    for key in ("mse", "cos", "top10_overlap"):
        md.append(
            f"| {key} | {da[key + '_baseline']:.4f} | "
            f"{da[key + '_branch_change']:.4f} |"
        )
    md += [
        "",
        "## Near-Miss Repair",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    nr = result["near_miss_repair"]
    for key, value in nr.items():
        md.append(f"| {key} | {value:.4f} |")
    md += [
        "",
        "## Ambiguity Buckets",
        "",
        "| bucket | positions | baseline top1 agree | branch-change top1 agree | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, v in result["ambiguity_buckets"].items():
        b = v["top1_agreement_baseline"]
        c = v["top1_agreement_branch_change"]
        md.append(f"| {key} | {v['positions']:.0f} | {100*b:.2f}% | {100*c:.2f}% | {100*(c-b):+.2f} pts |")
    (out_dir / "mechanism_probe.md").write_text("\n".join(md) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
