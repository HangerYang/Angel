"""Train a Q-Sampler to compress SmolVLM visual tokens, 64 -> N per tile.

Everything in SmolVLM is frozen. Two forwards run per step over the same frozen
backbone:

  teacher   64 tokens/tile  ->  logits_t, h_t
  student   N  tokens/tile  ->  logits_s, h_s          (N tokens come from the Q-Sampler)

The vision tower runs ONCE per step: its connector output feeds the teacher
directly and the Q-Sampler for the student, so the extra cost is one text
forward, not a second full VLM forward.

Loss:  L = KL(teacher || student) + lambda * (1 - cos(h_t, h_s))
Both terms are evaluated only at text-token positions, gathered through the
collator's alignment index -- the two branches have different sequence lengths,
so absolute positions do not correspond.

Usage (torchrun):
    torchrun --nproc_per_node=4 tools/train_qsampler.py \
        --train_data_path dataset/.../train_images_only.jsonl \
        --num_queries 4 --num_train_epochs 2 --output_dir output/qsampler-n4
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from angelslim.compressor.vistoken import QSampler, QSamplerConfig  # noqa: E402
from angelslim.compressor.vistoken.data import (  # noqa: E402
    JsonlOffsetDataset,
    QSamplerCollator,
)


def rank0(*a):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    p.add_argument("--train_data_path", required=True)
    p.add_argument("--eval_data_path", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_queries", type=int, default=4)
    p.add_argument("--num_blocks", type=int, default=1)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_scheduler", choices=["cosine", "linear", "constant"],
                   default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="overrides --warmup_ratio when > 0")
    p.add_argument("--min_lr_ratio", type=float, default=0.1,
                   help="floor of the decay, as a fraction of --learning_rate")
    p.add_argument("--no_calibrate_output_scale", action="store_true",
                   help="skip matching the sampler output RMS to the connector's")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--hidden_layers", default="26",
                   help="comma-separated decoder-block ids for the auxiliary loss; "
                        "these index hidden_states[i+1] (same convention as "
                        "aux_hidden_states_layer_ids)")
    p.add_argument("--hidden_loss", choices=["cosine", "nmse"], default="cosine")
    p.add_argument("--lambda_hidden", type=float, default=0.3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit_rows", type=int, default=0)
    return p.parse_args()


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return dist.get_rank(), dist.get_world_size(), local
    return 0, 1, 0


@torch.no_grad()
def tile_features(model, pixel_values):
    """Frozen vision tower + connector -> [total_tiles, 64, hidden]."""
    return model.model.get_image_features(pixel_values=pixel_values).pooler_output


def run_text_branch(model, input_ids, attention_mask, features, want_layers):
    """Scatter `features` into the <image> slots and run the frozen text model."""
    embed = model.model.get_input_embeddings()
    inputs_embeds = embed(input_ids)
    image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1)
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask, features.to(inputs_embeds.dtype)
    )
    out = model.model.text_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=bool(want_layers),
        use_cache=False,
    )
    logits = model.lm_head(out.last_hidden_state)
    hidden = (
        [out.hidden_states[i + 1] for i in want_layers] if want_layers else []
    )
    return logits, hidden


def gather_text(x, index, valid):
    """Pick text-token rows: x[b, index[b, k]] where valid[b, k]."""
    idx = index.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    picked = torch.gather(x, 1, idx)
    return picked[valid]


def compute_losses(args, t_logits, s_logits, t_hidden, s_hidden, batch):
    ti = batch["teacher_text_index"]
    si = batch["student_text_index"]
    vm = batch["text_valid_mask"]

    tl = gather_text(t_logits.float(), ti, vm)
    sl = gather_text(s_logits.float(), si, vm)
    # Forward KL(teacher || student): preserve the teacher's behaviour.
    kl = F.kl_div(
        F.log_softmax(sl, dim=-1),
        F.log_softmax(tl, dim=-1),
        log_target=True,
        reduction="batchmean",
    )

    hid = torch.zeros((), device=kl.device, dtype=kl.dtype)
    for th, sh in zip(t_hidden, s_hidden):
        a = gather_text(th.float(), ti, vm)
        b = gather_text(sh.float(), si, vm)
        if args.hidden_loss == "cosine":
            hid = hid + (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()
        else:
            # Normalised MSE: scale-free, so late layers with large activation
            # norms do not dominate earlier ones.
            hid = hid + (
                (a - b).pow(2).sum(-1) / a.pow(2).sum(-1).clamp_min(1e-6)
            ).mean()
    if t_hidden:
        hid = hid / len(t_hidden)

    with torch.no_grad():
        agree = (tl.argmax(-1) == sl.argmax(-1)).float().mean()
        cos = F.cosine_similarity(
            gather_text(t_hidden[-1].float(), ti, vm),
            gather_text(s_hidden[-1].float(), si, vm),
            dim=-1,
        ).mean() if t_hidden else torch.zeros((), device=kl.device)

    return kl + args.lambda_hidden * hid, {
        "kl": kl.item(),
        "hidden": hid.detach().item(),
        "top1_agree": agree.item(),
        "hidden_cos": cos.item(),
    }


def main():
    args = parse_args()
    rank, world, local = setup_dist()
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    want_layers = [int(x) for x in args.hidden_layers.split(",") if x.strip() != ""]

    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(args.target_model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.target_model, dtype=torch.bfloat16
    ).to(device)
    model.eval()
    model.requires_grad_(False)

    hidden_size = model.config.text_config.hidden_size
    sampler_cfg = QSamplerConfig(
        hidden_size=hidden_size,
        num_queries=args.num_queries,
        num_blocks=args.num_blocks,
        num_heads=model.config.text_config.num_attention_heads,
    )
    qsampler = QSampler(sampler_cfg).to(device=device, dtype=torch.float32)
    n_params = sum(p.numel() for p in qsampler.parameters())
    rank0(
        f"Q-Sampler: num_queries={args.num_queries} blocks={args.num_blocks} "
        f"params={n_params:,}  (64 -> {args.num_queries} per tile)"
    )
    rank0(
        f"aux hidden layers (decoder blocks) {want_layers} "
        f"-> hidden_states{[i + 1 for i in want_layers]}"
    )

    # The Q-Sampler is the only hot part: assert it, do not just intend it.
    hot_backbone = [n for n, q in model.named_parameters() if q.requires_grad]
    if hot_backbone:
        raise RuntimeError(
            f"{len(hot_backbone)} SmolVLM parameters are still trainable, "
            f"e.g. {hot_backbone[:5]}"
        )
    rank0(
        f"frozen SmolVLM: {sum(q.numel() for q in model.parameters()):,} params, "
        f"0 trainable | Q-Sampler: {n_params:,} params, all trainable"
    )

    ddp = DDP(qsampler, device_ids=[local]) if world > 1 else qsampler
    # `sampler` is what the training forward MUST go through: calling .module
    # directly skips DDP's autograd hooks, so gradients would never all-reduce
    # and each rank would silently train its own copy. `trainable` is the raw
    # module, used only for parameters / state_dict / no-grad eval.
    sampler = ddp
    trainable = ddp.module if world > 1 else ddp

    collate = QSamplerCollator(
        processor,
        num_queries=args.num_queries,
        max_length=args.max_length,
        image_token_id=model.config.image_token_id,
    )
    train_ds = JsonlOffsetDataset(args.train_data_path, limit=args.limit_rows)
    rank0(f"train rows: {len(train_ds):,}")
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
        if world > 1
        else None
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )

    eval_dl = None
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        eval_ds = JsonlOffsetDataset(args.eval_data_path)
        rank0(f"eval rows: {len(eval_ds):,}")
        eval_dl = DataLoader(
            eval_ds,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate,
        )

    steps_per_epoch = math.ceil(
        len(train_dl) / args.gradient_accumulation_steps
    )
    total_steps = args.max_steps or int(steps_per_epoch * args.num_train_epochs)
    # Decay on weights only; norms, biases, and the learned query/positional
    # tables are excluded -- weight-decaying the queries pulls them toward zero,
    # which is exactly the degenerate all-identical-slot solution.
    decay, no_decay = [], []
    for n, q in trainable.named_parameters():
        (no_decay if q.ndim <= 1 or n.endswith(("queries", "kv_pos")) else decay).append(q)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    warmup = args.warmup_steps or max(1, int(total_steps * args.warmup_ratio))
    floor = args.min_lr_ratio

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        p = min(1.0, p)
        if args.lr_scheduler == "constant":
            return 1.0
        if args.lr_scheduler == "linear":
            return floor + (1.0 - floor) * (1.0 - p)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rank0(
        f"optim: AdamW lr={args.learning_rate} betas=({args.adam_beta1},"
        f"{args.adam_beta2}) wd={args.weight_decay} (decay {len(decay)} tensors / "
        f"no-decay {len(no_decay)}) | sched={args.lr_scheduler} warmup={warmup} "
        f"min_lr_ratio={floor} clip={args.max_grad_norm}"
    )
    rank0(
        f"steps/epoch={steps_per_epoch} total_optimizer_steps={total_steps} "
        f"world={world} eff_batch="
        f"{args.per_device_batch_size * args.gradient_accumulation_steps * world}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if rank == 0:
        with open(os.path.join(args.output_dir, "qsampler_config.json"), "w") as f:
            json.dump({**sampler_cfg.__dict__, "args": vars(args)}, f, indent=2)

    def save(tag):
        if rank != 0:
            return
        d = os.path.join(args.output_dir, tag)
        os.makedirs(d, exist_ok=True)
        torch.save(trainable.state_dict(), os.path.join(d, "qsampler.pt"))
        # Optimizer + schedule travel with the weights so a run can resume
        # mid-schedule instead of restarting warmup.
        torch.save(
            {
                "step": step,
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "lr": sched.get_last_lr()[0],
                "args": vars(args),
            },
            os.path.join(d, "trainer_state.pt"),
        )
        rank0(f"saved {d}")

    @torch.no_grad()
    def evaluate():
        if eval_dl is None:
            return {}
        trainable.eval()
        acc, n = {}, 0
        for i, batch in enumerate(eval_dl):
            if i >= args.eval_batches or batch is None:
                if batch is None:
                    continue
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            feats = tile_features(model, batch["pixel_values"])
            t_logits, t_hidden = run_text_branch(
                model, batch["teacher_input_ids"],
                batch["teacher_attention_mask"], feats, want_layers
            )
            q = trainable(feats.float()).to(feats.dtype)
            s_logits, s_hidden = run_text_branch(
                model, batch["student_input_ids"],
                batch["student_attention_mask"], q, want_layers
            )
            _, m = compute_losses(
                args, t_logits, s_logits, t_hidden, s_hidden, batch
            )
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + v
            n += 1
        trainable.train()
        return {f"eval/{k}": v / max(1, n) for k, v in acc.items()}

    step = 0
    micro = 0
    t0 = time.time()
    trainable.train()
    done = False
    epoch = 0
    while not done:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_dl:
            if batch is None:
                continue
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if step == 0 and micro == 0:
                img = model.config.image_token_id
                nt = int((batch["teacher_input_ids"] == img).sum())
                ns = int((batch["student_input_ids"] == img).sum())
                rank0(json.dumps({
                    "first_batch": True,
                    "teacher_seq": list(batch["teacher_input_ids"].shape),
                    "student_seq": list(batch["student_input_ids"].shape),
                    "teacher_image_tokens": nt,
                    "student_image_tokens": ns,
                    "tiles": nt // 64,
                    "text_tokens_aligned": int(batch["text_valid_mask"].sum()),
                }))
            feats = tile_features(model, batch["pixel_values"])
            if step == 0 and micro == 0 and not args.no_calibrate_output_scale:
                tgt = trainable.calibrate_output_scale(feats.float())
                if world > 1:
                    # Each rank calibrated on its own batch, so the gains differ.
                    # DDP only syncs gradients, never parameters -- broadcast so
                    # the ranks do not start from divergent weights.
                    dist.broadcast(trainable.out_norm.weight.data, src=0)
                    dist.broadcast(trainable.out_norm.bias.data, src=0)
                rank0(f"calibrated Q-Sampler output RMS to connector RMS {tgt:.4f}")
            with torch.no_grad():
                t_logits, t_hidden = run_text_branch(
                    model, batch["teacher_input_ids"],
                    batch["teacher_attention_mask"], feats, want_layers
                )
            q = sampler(feats.float()).to(feats.dtype)
            s_logits, s_hidden = run_text_branch(
                model, batch["student_input_ids"],
                batch["student_attention_mask"], q, want_layers
            )
            loss, metrics = compute_losses(
                args, t_logits, s_logits, t_hidden, s_hidden, batch
            )
            (loss / args.gradient_accumulation_steps).backward()
            micro += 1
            if micro % args.gradient_accumulation_steps:
                continue

            gnorm = torch.nn.utils.clip_grad_norm_(
                trainable.parameters(), args.max_grad_norm
            )
            if step == 0 and float(gnorm) == 0.0:
                raise RuntimeError(
                    "Q-Sampler received zero gradient on the first optimizer "
                    "step -- the student branch is not wired to the sampler."
                )
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.logging_steps == 0:
                el = time.time() - t0
                rank0(
                    json.dumps({
                        "step": step, "total": total_steps,
                        "loss": round(loss.item(), 4),
                        "grad_norm": round(float(gnorm), 4),
                        **{k: round(v, 4) for k, v in metrics.items()},
                        "lr": round(sched.get_last_lr()[0], 8),
                        "s/step": round(el / step, 3),
                        "eta_h": round((total_steps - step) * el / step / 3600, 2),
                    })
                )
            if args.eval_steps and step % args.eval_steps == 0:
                m = evaluate()
                if m:
                    rank0(json.dumps({"step": step,
                                      **{k: round(v, 4) for k, v in m.items()}}))
            if args.save_steps and step % args.save_steps == 0:
                save(f"checkpoint-{step}")
            if step >= total_steps:
                done = True
                break
        epoch += 1
        if not done:
            save(f"epoch-{epoch}")
        if epoch >= math.ceil(args.num_train_epochs) and not args.max_steps:
            done = True

    m = evaluate()
    if m:
        rank0(json.dumps({"step": step, "final": True,
                          **{k: round(v, 4) for k, v in m.items()}}))
    save("final")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
