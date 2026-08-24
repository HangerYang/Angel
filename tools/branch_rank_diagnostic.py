"""Where do wrong draft top-1 tokens sit in the teacher's ranking?

For every held-out position where the draft's top-1 differs from the teacher's
top-1, bucket by the teacher rank of the draft token and report, per bucket:
share of wrong tokens, p_T(draft), teacher centered-logit delta RMS, the
branch-change MSE the delta objective would score, and how that MSE splits into
direction (cosine) and scale (||delta_D|| / ||delta_T||).

Runs one draft checkpoint at a time against the same eval jsonl the trainer
uses. Mirrors TTT step 0 -> step 1 exactly (see eagle3_trainer.py:723 and
_branch_decide at eagle3_trainer.py:530); the branch teacher forward reuses the
trainer's own branch_teacher_logits.

  python tools/branch_rank_diagnostic.py \
    --draft_model_config_path <cfg.json> --draft_ckpt <checkpoint-66466> \
    --vocab_cache <run>/vocab_mapping_cache.pt --num_samples 64 --out out.json
"""

import argparse
import json
import os
import sys

import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from angelslim.compressor.speculative import (  # noqa: E402
    DatasetManager,
    DraftModelConfig,
    Eagle3TrainerFactory,
    create_draft_model,
    create_target_model,
)
from angelslim.compressor.speculative.train.trainer.eagle3_trainer import (  # noqa: E402
    padding,
)

BUCKETS = [("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5", 5, 5),
           ("6-10", 6, 10), (">10", 11, 10**9)]


class DataArgs:
    """Stand-in for the argparse namespace DatasetManager reads."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def build(args):
    cfg = DraftModelConfig.from_file(args.draft_model_config_path)
    # Trainer-side only: forces _branch_ctx to be stashed and makes
    # _branch_decide's candidate set "wrong and inside teacher top-k".
    cfg.branch_distill_loss_weight = 0.1
    cfg.branch_distill_objective = "change"
    cfg.branch_distill_prob_ratio_threshold = 0.0
    cfg.branch_distill_target_top_k = args.verify_top_k
    target_model_type = getattr(cfg, "target_model_type", None)

    target_model = create_target_model(
        backend="hf",
        model_path=args.target_model_name_or_path,
        modal_type="VLM",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        target_model_type=target_model_type,
    )
    draft_model = create_draft_model(cfg)
    draft_model.load_embed_weights(
        args.target_model_name_or_path, args.embed_weight_key
    )
    draft_model.freeze_embed_weights()
    # Same warm-start path as training (tools/train_eagle3_online.py:475): build
    # the model first, then copy weights in, so rope caches stay real.
    warm = draft_model.__class__.from_pretrained(
        args.draft_ckpt, config=cfg, torch_dtype=torch.bfloat16
    )
    missing, unexpected = draft_model.load_state_dict(warm.state_dict(), strict=False)
    del warm
    print(f"loaded {args.draft_ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    if any("banded" in k or "fc" in k for k in missing):
        raise RuntimeError(f"checkpoint did not cover fusion weights: {missing[:10]}")

    data_args = DataArgs(
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        sample_num=args.num_samples,
        num_proc=1,
        load_from_cache_file=True,
        modal_type="VLM",
        training_mode="online",
        display=False,
        shuffle_seed=42,
        target_model_name_or_path=args.target_model_name_or_path,
        train_hidden_path=None,
        eval_hidden_path=None,
        output_dir=args.scratch_dir,
    )
    dm = DatasetManager(
        data_args=data_args,
        tokenizer=target_model.tokenizer,
        model_max_length=args.model_max_length,
        chat_template_type=args.chat_template_type,
        display=False,
        target_model_type=target_model_type,
    )
    train_dataset, eval_dataset, collator = dm.create_online_datasets()
    draft_model.build_vocab_mapping(dataset=train_dataset, cache_path=args.vocab_cache)

    targs = transformers.TrainingArguments(
        output_dir=args.scratch_dir,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Eagle3TrainerFactory.create(
        training_mode="online",
        modal_type="VLM",
        draft_model=draft_model,
        target_model=target_model,
        length=7,
        draft_model_config=cfg,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    return trainer, eval_dataset


def rollout(trainer, inputs):
    """Draft step 0 + step 1 on one batch, returning everything the buckets need."""
    draft = trainer.draft_model
    data = trainer.prepare_data_for_draft_model(inputs)
    hidden = trainer.down_project_hidden_states(
        data["hidden_states"], data.get("gist_embeddings")
    )
    attn, pos = trainer.prepare_attention_mask_and_position_ids(
        hidden, data["attention_mask"], data["position_ids"]
    )
    cache = draft.init_cache_hidden() if hasattr(draft, "init_cache_hidden") else [[], []]
    ids0, tl0, lm0 = data["input_ids"], data["target_logits"], data["loss_mask"]

    h0, cache = draft.encode_layers(
        inputs_embeds=draft.embed_input_ids(ids0),
        hidden_states=hidden,
        cache_hidden=cache,
        attention_mask=attn,
        position_ids=pos,
        use_cache=True,
        gist_embeddings=None,
    )
    logits0 = draft.compute_logits(h0)

    ids1 = padding(ids0, left=False)
    pre_step_cache = trainer._fork_cache(cache)
    h1, _ = draft.encode_layers(
        inputs_embeds=draft.embed_input_ids(ids1),
        hidden_states=h0,
        cache_hidden=cache,
        attention_mask=attn,
        position_ids=pos,
        use_cache=True,
        gist_embeddings=None,
    )
    base_logits = draft.compute_logits(h1)
    return dict(
        logits0=logits0, target_logits=tl0, loss_mask=lm0, ids0=ids0, ids1=ids1,
        h0=h0, cache=pre_step_cache, attn=attn, pos=pos, base_logits=base_logits,
    )


def branch_pass(trainer, st, sub_mask, draft_top1):
    """One teacher re-score + one forked draft step for the given branch mask.

    Same construction as _branch_decide/_branch_loss, but keeping the
    per-position delta and MSE instead of the masked mean.
    """
    draft = trainer.draft_model
    offset = 2  # idx 0 -> absolute position j + 2, eagle3_trainer.py:571
    orig_ids = trainer._branch_ctx["input_ids"]
    sub_ids = orig_ids.clone()
    rows, at = torch.nonzero(sub_mask, as_tuple=True)
    sub_ids[rows, at + offset] = draft_top1[rows, at].to(sub_ids.dtype)

    branch_teacher = trainer._shift_left(
        trainer.branch_teacher_logits(sub_ids).detach(), offset
    ).to(st["target_logits"].device)
    real_next = trainer._shift_left(st["target_logits"], 1).to(branch_teacher.device)
    b_in = branch_teacher[..., draft.t2d].float()
    r_in = real_next[..., draft.t2d].float()
    target_delta = (b_in - b_in.mean(-1, keepdim=True)) - (
        r_in - r_in.mean(-1, keepdim=True)
    )

    branch_ids = torch.where(
        sub_mask, draft_top1.to(st["ids1"].dtype), st["ids1"]
    )
    hb, _ = draft.encode_layers(
        inputs_embeds=draft.embed_input_ids(branch_ids),
        hidden_states=st["h0"],
        cache_hidden=trainer._fork_cache(st["cache"]),
        attention_mask=st["attn"],
        position_ids=st["pos"],
        use_cache=True,
        gist_embeddings=None,
    )
    bl = draft.compute_logits(hb).float()
    base = st["base_logits"].float()
    draft_delta = (bl - bl.mean(-1, keepdim=True)) - (base - base.mean(-1, keepdim=True))
    # Direction and scale of the draft's change, separately from the MSE: a
    # draft can point the right way and still over- or under-shoot.
    dot = (draft_delta * target_delta).sum(-1)
    nd = draft_delta.pow(2).sum(-1).sqrt()
    nt = target_delta.pow(2).sum(-1).sqrt()
    return dict(
        dsq=target_delta.pow(2).mean(-1),                    # per-position E[dT^2]
        mse=(draft_delta - target_delta).pow(2).mean(-1),    # per-position MSE
        cos=dot / (nd * nt).clamp_min(1e-6),
        rnorm=nd / nt.clamp_min(1e-6),
        target_delta=target_delta,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft_model_config_path", required=True)
    p.add_argument("--draft_ckpt", required=True)
    p.add_argument("--vocab_cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--groups", type=int, default=4,
                   help="teacher re-score passes per batch; keeps the fraction of "
                        "substituted tokens near the training branch rate")
    p.add_argument("--verify_top_k", type=int, default=2048)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--target_model_name_or_path",
                   default="HuggingFaceTB/SmolVLM-256M-Instruct")
    p.add_argument("--embed_weight_key",
                   default="model.text_model.embed_tokens.weight")
    p.add_argument("--train_data_path", nargs="+",
                   default=["dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl"])
    p.add_argument("--eval_data_path",
                   default="dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl")
    p.add_argument("--chat_template_type", default="smolvlm")
    p.add_argument("--model_max_length", type=int, default=4096)
    p.add_argument("--scratch_dir", default=os.path.expanduser("~/tmp/branch_diag"))
    args = p.parse_args()
    os.makedirs(args.scratch_dir, exist_ok=True)

    trainer, eval_dataset = build(args)
    draft = trainer.draft_model
    draft.eval()
    device = next(draft.parameters()).device
    loader = trainer.get_eval_dataloader()
    gen = torch.Generator(device="cpu").manual_seed(0)

    stats = {
        name: dict(n=0, p=0.0, drms=0.0, mse=0.0, cos=0.0, rnorm=0.0)
        for name, _, _ in BUCKETS
    }
    n_wrong = n_positions = 0
    ce_sum = ce_positions = 0.0
    n_batches = 0
    verified = not args.verify

    with torch.no_grad():
        for inputs in loader:
            if n_batches >= args.num_samples:
                break
            inputs = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in inputs.items()
            }
            st = rollout(trainer, inputs)
            tl = st["target_logits"]
            draft_top1_d = st["logits0"].argmax(-1)
            draft_top1 = draft_top1_d + draft.d2t[draft_top1_d]
            teacher_top1 = tl.argmax(-1)
            draft_logit = tl.gather(-1, draft_top1[..., None].to(tl.device)).squeeze(-1)
            rank = 1 + (tl > draft_logit[..., None]).sum(-1)
            p_t = tl.float().softmax(-1).gather(
                -1, draft_top1[..., None].to(tl.device)
            ).squeeze(-1)

            # Main-head CE on the same frame, over the positions the trainer
            # counts as active (teacher top-1 inside the draft vocab), so this
            # is directly comparable to train/ploss_active_0.
            head = tl[..., draft.t2d].float()
            pmask = draft.t2d[teacher_top1][..., None].int() * st["loss_mask"]
            ce_sum += float(
                (
                    pmask
                    * head.softmax(-1)
                    * st["logits0"].float().log_softmax(-1)
                ).sum(-1).sum().neg()
            )
            ce_positions += float(pmask.sum())

            cols = torch.arange(tl.shape[1], device=tl.device)
            in_range = (cols + 2 < trainer._branch_ctx["input_ids"].shape[1])[None, :]
            valid = st["loss_mask"][..., 0].bool() & in_range
            wrong = valid & (draft_top1 != teacher_top1)
            n_positions += int(valid.sum())
            n_wrong += int(wrong.sum())
            if not bool(wrong.any()):
                n_batches += 1
                continue

            if not verified:
                # The mask and per-position delta must match what the trainer
                # itself would build for this frame.
                pending = trainer._branch_decide(
                    0, draft, st["logits0"], tl, st["loss_mask"]
                )
                want = wrong & (rank <= args.verify_top_k)
                assert torch.equal(pending["mask"], want), "branch mask mismatch"
                td = branch_pass(trainer, st, want, draft_top1)["target_delta"]
                assert torch.allclose(
                    td[want], pending["target_delta"][want], atol=1e-3
                ), "target delta mismatch vs trainer"
                print("verified: mask and target delta match the trainer")
                verified = True

            # Split the wrong positions across passes so each teacher re-score
            # substitutes only a training-like fraction of the sequence.
            assign = torch.randint(
                0, args.groups, wrong.shape, generator=gen
            ).to(wrong.device)
            for g in range(args.groups):
                gmask = wrong & (assign == g)
                if not bool(gmask.any()):
                    continue
                bp = branch_pass(trainer, st, gmask, draft_top1)
                dsq, mse = bp["dsq"], bp["mse"]
                cos, rnorm = bp["cos"], bp["rnorm"]
                r = rank[gmask]
                for name, lo, hi in BUCKETS:
                    sel = (r >= lo) & (r <= hi)
                    if not bool(sel.any()):
                        continue
                    s = stats[name]
                    s["n"] += int(sel.sum())
                    s["p"] += float(p_t[gmask][sel].sum())
                    s["drms"] += float(dsq[gmask][sel].clamp_min(0).sqrt().sum())
                    s["mse"] += float(mse[gmask][sel].sum())
                    s["cos"] += float(cos[gmask][sel].sum())
                    s["rnorm"] += float(rnorm[gmask][sel].sum())
            n_batches += 1
            if n_batches % 8 == 0:
                print(f"  {n_batches}/{args.num_samples} samples", flush=True)

    total = sum(s["n"] for s in stats.values())
    out = dict(
        tag=args.tag or os.path.basename(os.path.dirname(args.draft_ckpt)),
        ckpt=args.draft_ckpt, samples=n_batches, groups=args.groups,
        positions=n_positions, wrong=n_wrong,
        main_ce=ce_sum / max(ce_positions, 1.0),
        wrong_rate=(n_wrong / max(n_positions, 1)), buckets={},
    )
    for name, _, _ in BUCKETS:
        s = stats[name]
        n = max(s["n"], 1)
        out["buckets"][name] = dict(
            n=s["n"], share=s["n"] / max(total, 1),
            p_teacher=s["p"] / n, delta_rms=s["drms"] / n, change_mse=s["mse"] / n,
            cos=s["cos"] / n, norm_ratio=s["rnorm"] / n,
        )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
