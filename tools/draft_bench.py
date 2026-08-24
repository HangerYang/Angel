#!/usr/bin/env python3
"""Pure draft-model speed tester. One command = one self-contained experiment.

Each invocation loads the target and ONE draft checkpoint, and for every sample:
  1. the target runs once, UNTIMED, only to produce realistic draft inputs
     (aux hidden states + the first sampled token)
  2. the draft is rolled forward K steps, TIMED, with no verification and no
     acceptance -- so the number is the draft's own decode speed, independent of
     how often it happens to be right.

The draft's aux layer ids are read from its own checkpoint config, so models with
different aux signatures (e.g. a 9-layer banded_mix vs a 3-layer staged) need no
coordination -- just point at the checkpoint.

Ablations:
  --prefill full   draft prefills the whole prompt   (realistic KV; the default)
           noimg   same, minus image-token positions (HiViS-shaped: hide visual
                   tokens from the drafter; target untouched)
           none    empty KV, roll from the last position only (cheapest; NOTE it
                   understates attention cost and makes --prefill ablation moot)
  --sampler argmax|none|topk|multinomial   what runs between steps; `none` skips
                   token selection to separate sampler cost from transformer cost
  --depth N        truncate the draft to its first N layers
  --steps K        how many tokens to draft

Examples:
  python3 tools/draft_bench.py run --ckpt <ckpt> --dataset MMMU/MMMU -n 10 --steps 32
  python3 tools/draft_bench.py run --ckpt <ckpt> --dataset MMMU/MMMU -n 10 --prefill noimg
  python3 tools/draft_bench.py profile --ckpt <ckpt> --dataset MMMU/MMMU -n 2 --steps 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict

import torch

warnings.filterwarnings("ignore")

DEFAULT_TARGET = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEV, DT = "cuda:0", torch.bfloat16

# dataset id -> (load_dataset args, image field, question field, placeholder)
DATASETS = {
    "MMMU/MMMU":            (dict(name="History", split="test"), "image_1", "question", "<image 1>"),
    "Lin-Chen/MMStar":      (dict(split="val"),                  "image",   "question", "<image>"),
    "lmms-lab/textvqa":     (dict(split="validation"),           "image",   "question", None),
    "lmms-lab/ChartQA":     (dict(split="test"),                 "image",   "question", None),
    "lmms-lab/COCO-Caption": (dict(split="val"),                 "image",   None,       None),
}


def load_samples(dataset: str, n: int):
    from datasets import load_dataset
    spec = DATASETS.get(dataset)
    if spec is None:
        raise SystemExit(f"unknown dataset {dataset!r}; known: {list(DATASETS)}")
    kw, img_f, q_f, ph = spec
    name = kw.pop("name", None)
    ds = (load_dataset(dataset, name, trust_remote_code=True, **kw) if name
          else load_dataset(dataset, trust_remote_code=True, **kw))
    out = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        q = str(it[q_f]) if q_f else "Describe the image."
        if ph:
            q = q.replace(ph, "").strip()
        out.append((it[img_f].convert("RGB"), q))
    return out


def cuda_time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def load_draft(ckpt, depth):
    from angelslim.compressor.speculative.train.models.draft.llama_eagle3 import (
        Eagle3LlamaForCausalLM)
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    aux = cfg.get("aux_hidden_states_layer_ids") or [1, 14, 26]
    m = Eagle3LlamaForCausalLM.from_pretrained(ckpt, dtype=DT).to(DEV).eval()
    # rotary cos/sin are non-persistent buffers; from_pretrained's meta init leaves
    # them uninitialised (NaN). Rebuild before use.
    for l in m.layers:
        r = l.self_attn.rotary_emb
        r._set_cos_sin_cache(r.max_position_embeddings, DEV, DT)
    m._early_exit_threshold = -1.0
    if depth is not None and depth < len(m.layers):
        n_full = len(m.layers)
        m.layers = m.layers[:depth]
        if getattr(m, "progressive_staged", False):
            _fb = m.take_progressive_draft_feedback
            def fb(*a, **k):
                m._last_layer_outs = [m._last_layer_outs[0]] * n_full
                return _fb(*a, **k)
            m.take_progressive_draft_feedback = fb
    return m, aux, cfg


@torch.no_grad()
def target_inputs(model, proc, image, question, aux_ids, img_tok):
    """UNTIMED. Returns (aux_hs [1,S,nH], draft_ids [1,S], img_mask [S], first_tok)."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=prompt, images=[image], return_tensors="pt").to(DEV)
    out = model(**inp, output_hidden_states=True)
    hs = out.hidden_states                                   # len = n_layers + 1
    aux = torch.cat([hs[j + 1] for j in aux_ids], dim=-1).to(DT)
    first = int(out.logits[0, -1].argmax())
    ids = inp["input_ids"]
    draft_ids = torch.cat([ids[:, 1:], torch.tensor([[first]], device=DEV)], dim=1)
    mask = (ids == img_tok)[0] if img_tok is not None else torch.zeros(ids.shape[1], dtype=torch.bool, device=DEV)
    return aux, draft_ids, mask, first


@torch.no_grad()
def make_roll(m, aux, ids, mask, first, steps, prefill, sampler):
    progressive = bool(getattr(m, "progressive_staged", False))
    if prefill == "none":
        a0, i0, kv = aux[:, -1:], None, 0
    else:
        keep = torch.ones(ids.shape[1], dtype=torch.bool, device=DEV)
        if prefill == "noimg":
            keep = ~mask
            keep[-1] = True                     # keep the final position
        a0, i0, kv = aux[:, keep], ids[:, keep], int(keep.sum())

    def roll():
        if prefill == "none":
            h = m.combine_hidden_states(a0)
            tok = torch.tensor([[first]], device=DEV)
            start = 0
        else:
            h = m.combine_hidden_states(a0)
            c0 = m.init_cache_hidden()
            pos0 = torch.arange(kv, device=DEV).unsqueeze(0)
            msk0 = torch.full((1, 1, kv, kv), float("-inf"), device=DEV, dtype=h.dtype).triu(1)
            h, c0 = m.encode_layers(m.embed_tokens(i0).to(h.dtype), h, c0, msk0, pos0, True)
            h = h[:, -1:]
            t = m.compute_logits(h).argmax(-1)
            tok = t + m.d2t[t]
            h = (m.take_progressive_draft_feedback() if progressive
                 else m.next_hidden_from_encode(h))
            if h.shape[1] > 1:
                h = h[:, -1:]
            start = kv
        cache = m.init_cache_hidden()
        for i in range(steps):
            pos = torch.full((1, 1), start + i, dtype=torch.long, device=DEV)
            msk = torch.zeros(1, 1, 1, 1, device=DEV, dtype=h.dtype)
            h, cache = m.encode_layers(m.embed_tokens(tok).to(h.dtype), h, cache, msk, pos, True)
            if sampler != "none":
                lg = m.compute_logits(h)
                if sampler == "argmax":
                    nt = lg.argmax(-1)
                elif sampler == "topk":
                    nt = lg.topk(8, -1).indices[..., :1].squeeze(-1)
                else:
                    nt = torch.multinomial(lg.float().softmax(-1)[0, -1], 1).view(1, 1)
                tok = nt + m.d2t[nt]
            h = (m.take_progressive_draft_feedback() if progressive
                 else m.next_hidden_from_encode(h))
            if h.shape[1] > 1:
                h = h[:, -1:]
        return tok

    return roll, kv


def setup(args):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    m, aux_ids, cfg = load_draft(args.ckpt, args.depth)
    proc = AutoProcessor.from_pretrained(args.target)
    tgt = AutoModelForImageTextToText.from_pretrained(
        args.target, dtype=DT, device_map=DEV).eval()
    img_tok = getattr(tgt.config, "image_token_id", None)
    samples = load_samples(args.dataset, args.n)
    print(f"ckpt    : {args.ckpt}")
    print(f"draft   : mode={cfg.get('eagle_aux_injection_mode')} layers={len(m.layers)} "
          f"aux={aux_ids} (n={len(aux_ids)})")
    print(f"data    : {args.dataset}  n={len(samples)}  image_token_id={img_tok}")
    print(f"ablation: prefill={args.prefill} sampler={args.sampler} steps={args.steps}")
    return m, aux_ids, proc, tgt, img_tok, samples


def cmd_run(args):
    m, aux_ids, proc, tgt, img_tok, samples = setup(args)
    per, kvs, imgfrac, plens, nimg = [], [], [], [], []
    for k, (img, q) in enumerate(samples):
        aux, ids, mask, first = target_inputs(tgt, proc, img, q, aux_ids, img_tok)
        imgfrac.append(float(mask.float().mean()))
        plens.append(int(ids.shape[1])); nimg.append(int(mask.sum()))
        roll, kv = make_roll(m, aux, ids, mask, first, args.steps, args.prefill, args.sampler)
        ms = cuda_time(roll, args.iters, args.warmup)
        per.append(ms); kvs.append(kv)
        print(f"  [{k}] S={ids.shape[1]:5d} kv={kv:5d} img={100*imgfrac[-1]:4.1f}%  "
              f"{ms:8.3f} ms/roll  {ms/args.steps:7.4f} ms/tok  {1000*args.steps/ms:8.1f} tok/s")
    mean = sum(per) / len(per)
    res = dict(ckpt=args.ckpt, dataset=args.dataset, n=len(samples),
               mean_prompt_tokens=sum(plens)/len(plens),
               mean_image_tokens=sum(nimg)/len(nimg),
               mean_text_tokens=(sum(plens)-sum(nimg))/len(plens),
               draft_output_tokens=args.steps,
               layers=len(m.layers), aux_layer_ids=aux_ids,
               steps=args.steps, prefill=args.prefill, sampler=args.sampler,
               depth=args.depth, mean_kv=sum(kvs)/len(kvs),
               mean_image_frac=sum(imgfrac)/len(imgfrac),
               ms_per_roll=mean, ms_per_token=mean/args.steps,
               tok_s=1000*args.steps/mean, per_sample_ms=[round(p, 4) for p in per])
    print(f"\nprompt: {res['mean_prompt_tokens']:.0f} tok "
          f"({res['mean_image_tokens']:.0f} image = {100*res['mean_image_frac']:.1f}%, "
          f"{res['mean_text_tokens']:.0f} text)   draft generates {args.steps} tok")
    print(f"MEAN  kv={res['mean_kv']:.0f}  {mean:.3f} ms/roll  "
          f"{res['ms_per_token']:.4f} ms/tok  {res['tok_s']:.1f} tok/s")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(res, open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")


def cmd_profile(args):
    m, aux_ids, proc, tgt, img_tok, samples = setup(args)
    img, q = samples[0]
    aux, ids, mask, first = target_inputs(tgt, proc, img, q, aux_ids, img_tok)
    roll, kv = make_roll(m, aux, ids, mask, first, args.steps, args.prefill, args.sampler)
    tot, cnt, handles = defaultdict(float), defaultdict(int), []

    def mk(name, mod):
        st = {}
        def pre(_m, _i):
            torch.cuda.synchronize(); st["t"] = time.perf_counter()
        def post(_m, _i, _o):
            torch.cuda.synchronize()
            tot[name] += (time.perf_counter() - st["t"]) * 1000.0; cnt[name] += 1
        handles += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]

    for name, mod in m.named_modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)) or mod.__class__.__name__ in (
                "LlamaRMSNorm", "LlamaAttention", "LlamaMLP", "LlamaDecoderLayeremb", "EarlyExitBridge"):
            mk(name or mod.__class__.__name__, mod)

    for _ in range(args.warmup):
        roll()
    tot.clear(); cnt.clear()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(args.iters):
        roll()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000.0 / args.iters
    for h in handles:
        h.remove()
    print(f"\nwall = {wall:.3f} ms/roll ({wall/args.steps:.4f} ms/tok, {1000*args.steps/wall:.1f} tok/s), kv={kv}")
    print("\nNOTE: hooks CUDA-synchronize around every module, so summed time far exceeds\n"
          "the unhooked wall time. Read SHARE (relative cost), not absolute ms.\n")
    rows = sorted(tot.items(), key=lambda kv_: -kv_[1]); ssum = sum(tot.values()) or 1.0
    print(f"{'module':<50} {'calls':>6} {'ms':>9} {'share':>7}")
    print("-" * 76)
    for nm, ms in rows[: args.top]:
        print(f"{nm[:50]:<50} {cnt[nm]/args.iters:>6.0f} {ms/args.iters:>9.3f} {100*ms/ssum:>6.1f}%")
    if args.json:
        json.dump(dict(ckpt=args.ckpt, wall_ms=wall, kv=kv, steps=args.steps,
                       modules={k: dict(ms=v/args.iters, calls=cnt[k]/args.iters,
                                        share=100*v/ssum) for k, v in rows}),
                  open(args.json, "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("run", cmd_run), ("profile", cmd_profile)):
        b = sub.add_parser(nm)
        b.add_argument("--ckpt", required=True)
        b.add_argument("--target", default=DEFAULT_TARGET)
        b.add_argument("--dataset", default="MMMU/MMMU")
        b.add_argument("-n", "--n", type=int, default=10)
        b.add_argument("--steps", type=int, default=32)
        b.add_argument("--prefill", choices=["full", "noimg", "none"], default="full")
        b.add_argument("--sampler", choices=["argmax", "none", "topk", "multinomial"], default="argmax")
        b.add_argument("--depth", type=int, default=None)
        b.add_argument("--iters", type=int, default=10)
        b.add_argument("--warmup", type=int, default=3)
        b.add_argument("--json", default=None)
        if nm == "profile":
            b.add_argument("--top", type=int, default=25)
        b.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
