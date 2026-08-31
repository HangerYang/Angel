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


# Lengthens a raw VQA-style question so there's enough decode length for a
# speculative-decoding comparison to show anything -- same template as
# tools/vllm_offline_eagle3_vlm_batch.py's answer_then_describe.
ANSWER_THEN_DESCRIBE = (
    "Answer this question: {q} Then describe the image in detail to justify your answer."
)


def load_samples(dataset: str, n: int, prompt_style: str = "raw"):
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
        if prompt_style == "answer_then_describe" and q_f:
            q = ANSWER_THEN_DESCRIBE.format(q=q)
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


def fix_dtype_skipped_params(model, ckpt, dtype):
    """from_pretrained(dtype=...) can silently miss a parameter that was
    registered as an already-materialized (non-meta) tensor during __init__
    instead of lazily -- e.g. llama_eagle3.py's bandN_mix_logits hardcodes
    `torch.zeros(len(band), dtype=torch.float32)` at construction time, so it
    never picks up the requested dtype even though the checkpoint's own
    saved copy is correctly bf16. Symptom: "expected mat1 and mat2 to have
    the same dtype" inside combine_hidden_states's banded_mix_fc path.
    Re-copy any parameter still at the wrong dtype directly from the
    checkpoint's safetensors file.
    """
    path = os.path.join(ckpt, "model.safetensors")
    if not os.path.isfile(path):
        return
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        ckpt_keys = set(f.keys())
        for name, param in model.named_parameters():
            if param.dtype == dtype or name not in ckpt_keys:
                continue
            # .copy_() would cast the source INTO the destination's existing
            # (wrong) dtype, leaving the mismatch in place -- .data= actually
            # swaps the storage's dtype while keeping the same Parameter object.
            with torch.no_grad():
                param.data = f.get_tensor(name).to(device=param.device, dtype=dtype)


def _ckpt_has_key(ckpt, key):
    """Whether `key` is one of the checkpoint's OWN saved tensors (vs. left at
    from_pretrained's random init) -- some published checkpoints (e.g.
    AngelSlim/Qwen3-VL-4B-Instruct_eagle3) omit embed_tokens.weight on purpose,
    meaning to have it copied from the target instead of trained."""
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx):
        return key in json.load(open(idx))["weight_map"]
    st = os.path.join(ckpt, "model.safetensors")
    if os.path.exists(st):
        from safetensors import safe_open
        with safe_open(st, framework="pt") as f:
            return key in f.keys()
    return True  # unknown layout; don't false-positive into an embed overwrite


def _default_aux_layer_ids(target_model_name_or_path):
    """Mirror target_model_wrapper.py's _get_default_aux_layer_ids: early/mid/
    late decoder layers, used whenever a checkpoint's config doesn't pin
    aux_hidden_states_layer_ids explicitly (true of every currently-published
    Qwen3-VL eagle3 config)."""
    from transformers import AutoConfig
    tcfg = AutoConfig.from_pretrained(target_model_name_or_path)
    tcfg = getattr(tcfg, "text_config", tcfg)
    n = tcfg.num_hidden_layers
    return [1, n // 2 - 1, n - 4]


def load_draft(ckpt, depth, target_model_name_or_path=None):
    from angelslim.compressor.speculative.train.models.draft.llama_eagle3 import (
        Eagle3LlamaForCausalLM)
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    aux = cfg.get("aux_hidden_states_layer_ids")
    if not aux:
        aux = (_default_aux_layer_ids(target_model_name_or_path) if target_model_name_or_path
               else [1, 14, 26])
    m = Eagle3LlamaForCausalLM.from_pretrained(ckpt, dtype=DT).to(DEV).eval()
    fix_dtype_skipped_params(m, ckpt, DT)
    if not _ckpt_has_key(ckpt, "embed_tokens.weight"):
        # See _ckpt_has_key: copy the token embedding from the TARGET model
        # instead of leaving it at from_pretrained's random init, which would
        # make every draft prediction noise regardless of aux layers/rope --
        # confirmed empirically on AngelSlim/Qwen3-VL-4B-Instruct_eagle3.
        from angelslim.compressor.speculative.train.models.model_utils import MODEL_TYPE_PARAM_MAP
        target_type = cfg.get("target_model_type")
        entry = MODEL_TYPE_PARAM_MAP.get(target_type)
        if entry is None or target_model_name_or_path is None:
            raise RuntimeError(
                f"{ckpt} has no embed_tokens.weight and cannot source one: "
                f"target_model_type={target_type!r} in map={entry is not None}, "
                f"target_model_name_or_path={target_model_name_or_path!r}")
        _, embed_weight_key, _ = entry
        print(f"Loading draft embed_tokens from {target_model_name_or_path}:{embed_weight_key}")
        m.load_embed_weights(target_model_name_or_path, embed_weight_key)
        m.to(device=DEV, dtype=DT)
    # rotary cos/sin are non-persistent buffers; from_pretrained's meta init leaves
    # them uninitialised (NaN). Rebuild before use. MRotaryEmbedding (Qwen-VL-family
    # mrope drafts) has no such cache -- it recomputes cos/sin from inv_freq fresh on
    # every forward call, so there's nothing to rebuild there.
    for l in m.layers:
        r = l.self_attn.rotary_emb
        if hasattr(r, "_set_cos_sin_cache"):
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
def eagle3_target_prefill(model, proc, image, question, aux_ids, img_tok=None):
    """Like target_inputs, but WITH use_cache=True -- for round-based (real
    multi-round) generation, where the target's KV cache must persist and
    grow incrementally (via compact_cache) across rounds instead of being
    rebuilt from scratch. Returns (aux, draft_ids, first, prefix_cache,
    prev_len, mask): same "shift by one" aux/draft_ids pairing as
    target_inputs (aux[i] pairs with the REAL token at prompt position i+1,
    matching EAGLE's hidden[t]-predicts-token[t+1] convention), `prefix_cache`
    is a DynamicCache the caller only ever grows via compact_cache, `prev_len`
    is its current length (the real prompt's token count). `mask` is the
    image-token boolean mask on the UNSHIFTED prompt ids (None if `img_tok`
    is None) -- same one-position-off-from-draft_ids approximation make_roll's
    --prefill noimg already uses, since the image span is far from either
    boundary. `mrope` is (prefix_pos, rope_delta) from qwen_mrope_prefix, or
    None for non-mrope targets -- see that function's docstring for why this
    is required (not merely nicer) for Qwen-VL-family targets.
    """
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=prompt, images=[image], return_tensors="pt").to(DEV)
    mrope = qwen_mrope_prefix(model, inp)
    out = model(**inp, position_ids=(mrope[0] if mrope is not None else None),
                output_hidden_states=True, use_cache=True)
    hs = out.hidden_states
    aux = torch.cat([hs[j + 1] for j in aux_ids], dim=-1).to(DT)
    first = int(out.logits[0, -1].argmax())
    ids = inp["input_ids"]
    draft_ids = torch.cat([ids[:, 1:], torch.tensor([[first]], device=DEV)], dim=1)
    mask = (ids == img_tok)[0] if img_tok is not None else None
    return aux, draft_ids, first, out.past_key_values, ids.shape[1], mask, mrope


def qwen_mrope_prefix(model, inp):
    """Real (T,H,W) mrope prefix positions for a Qwen-VL-family target, via
    its own get_rope_index -- None for non-mrope targets (no image_grid_thw
    in the processor output, e.g. SmolVLM/Idefics3). Manually calling a
    Qwen-VL target outside its own generate() loop with position_ids=None is
    NOT safe -- confirmed by reproduction: it corrupts the attention output
    shape on the second (incremental-decode) forward. Explicit real mrope
    positions sidestep this and are the only path validated here.
    """
    image_grid_thw = inp.get("image_grid_thw")
    if image_grid_thw is None:
        return None
    inner = getattr(model, "model", model)
    fn = getattr(inner, "get_rope_index", None)
    if fn is None:
        return None
    kw = dict(image_grid_thw=image_grid_thw, video_grid_thw=inp.get("video_grid_thw"),
              attention_mask=inp.get("attention_mask"))
    try:
        # transformers>=5: mm_token_type_ids is a required positional arg.
        pos, delta = fn(inp["input_ids"], inp["mm_token_type_ids"], **kw)
    except (TypeError, KeyError):
        pos, delta = fn(inp["input_ids"], **kw)
    return pos, delta


def mrope_positions_for_len(prefix_pos, rope_delta, seq_len, device):
    """Extend real [3,1,prefix_len] mrope prefix positions to a longer
    image-inclusive draft/target sequence: prompt rows copied verbatim, every
    row past the prefix continues Qwen's own post-vision-span scalar
    convention (index + mrope_position_delta, same value on T/H/W)."""
    prefix_len = prefix_pos.shape[-1]
    if seq_len <= prefix_len:
        return prefix_pos[:, :, :seq_len].to(device)
    extra_n = seq_len - prefix_len
    idx = torch.arange(prefix_len, prefix_len + extra_n, device=device, dtype=prefix_pos.dtype).view(1, 1, -1)
    delta = rope_delta.reshape(1, 1, 1).to(device=device, dtype=prefix_pos.dtype)
    extra = (idx + delta).expand(3, prefix_pos.shape[1], -1)
    return torch.cat([prefix_pos.to(device), extra], dim=-1)


def _pos_at(step_index, mrope, device):
    """Position id for ONE new row at absolute index `step_index`. Every
    caller in make_eagle3_tree_roll uses this only for rows PAST the initial
    prefix (newly drafted/accepted tokens are always plain text), so the
    scalar continuation formula applies unconditionally -- never the real
    per-row prefix positions (those come from mrope_positions_for_len /
    `mrope[0]` directly, only for the very first prefill row-block)."""
    if mrope is None:
        return torch.full((1, 1), step_index, dtype=torch.long, device=device)
    prefix_pos, rope_delta = mrope
    val = step_index + int(rope_delta.reshape(-1)[0].item())
    return torch.full((3, 1, 1), val, dtype=prefix_pos.dtype, device=device)


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


def clone_cache_hidden(cache):
    """Deep-copy the container, sharing the underlying k/v tensor objects.

    encode_layers appends new k/v tensors per call rather than mutating
    existing ones in place, so two branches can safely share references to
    the tensors already in the cache -- only the list *structure* needs to
    fork so appends on one branch don't show up on a sibling.
    """
    return [[list(ks), list(vs)] for ks, vs in cache]


@torch.no_grad()
def make_tree_roll(m, aux, ids, mask, first, prefill, sampler, tree_depth, top_k, total_token):
    """Beam-style speculative tree: like HiViS's own topK_genrate, but pruned
    by cumulative log-prob to a `total_token` budget instead of a fixed
    hand-tuned sparse topology (mc_sim_7b_63) -- branching factor `top_k` per
    step, degenerates to make_roll's linear chain when top_k=1.

    Draft-only, untimed verification (matches make_roll's scope) -- this
    measures how long the draft itself takes to PROPOSE a tree, not a full
    verify+accept round trip against the target.
    """
    progressive = bool(getattr(m, "progressive_staged", False))
    if prefill == "none":
        a0, i0, kv = aux[:, -1:], None, 0
    else:
        keep = torch.ones(ids.shape[1], dtype=torch.bool, device=DEV)
        if prefill == "noimg":
            keep = ~mask
            keep[-1] = True
        a0, i0, kv = aux[:, keep], ids[:, keep], int(keep.sum())

    def sample_children(h):
        lg = m.compute_logits(h)
        logp = torch.log_softmax(lg.float(), dim=-1)[0, 0]
        k = min(top_k, logp.numel())
        vals, idx = logp.topk(k)
        return vals, idx

    def roll():
        if prefill == "none":
            h = m.combine_hidden_states(a0)
            tok0, start = torch.tensor([[first]], device=DEV), 0
            cache0 = m.init_cache_hidden()
        else:
            h = m.combine_hidden_states(a0)
            c0 = m.init_cache_hidden()
            pos0 = torch.arange(kv, device=DEV).unsqueeze(0)
            msk0 = torch.full((1, 1, kv, kv), float("-inf"), device=DEV, dtype=h.dtype).triu(1)
            h, c0 = m.encode_layers(m.embed_tokens(i0).to(h.dtype), h, c0, msk0, pos0, True)
            h = h[:, -1:]
            t = m.compute_logits(h).argmax(-1)
            tok0 = t + m.d2t[t]
            h = (m.take_progressive_draft_feedback() if progressive
                 else m.next_hidden_from_encode(h))
            if h.shape[1] > 1:
                h = h[:, -1:]
            cache0, start = c0, kv

        frontier = [{"cache": cache0, "h": h, "tok": tok0, "score": 0.0, "len": 0}]
        for d in range(tree_depth):
            expanded = []
            for node in frontier:
                pos = torch.full((1, 1), start + d, dtype=torch.long, device=DEV)
                msk = torch.zeros(1, 1, 1, 1, device=DEV, dtype=node["h"].dtype)
                h_new, cache_new = m.encode_layers(
                    m.embed_tokens(node["tok"]).to(node["h"].dtype),
                    node["h"], node["cache"], msk, pos, True)
                vals, idx = sample_children(h_new)
                h_next = (m.take_progressive_draft_feedback() if progressive
                          else m.next_hidden_from_encode(h_new))
                if h_next.shape[1] > 1:
                    h_next = h_next[:, -1:]
                n_children = idx.numel()
                for c in range(n_children):
                    nt = idx[c:c + 1].view(1, 1)
                    real_tok = nt + m.d2t[nt]
                    expanded.append({
                        "cache": clone_cache_hidden(cache_new) if c < n_children - 1 else cache_new,
                        "h": h_next,
                        "tok": real_tok,
                        "score": node["score"] + float(vals[c]),
                        "len": node["len"] + 1,
                    })
            expanded.sort(key=lambda n: -n["score"])
            frontier = expanded[:total_token]
        return frontier

    return roll, kv


@torch.no_grad()
def make_eagle3_tree_roll(m, aux, ids, tree_depth, top_k, total_token,
                           draft_cache=None, draft_h=None, cache_len=0, mrope=None):
    """Round-based (real multi-round) tree builder for EAGLE3 -- a SEPARATE
    function from make_tree_roll (that one stays untouched for cmd_tree's
    draft-only timing use), because round decoding needs draft-side
    incremental continuation across rounds and a global (not per-depth)
    final node-count selection, matching the fixes already validated for
    HiViS-way's make_hivis_tree_roll.

    Two things are genuinely different from HiViS-way here, both forced by
    this draft architecture (not a choice):

    1. LlamaAttention.forward's cache design only lets cache SLOT 0 carry an
       arbitrary attention_mask (real matmul + mask add); every later slot
       gets an elementwise/diagonal broadcast against the CURRENT query with
       NO masking at all. So: (a) round 1's full prompt must land in ONE
       call as slot 0 (real causal mask, as before), and (b) a later round's
       several newly-accepted tokens must be appended ONE AT A TIME (each
       becoming its own slot) rather than batched -- a single multi-token
       batched call would let those new positions see each other
       non-causally, since only slot 0 is ever mask-gated.
    2. There is no shared-cache + tree-mask mechanism like cnets_hivis's
       `self.tree_mask` here -- each candidate needs an independently
       continuable cache lineage, so tree branching still costs one
       encode_layers call per node (cloning the cache for every child but
       the last, same as the original make_tree_roll). What IS fixed here:
       the continuing frontier between depths is capped at `top_k` (not
       `total_token`), and ONE global top-`total_token` selection over the
       WHOLE candidate pool happens at the very end -- exactly like
       make_hivis_tree_roll, and for the same reason (log-probs are always
       <= 0, so a global topk-by-cumulative-score is automatically ancestor
       closed).

    `aux`/`ids` are the FULL initial prompt on round 1 (`draft_cache=None`),
    or just the newly-accepted portion on later rounds. roll() returns
    (all_nodes, root_cache, root_h, root_kv): the first three make up what
    the caller persists as next round's draft_cache/draft_h/cache_len --
    snapshotted (via clone_cache_hidden) BEFORE tree exploration starts, so
    the tree loop's own appends (on the SAME list objects) can never leak
    into it. `all_nodes` matches verify_tree_hivis/verify_tree_eagle3's
    input shape: a flat list of {"tok","parent","depth","score"} dicts.

    `mrope` is (prefix_pos, rope_delta) from qwen_mrope_prefix, or None for
    non-mrope drafts / once --prefill noimg has dropped the image span (see
    cmd_round_eagle3). Row positions here describe the ROW, not which token
    happens to sit in it, so `ids`/`aux` being the EAGLE-shifted draft_ids
    doesn't change what position each row gets.
    """
    progressive = bool(getattr(m, "progressive_staged", False))

    def roll():
        if draft_cache is None:
            h = m.combine_hidden_states(aux)
            cache = m.init_cache_hidden()
            S = ids.shape[1]
            pos0 = mrope[0] if mrope is not None else torch.arange(S, device=DEV).unsqueeze(0)
            msk0 = torch.full((1, 1, S, S), float("-inf"), device=DEV, dtype=h.dtype).triu(1)
            h, cache = m.encode_layers(m.embed_tokens(ids).to(h.dtype), h, cache, msk0, pos0, True)
            h = h[:, -1:]
            kv_now = S
        else:
            h, cache, kv_now = draft_h, draft_cache, cache_len
            S = ids.shape[1]
            for i in range(S):
                h_in = m.combine_hidden_states(aux[:, i:i + 1])
                pos_i = _pos_at(kv_now, mrope, DEV)
                msk_i = torch.zeros(1, 1, 1, 1, device=DEV, dtype=h_in.dtype)
                h_in, cache = m.encode_layers(
                    m.embed_tokens(ids[:, i:i + 1]).to(h_in.dtype), h_in, cache, msk_i, pos_i, True)
                h = h_in
                kv_now += 1

        root_cache = clone_cache_hidden(cache)
        root_h = h.clone()
        root_kv = kv_now

        t = m.compute_logits(h).argmax(-1)
        tok0 = t + m.d2t[t]
        h_tree = (m.take_progressive_draft_feedback() if progressive
                  else m.next_hidden_from_encode(h))
        if h_tree.shape[1] > 1:
            h_tree = h_tree[:, -1:]
        start = kv_now

        pool = [{"tok": tok0, "parent": None, "depth": 0, "score": 0.0,
                 "cache": cache, "h": h_tree}]
        frontier_idx = [0]
        for d in range(tree_depth):
            new_idx = []
            for node_i in frontier_idx:
                node = pool[node_i]
                pos = _pos_at(start + d, mrope, DEV)
                msk = torch.zeros(1, 1, 1, 1, device=DEV, dtype=node["h"].dtype)
                h_new, cache_new = m.encode_layers(
                    m.embed_tokens(node["tok"]).to(node["h"].dtype),
                    node["h"], node["cache"], msk, pos, True)
                lg = m.compute_logits(h_new)
                logp = torch.log_softmax(lg.float(), dim=-1)[0, 0]
                k = min(top_k, logp.numel())
                vals, idx = logp.topk(k)
                h_next = (m.take_progressive_draft_feedback() if progressive
                          else m.next_hidden_from_encode(h_new))
                if h_next.shape[1] > 1:
                    h_next = h_next[:, -1:]
                for c in range(k):
                    nt = idx[c:c + 1].view(1, 1)
                    real_tok = nt + m.d2t[nt]
                    new_idx.append(len(pool))
                    pool.append({
                        "tok": real_tok, "parent": node_i, "depth": d + 1,
                        "score": node["score"] + float(vals[c]),
                        "cache": clone_cache_hidden(cache_new) if c < k - 1 else cache_new,
                        "h": h_next,
                    })
            new_idx.sort(key=lambda i: -pool[i]["score"])
            frontier_idx = new_idx[:top_k]

        n_final = min(total_token, len(pool))
        final_idx = sorted(range(len(pool)), key=lambda i: -pool[i]["score"])[:n_final]
        final_idx.sort()
        old_to_new = {old: new for new, old in enumerate(final_idx)}
        all_nodes = []
        for old in final_idx:
            node = pool[old]
            p_old = node["parent"]
            all_nodes.append({
                "tok": int(node["tok"]), "parent": (None if p_old is None else old_to_new[p_old]),
                "depth": node["depth"], "score": node["score"],
            })
        return all_nodes, root_cache, root_h, root_kv

    return roll


def load_hivis_draft(ckpt):
    """HiViS-way (EAGLE2-style: single last-hidden-state, no draft-vocab
    remap) draft, loaded via the actual HiViS eval-time Model class so tree
    building can reuse the real generation-time forward() this checkpoint
    was designed for -- see HiViS/hivis/model/cnets_hivis.py.
    """
    hivis_root = os.environ.get("HIVIS_ROOT", "/home/hyang/Angel/HiViS")
    if hivis_root not in sys.path:
        sys.path.insert(0, hivis_root)
    from hivis.model.cnets_hivis import Model as HivisModel
    from hivis.model.configs import EConfig
    from safetensors import safe_open

    cfg = EConfig.from_pretrained(ckpt)
    with safe_open(os.path.join(ckpt, "model.safetensors"), framework="pt", device="cpu") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    # Trust the checkpoint's own saved shape over config.json's "bias" field:
    # hivis/train/main_mix.py always constructs Model with the bias=True
    # default regardless of what the config says, so a config claiming
    # bias=false (e.g. smolvlm_256m_config.json) can disagree with what was
    # actually trained -- load_state_dict(strict=True) against the wrong
    # bias setting fails immediately with "Unexpected/Missing key: fc.bias".
    bias = "fc.bias" in state_dict
    residual_count = state_dict["residual"].shape[0]
    m = HivisModel(cfg, bias=bias, residual_count=residual_count)
    m.load_state_dict(state_dict, strict=True)
    m = m.to(device=DEV, dtype=DT).eval()
    return m, cfg


@torch.no_grad()
def hivis_target_inputs(model, proc, image, question, img_tok, is_qwen, extra_ids=None):
    """Like target_inputs(), but extracts the single last hidden state
    (HiViS-way's draft input) instead of an EAGLE3 multi-layer aux concat,
    and prunes the image-token span the way ge_data_qwen.py/ge_data_smolvlm.py
    did when this checkpoint's training data was generated.

    `extra_ids` (optional list[int]): tokens already accepted in EARLIER
    rounds of a multi-round generation, appended to the prompt before this
    round's real prefill -- a full from-scratch reprocess of the growing
    sequence rather than incremental cache continuation across rounds. Not
    the fastest way to do multi-round generation, but it reuses this same
    already-verified function unchanged instead of needing separate target-
    and draft-side KV-cache trim/splice logic (what HiViS's own
    update_inference_inputs does) for something that's here to make the
    round-count/accept_length statistics comparable to HiViS's own, not to
    benchmark peak multi-round throughput.
    """
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=prompt, images=[image], return_tensors="pt").to(DEV)
    if extra_ids:
        extra = torch.tensor([extra_ids], device=DEV, dtype=inp["input_ids"].dtype)
        inp["input_ids"] = torch.cat([inp["input_ids"], extra], dim=1)
        inp["attention_mask"] = torch.ones_like(inp["input_ids"])
        # mm_token_type_ids (Qwen2.5-VL) is sized by the processor for the
        # ORIGINAL (shorter) prompt -- same stale-length issue as
        # verify_candidates_hivis hit; drop it so the model recomputes it
        # fresh against the now-longer input_ids.
        inp.pop("mm_token_type_ids", None)
    out = model(**inp, output_hidden_states=True, use_cache=True)
    hs_last = out.hidden_states[-1].to(DT)
    first = int(out.logits[0, -1].argmax())
    prefix_cache = out.past_key_values
    # Qwen2.5-VL's mrope caches this on the model instance during the same
    # forward call above; tree verify needs it to place new tokens' mrope
    # positions correctly relative to the (variable-length, image-token-
    # dependent) prefix, without having to recompute the prefix's own mrope.
    rope_deltas = getattr(getattr(model, "model", model), "rope_deltas", None)
    ids = inp["input_ids"]
    full_ids = torch.cat([ids[:, 1:], torch.tensor([[first]], device=DEV)], dim=1)
    full_pos = torch.arange(ids.shape[1], device=DEV).unsqueeze(0)
    keep = (ids != img_tok)[0] if img_tok is not None else torch.ones(
        ids.shape[1], dtype=torch.bool, device=DEV)
    kept_ids = full_ids[:, keep]
    kept_hs = hs_last[:, keep]
    if is_qwen:
        # ge_data_qwen.py's remove_visual_span keeps each surviving token's
        # ORIGINAL absolute position -- a gap is left where the image span
        # was, not renumbered contiguous (this is exactly why cnets_dyn_res.py
        # needed a position_ids.max()+1 rope-cache-sizing fix earlier this
        # session: gapped position ids can exceed the naive seq-length bound).
        kept_pos = full_pos[:, keep]
    else:
        # ge_data_smolvlm.py's remove_image_tokens only drops rows; training
        # never saw a preserved-gap position id, just a contiguous arange.
        kept_pos = torch.arange(int(keep.sum()), device=DEV).unsqueeze(0)
    return kept_hs, kept_ids, kept_pos, inp, first, prefix_cache, rope_deltas


@torch.no_grad()
def make_hivis_tree_roll(m, hs, ids, pos, lm_head, tree_depth, top_k, total_token,
                          draft_cache=None, cache_len=0):
    """Matches HiViS's own topK_genrate (cnets_hivis.py) algorithm exactly,
    not just its tree "shape" in spirit: branch into `top_k` candidates
    IMMEDIATELY from the root forward (native's depth-0 == the root's own
    top_k, not a single greedy pick), keep only `top_k` of them alive to
    seed each next depth's forward (bounding every depth step's batch size
    at top_k, not total_token), but RECORD every candidate ever generated at
    any depth into a pool -- then, once tree_depth forwards are done, pick
    the GLOBAL best `total_token` candidates across the WHOLE pool by
    cumulative log-prob (one final torch.topk, exactly like native's
    `torch.topk(scores_list, total_tokens)`) as the actual verify set.

    This two-stage prune (top_k to keep exploring, total_token only at the
    very end) is NOT an approximation of native's algorithm -- pruning to
    total_token after every single depth step (this function's previous
    version) inflates both node count (up to ~1+top_k+total_token*(depth-1)
    instead of total_token+1) and the continuing frontier's batch size
    (total_token instead of top_k) for no benefit, since a global topk by
    cumulative log-prob is automatically "ancestor closed": log-probs are
    always <= 0, so a child's cumulative score can never exceed its parent's,
    meaning if a node clears the global cutoff, so does every one of its
    ancestors (no missing-parent case to special-case, unlike native's own
    `mask_index`/searchsorted bookkeeping, which relies on the exact same
    invariant).

    logits via the TARGET's own lm_head (HiViS-way drafts don't own one --
    they reuse the target's, see model_hivis.py's EaModel). top-k=1
    degenerates to a linear chain, same as the EAGLE3 path.

    `hs`/`ids`/`pos` are the FULL initial state on the very first call for a
    sample (`draft_cache=None`, `cache_len=0`), or just the NEW portion since
    the last round on later calls, continuing `draft_cache` (length
    `cache_len`) -- true incremental draft-side continuation, not a full
    reprocess of the growing sequence every round.

    roll() returns (frontier, all_nodes, root_cache): `root_cache` is the
    draft's cache right after JUST this round's root-prefill step (`kv`
    positions) -- what the caller should keep as next round's `draft_cache`.
    It deliberately does NOT include this round's tree exploration (the
    frontier/all_nodes speculation beyond the root): those hidden states are
    the draft's OWN un-grounded guesses, not the target's real state, so
    caching them would let the next round build on top of potentially wrong
    context. Discarding them and only ever persisting root-prefill steps
    keeps every future round's draft input grounded in verify_tree_hivis's
    real `accept_hidden` -- matching HiViS's own "re-ground every round"
    design (see cmd_round_hivis's docstring).
    """
    kv = cache_len + ids.shape[1]

    def roll():
        # Mask convention matches HiViS's own topK_genrate exactly (see
        # cnets_hivis.py): a plain all-ones 2D attention_mask (standard
        # causal -- every new position sees the whole cache so far,
        # INCLUDING other branches' nodes from earlier depths, not just its
        # own true ancestors) PLUS `self.tree_mask`, a separate attribute
        # cnets_hivis.py's _prepare_decoder_attention_mask overlays onto only
        # the trailing (this step's new positions) x (same) block, making
        # siblings within THIS batch mutually invisible (an identity mask).
        # HiViS accepts the cross-branch leakage into older cached positions
        # as an approximation for the draft's own (inherently speculative)
        # proposal step; only the final target verify needs to be exact.
        attn_mask = torch.ones((1, kv), dtype=torch.bool, device=DEV)
        m.tree_mask = None
        h_out, root_cache = m(hidden_states=hs, input_ids=ids, attention_mask=attn_mask,
                              position_ids=pos, past_key_values=draft_cache, use_cache=True, forward_num=0)
        last_h = h_out[:, -1:]
        logits = lm_head(last_h.to(lm_head.weight.dtype)).float()
        logp0 = torch.log_softmax(logits, dim=-1)[0, 0]  # [V]
        start_pos = int(pos[0, -1].item()) + 1
        k0 = min(top_k, logp0.shape[-1])
        vals0, idx0 = logp0.topk(k0)  # depth-0 candidates, straight from the root -- matches native

        # Pool: every candidate ever generated, any depth (for the final
        # global top-total_token selection). Continuing frontier (forwarded
        # into the NEXT depth's batch) is a SEPARATE, top_k-capped subset --
        # see docstring.
        pool_tok = idx0
        pool_parent = torch.full((k0,), -1, device=DEV, dtype=torch.long)  # -1 == parent is `first`
        pool_depth = torch.zeros(k0, device=DEV, dtype=torch.long)
        pool_score = vals0
        pool_h = last_h.expand(-1, k0, -1).contiguous()  # all depth-0 candidates share the root's hidden state

        frontier = torch.arange(k0, device=DEV)  # indices into the pool
        cache = root_cache
        tree_cache_len = kv

        for d in range(1, tree_depth + 1):
            B = frontier.numel()
            tok_batch = pool_tok.index_select(0, frontier).unsqueeze(0)  # [1, B]
            h_batch = pool_h.index_select(1, frontier)                   # [1, B, H]
            p = torch.full((1, B), start_pos + d - 1, dtype=torch.long, device=DEV)

            attn_mask = torch.ones((1, tree_cache_len + B), dtype=torch.bool, device=DEV)
            m.tree_mask = torch.eye(B, device=DEV, dtype=torch.bool)[None, None]
            h_new, cache = m(hidden_states=h_batch, input_ids=tok_batch, attention_mask=attn_mask,
                              position_ids=p, past_key_values=cache, use_cache=True, forward_num=d)
            tree_cache_len += B
            logits = lm_head(h_new.to(lm_head.weight.dtype)).float()  # [1, B, V]
            logp = torch.log_softmax(logits, dim=-1)[0]  # [B, V]

            k = min(top_k, logp.shape[-1])
            vals, idx = logp.topk(k, dim=-1)  # [B, k] each -- ONE batched call, not B calls
            cand_scores = pool_score.index_select(0, frontier).unsqueeze(1) + vals  # [B, k]

            base = pool_tok.numel()
            new_tok = idx.reshape(-1)
            new_parent = frontier.repeat_interleave(k)
            new_depth = torch.full((B * k,), d, device=DEV, dtype=torch.long)
            new_score = cand_scores.reshape(-1)
            new_h = h_new.repeat_interleave(k, dim=1)  # [1, B*k, H] -- each of B parents' h repeated k times

            pool_tok = torch.cat([pool_tok, new_tok])
            pool_parent = torch.cat([pool_parent, new_parent])
            pool_depth = torch.cat([pool_depth, new_depth])
            pool_score = torch.cat([pool_score, new_score])
            pool_h = torch.cat([pool_h, new_h], dim=1)

            # Continuing frontier for the NEXT depth: top_k best of THIS
            # depth's B*k new candidates -- bounds every forward's batch
            # size at top_k, matching native (not total_token, which is only
            # applied once, globally, at the very end below).
            n_continue = min(top_k, new_score.numel())
            _, top_local_idx = new_score.topk(n_continue)
            frontier = base + top_local_idx

        # Final global selection: best `total_token` candidates across the
        # WHOLE pool (every depth). Ancestor-closed by score monotonicity
        # (see docstring), so every selected node's parent is either -1 or
        # also present in final_idx -- no orphaned nodes to special-case.
        n_final = min(total_token, pool_score.numel())
        _, final_idx = pool_score.topk(n_final)
        final_idx, _ = torch.sort(final_idx)
        final_idx_l = final_idx.tolist()
        old_to_new = {old: new for new, old in enumerate(final_idx_l)}

        pool_tok_l, pool_parent_l, pool_depth_l, pool_score_l = (
            pool_tok.tolist(), pool_parent.tolist(), pool_depth.tolist(), pool_score.tolist())
        all_nodes = []
        for old in final_idx_l:
            p_old = pool_parent_l[old]
            new_parent = None if p_old < 0 else old_to_new[p_old]
            all_nodes.append({
                "tok": pool_tok_l[old], "parent": new_parent, "depth": pool_depth_l[old],
                "score": pool_score_l[old],
            })

        has_child = set(p for p in (n["parent"] for n in all_nodes) if p is not None)
        frontier_out = []
        for leaf_i in range(len(all_nodes)):
            if leaf_i in has_child:
                continue
            path, j = [], leaf_i
            while j is not None:
                path.append(all_nodes[j]["tok"])
                j = all_nodes[j]["parent"]
            path.reverse()
            frontier_out.append({"path": path, "score": all_nodes[leaf_i]["score"]})
        return frontier_out, all_nodes, root_cache

    return roll, kv


@torch.no_grad()
def verify_candidates_hivis(tgt, inp, prefix_ids, candidates):
    """Real verify+accept step, matching evaluate_posterior's greedy
    accept-until-first-mismatch semantics: for each candidate token chain
    (target-vocab ids, from make_hivis_tree_roll's frontier "path"), run the
    target ONCE teacher-forced over [real prompt + candidate] and compare its
    argmax predictions against the candidate. Returns the BEST (longest)
    accepted length across all candidates -- what a real round would accept.

    Looped one candidate at a time (batch=1) rather than batched: simpler to
    get right, and this is a correctness/validation tool, not a speed
    benchmark of the verify step itself.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    prefix_len = prefix_ids.shape[1]
    # mm_token_type_ids (Qwen2.5-VL) is computed by the processor for the
    # ORIGINAL (shorter) prompt and cached in `inp`; passing it alongside our
    # longer extended full_ids feeds the model a stale, wrong-length tensor.
    # Drop it and let the model recompute it fresh from the actual input_ids.
    kwargs = {k: v for k, v in inp.items()
              if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
    best_accept, best_chain, best_bonus = 0, candidates[0], None
    for chain in candidates:
        chain_t = torch.tensor([chain], device=DEV, dtype=prefix_ids.dtype)
        full_ids = torch.cat([prefix_ids, chain_t], dim=1)
        # Qwen2.5-VL's mrope get_rope_index() caches rope_deltas from the
        # earlier (shorter) prefill call on this same target instance and
        # reuses it if attention_mask is omitted, producing a stale expected
        # length -- "expanded size ... must match existing size" -- for this
        # longer extended sequence. An explicit full-length mask forces a
        # fresh, correctly-sized computation.
        attn_mask = torch.ones_like(full_ids, dtype=torch.long)
        out = tgt(input_ids=full_ids, attention_mask=attn_mask, **kwargs)
        logits = out.logits[0]
        L = len(chain)
        pred = logits[prefix_len - 1: prefix_len - 1 + L].argmax(-1)
        match = (pred == torch.tensor(chain, device=DEV)).int()
        accept = int(torch.cumprod(match, dim=0).sum())
        if accept >= best_accept:
            # The "bonus" token: the target's own real next-token prediction
            # at the mismatch point (or, if the whole candidate matched with
            # no mismatch observed, its last position's prediction as a
            # best-effort stand-in -- this tool doesn't probe further).
            bonus_idx = accept if accept < L else L - 1
            best_accept, best_chain, best_bonus = accept, chain[:accept], int(pred[bonus_idx])
    torch.cuda.synchronize()
    verify_s = time.perf_counter() - t0
    return best_accept, best_chain, best_bonus, verify_s


@torch.no_grad()
def verify_tree_hivis(tgt, prefix_len, prefix_cache, rope_deltas, first, all_nodes, is_qwen):
    """Single-pass tree verify, matching HiViS's own tree_decoding: flatten
    the whole speculative tree into ONE sequence and verify it with a SINGLE
    target forward call using a tree-structured attention mask (each node
    attends to the real prefix plus its own ancestor chain, not sibling
    branches) -- instead of verify_candidates_hivis's one-target-call-per-
    candidate loop, which is correct but does up to `total_token` redundant
    target forwards for a tree.

    Continues from `prefix_cache` -- a persistent DynamicCache the CALLER
    grows round over round via compact_cache(), never rebuilt from scratch.
    `prefix_len` is that cache's current length (an int, not re-derived from
    a fresh prefill each round -- see cmd_round_hivis's round loop). For
    Qwen2.5-VL's mrope, `rope_deltas` is computed ONCE at the very first
    prefill and stays valid for every later round (confirmed empirically:
    pure-text continuation after the image keeps a constant flat-position-to-
    mrope-position delta), so it's threaded through unchanged, never
    recomputed.

    Returns (best_accept, best_chain, best_bonus, accept_hidden,
    select_indices, verify_s): `accept_hidden` is the target's REAL hidden
    state at each of the `best_accept` accepted positions (in order), the
    correct grounding input for the NEXT round's draft tree (matching
    HiViS's own "re-ground to the target's real state every round" design,
    not the draft's own possibly-wrong guess). `select_indices` are those
    same positions' ABSOLUTE cache indices (already offset by `prefix_len`),
    ready to pass straight to compact_cache.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Flatten: index 0 is the free bonus token `first`; indices 1.. are
    # make_hivis_tree_roll's all_nodes, in registration order.
    toks = [first] + [n["tok"] for n in all_nodes]
    depths = [0] + [n["depth"] + 1 for n in all_nodes]  # `first` is one before the real tree root
    Tn = len(toks)
    new_ids = torch.tensor([toks], device=DEV, dtype=torch.long)

    # parent[i]: index (into this same 0..Tn-1 numbering) of i's parent.
    # parent[0] (`first`) is None -- it has no ancestor within this new
    # chunk, only the real prefix (handled separately below).
    parent = [None] + [(n["parent"] + 1) if n["parent"] is not None else 0 for n in all_nodes]
    # Vectorized ancestor-chain walk: node 0 (`first`) self-loops as its own
    # parent, so repeatedly following `parent_t` converges there and stays --
    # marking it redundantly on later hops is harmless. This replaces a
    # per-node Python loop (up to Tn iterations, each walking+writing one
    # GPU element at a time -- ~21ms measured for Tn~250) with `max_hops`
    # (<=tree_depth+2) batched tensor scatter ops, all nodes at once.
    parent_t = torch.tensor([0 if p is None else p for p in parent], device=DEV, dtype=torch.long)
    node_idx = torch.arange(Tn, device=DEV)
    ancestor_mask = torch.zeros(Tn, Tn, dtype=torch.bool, device=DEV)
    cur = node_idx.clone()
    max_hops = max(depths) + 1
    for _ in range(max_hops):
        ancestor_mask[node_idx, cur] = True
        cur = parent_t[cur]

    additive_mask = torch.zeros(1, 1, Tn, prefix_len + Tn, dtype=DT, device=DEV)
    additive_mask[:, :, :, prefix_len:] = torch.where(
        ancestor_mask, torch.zeros((), dtype=DT, device=DEV), torch.finfo(DT).min
    )
    # additive_mask[:, :, :, :prefix_len] stays 0 -- every new node attends
    # to the whole real prefix, unconditionally.

    positions = torch.tensor([[prefix_len + d for d in depths]], device=DEV, dtype=torch.long)
    if is_qwen and rope_deltas is not None:
        positions = positions + rope_deltas.view(-1, 1)
        positions = positions.unsqueeze(0).expand(3, -1, -1)

    out = tgt(input_ids=new_ids, attention_mask=additive_mask, position_ids=positions,
              past_key_values=prefix_cache, use_cache=True, output_hidden_states=True)
    logits = out.logits[0]  # [Tn, V]
    hs_new = out.hidden_states[-1]  # [1, Tn, H]

    # For each surviving LEAF -- any node that is not itself some other
    # node's parent, matching native's own `noleaf_index`/leaf bookkeeping in
    # topK_genrate, NOT restricted to the single deepest tree level (after
    # make_hivis_tree_roll's global top-total_token pruning, the best chains
    # can legitimately end at different depths, since cumulative log-prob
    # naturally favors shorter high-confidence chains over longer
    # low-confidence ones) -- walk its ancestor chain (via `parent`) and
    # check how much of it the target's own argmax (at the PARENT's logit
    # row) actually predicts -- the greedy accept-until-first-mismatch rule,
    # same as verify_candidates_hivis, just sourced from one shared forward
    # pass instead of per-candidate ones.
    has_child = set(p for p in parent if p is not None)
    best_accept, best_chain_idx, best_bonus = 0, None, None
    for leaf_idx in range(Tn):
        if leaf_idx in has_child:
            continue
        chain_idx = []
        j = leaf_idx
        while j is not None:
            chain_idx.append(j)
            j = parent[j]
        chain_idx.reverse()  # root(`first`) .. leaf
        # chain_idx[0] is `first` itself -- trivially "accepted" (it IS the
        # target's own real prediction), matching verify_candidates_hivis's
        # convention of always returning >= 1 so the caller's -1 correction
        # (excluding `first` from the reported accept_length) lands right.
        accept = 1
        bonus = None
        for pos_in_chain in range(1, len(chain_idx)):
            parent_node, child_node = chain_idx[pos_in_chain - 1], chain_idx[pos_in_chain]
            pred = int(logits[parent_node].argmax())
            if pred == toks[child_node]:
                accept += 1
            else:
                bonus = pred
                break
        if bonus is None:  # fully matched to the tree's own depth limit
            bonus = int(logits[chain_idx[-1]].argmax())
        if accept >= best_accept:
            best_accept = accept
            best_chain_idx = chain_idx[:accept]
            best_bonus = bonus

    best_chain = [toks[i] for i in best_chain_idx]
    # accept_hidden[i] is the hidden state that PREDICTED best_chain[i+1] (or
    # best_bonus for i == best_accept-1) -- the same "shifted" pairing
    # hivis_target_inputs itself uses (kept_hs[-1] predicts kept_ids[-1]).
    # The caller extends its growing (hs, ids, pos) history with this,
    # keeping every future make_hivis_tree_roll prefill grounded in real
    # target states instead of the draft's own (possibly wrong) guesses.
    accept_hidden = hs_new[:, best_chain_idx, :]
    # Absolute cache positions (this round's new_ids occupy [prefix_len,
    # prefix_len+Tn)), ready to hand straight to compact_cache.
    select_indices = torch.tensor([prefix_len + i for i in best_chain_idx], device=DEV, dtype=torch.long)

    torch.cuda.synchronize()
    verify_s = time.perf_counter() - t0
    return best_accept, best_chain, best_bonus, accept_hidden, select_indices, verify_s


@torch.no_grad()
def verify_tree_eagle3(tgt, prefix_len, prefix_cache, first, all_nodes, aux_ids, mrope=None):
    """Single-pass target verify for EAGLE3 round decoding -- same design as
    verify_tree_hivis (flatten the tree, ONE target forward with a
    tree-structured additive attention mask built via the same vectorized
    ancestor-chain walk, greedy accept-until-mismatch per leaf), generalized
    two ways for EAGLE3:

    - `mrope` (from qwen_mrope_prefix): SmolVLM's target uses plain 1D
      position_ids (mrope=None); Qwen-VL-family targets need the real
      rope_delta-continued scalar positions (see _pos_at) -- position_ids is
      always explicit either way, never left None (manual incremental decode
      with position_ids=None corrupts a Qwen-VL target's attention output).
    - accept_hidden concatenates EVERY aux_ids layer (not just the last
      one), since that's what the draft's combine_hidden_states expects as
      its next-round grounding input (make_eagle3_tree_roll's `aux` arg).

    Leaves are "any node that is not itself some other node's parent" (not
    restricted to the single deepest tree level), same fix as
    verify_tree_hivis: after make_eagle3_tree_roll's global top-total_token
    pruning, the best chains can legitimately end at different depths.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    toks = [first] + [n["tok"] for n in all_nodes]
    depths = [0] + [n["depth"] + 1 for n in all_nodes]
    Tn = len(toks)
    new_ids = torch.tensor([toks], device=DEV, dtype=torch.long)
    parent = [None] + [(n["parent"] + 1) if n["parent"] is not None else 0 for n in all_nodes]

    parent_t = torch.tensor([0 if p is None else p for p in parent], device=DEV, dtype=torch.long)
    node_idx = torch.arange(Tn, device=DEV)
    ancestor_mask = torch.zeros(Tn, Tn, dtype=torch.bool, device=DEV)
    cur = node_idx.clone()
    max_hops = max(depths) + 1
    for _ in range(max_hops):
        ancestor_mask[node_idx, cur] = True
        cur = parent_t[cur]

    additive_mask = torch.zeros(1, 1, Tn, prefix_len + Tn, dtype=DT, device=DEV)
    additive_mask[:, :, :, prefix_len:] = torch.where(
        ancestor_mask, torch.zeros((), dtype=DT, device=DEV), torch.finfo(DT).min
    )
    if mrope is not None:
        delta = int(mrope[1].reshape(-1)[0].item())
        positions = torch.tensor([[prefix_len + d + delta for d in depths]], device=DEV, dtype=torch.long)
        positions = positions.view(1, 1, -1).expand(3, -1, -1)
    else:
        positions = torch.tensor([[prefix_len + d for d in depths]], device=DEV, dtype=torch.long)

    out = tgt(input_ids=new_ids, attention_mask=additive_mask, position_ids=positions,
              past_key_values=prefix_cache, use_cache=True, output_hidden_states=True)
    logits = out.logits[0]
    hs_all = out.hidden_states  # tuple, len = n_layers + 1

    has_child = set(p for p in parent if p is not None)
    best_accept, best_chain_idx, best_bonus = 0, None, None
    for leaf_idx in range(Tn):
        if leaf_idx in has_child:
            continue
        chain_idx = []
        j = leaf_idx
        while j is not None:
            chain_idx.append(j)
            j = parent[j]
        chain_idx.reverse()
        accept = 1
        bonus = None
        for pos_in_chain in range(1, len(chain_idx)):
            parent_node, child_node = chain_idx[pos_in_chain - 1], chain_idx[pos_in_chain]
            pred = int(logits[parent_node].argmax())
            if pred == toks[child_node]:
                accept += 1
            else:
                bonus = pred
                break
        if bonus is None:
            bonus = int(logits[chain_idx[-1]].argmax())
        if accept >= best_accept:
            best_accept = accept
            best_chain_idx = chain_idx[:accept]
            best_bonus = bonus

    best_chain = [toks[i] for i in best_chain_idx]
    accept_aux = torch.cat([hs_all[j + 1][:, best_chain_idx, :] for j in aux_ids], dim=-1).to(DT)
    select_indices = torch.tensor([prefix_len + i for i in best_chain_idx], device=DEV, dtype=torch.long)

    torch.cuda.synchronize()
    verify_s = time.perf_counter() - t0
    return best_accept, best_chain, best_bonus, accept_aux, select_indices, verify_s


def compact_cache(cache, prev_len, select_indices):
    """TRUE incremental cache continuation across rounds: keep [:prev_len]
    (already-confirmed) plus the gathered accepted positions from this
    round's speculative block; drop the rest of this round's (rejected)
    entries. Ported unchanged from tools/hf_hivis_eval_tree.py -- this
    function is architecture-agnostic (only touches DynamicCache's own
    per-layer keys/values), so both HiViS-way and (later) EAGLE3 share it.

    This replaces cmd_round_hivis's earlier "reprocess the whole growing
    prompt from scratch every round" approach, which was confirmed buggy:
    the SAME target instance produced a different (wrong) prediction on a
    later round than an isolated fresh-model run of that identical round,
    with weights/buffers verified byte-identical before and after -- i.e.
    the bug was in doing something numerically non-equivalent to real
    incremental decoding, not in any one line of arithmetic. True KV-cache
    continuation (this function) is what HiViS's own update_inference_inputs
    does and doesn't have that problem.
    """
    for layer in cache.layers:
        acc_k = layer.keys[..., select_indices, :]
        acc_v = layer.values[..., select_indices, :]
        layer.keys = torch.cat([layer.keys[..., :prev_len, :], acc_k], dim=-2)
        layer.values = torch.cat([layer.values[..., :prev_len, :], acc_v], dim=-2)


@torch.no_grad()
def naive_baseline_hivis(tgt, inp, num_tokens):
    """Plain autoregressive generation of `num_tokens` tokens (no speculative
    decoding at all) -- the "how fast without any of this" reference point
    for computing speedup, matching what a round of draft+verify replaces.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tgt.generate(**inp, max_new_tokens=num_tokens, min_new_tokens=num_tokens,
                 do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def load_hivis_benchmark_samples(hivis_dataset_name, n, warmup=3):
    """(image, question) pairs sourced from HiViS's OWN benchmark_data.py
    (same shuffle, same fixed seed) instead of draft_bench.py's own
    load_samples (first n rows, no shuffle at all) -- comparing against
    HiViS's native eagenerate on "the same n samples" only means anything if
    both sides are actually looking at the same underlying questions.
    Skips the first `warmup` samples, matching /home/hyang/tmp/
    hivis_way_native_bench.py's own WARMUP=3 indexing, so "sample 0" here is
    the SAME question as "sample 0" in that script's timed (non-warmup) loop.
    Currently wired for ChartQA (this file's only --dataset in practice);
    extend the branch below if another --dataset needs the same treatment.
    """
    hivis_root = os.environ.get("HIVIS_ROOT", "/home/hyang/Angel/HiViS")
    if hivis_root not in sys.path:
        sys.path.insert(0, hivis_root)
    from hivis.evaluation.benchmark_data import load_benchmark
    if hivis_dataset_name != "ChartQA":
        raise NotImplementedError(
            f"load_hivis_benchmark_samples only knows ChartQA's field names; got {hivis_dataset_name!r}")
    dataset = load_benchmark("ChartQA", sample_count=n + warmup)
    rows = [dataset[i] for i in range(warmup, warmup + n)]
    return [(row["image"].convert("RGB"), str(row["query"])) for row in rows]


def setup_hivis(args):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    m, cfg = load_hivis_draft(args.ckpt)
    proc = AutoProcessor.from_pretrained(args.target)
    tgt = AutoModelForImageTextToText.from_pretrained(
        args.target, dtype=DT, device_map=DEV).eval()
    img_tok = getattr(tgt.config, "image_token_id", None)
    is_qwen = "qwen" in tgt.config.model_type.lower()
    samples = load_hivis_benchmark_samples("ChartQA", args.n)
    print(f"ckpt    : {args.ckpt}")
    print(f"draft   : hivis-way (EAGLE2) layers={len(m.layers)} bias={'fc.bias' in m.state_dict()} "
          f"residual_count={m.residual.shape[0]}")
    print(f"data    : {args.dataset}  n={len(samples)}  image_token_id={img_tok}  is_qwen={is_qwen}")
    return m, proc, tgt, img_tok, is_qwen, samples


def cmd_tree_hivis(args):
    m, proc, tgt, img_tok, is_qwen, samples = setup_hivis(args)
    lm_head = tgt.lm_head
    print(f"tree    : top-k={args.top_k} tree-depth={args.tree_depth} total-token={args.total_token} "
          f"(top-k=1 degenerates to a linear chain)")
    per, kvs, sizes = [], [], []
    for k, (img, q) in enumerate(samples):
        hs, ids, pos, _inp, _first, _cache, _rd = hivis_target_inputs(tgt, proc, img, q, img_tok, is_qwen)
        roll, kv = make_hivis_tree_roll(m, hs, ids, pos, lm_head, args.tree_depth, args.top_k, args.total_token)
        frontier, _all_nodes, _root_cache = roll()
        ms = cuda_time(lambda: roll()[0], args.iters, args.warmup)
        per.append(ms); kvs.append(kv); sizes.append(len(frontier))
        print(f"  [{k}] S={ids.shape[1]:5d} kv={kv:5d} tree_nodes={len(frontier):4d}  "
              f"{ms:8.3f} ms/tree  {ms/args.tree_depth:7.4f} ms/depth-step")
    mean = sum(per) / len(per)
    res = dict(ckpt=args.ckpt, dataset=args.dataset, n=len(samples), mode="tree", draft_arch="hivis",
               top_k=args.top_k, tree_depth=args.tree_depth, total_token=args.total_token,
               mean_kv=sum(kvs)/len(kvs), mean_tree_nodes=sum(sizes)/len(sizes),
               ms_per_tree=mean, ms_per_depth_step=mean/args.tree_depth,
               per_sample_ms=[round(p, 4) for p in per])
    print(f"\nMEAN  kv={res['mean_kv']:.0f}  nodes={res['mean_tree_nodes']:.1f}  "
          f"{mean:.3f} ms/tree  {res['ms_per_depth_step']:.4f} ms/depth-step")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(res, open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")


def cmd_run_hivis(args):
    # linear == tree with branching factor 1, so every depth only ever has
    # ONE candidate anyway; total_token must be >= steps+1 (the pool size
    # with top_k=1) so make_hivis_tree_roll's global top-total_token
    # selection (see its docstring) keeps the WHOLE chain instead of
    # collapsing it down to just its single best-scoring node. (total_token
    # used to be a PER-DEPTH cap, where 1 was a safe no-op here since width
    # was already 1 -- that per-depth-pruning design was replaced this
    # session with make_hivis_tree_roll's global final selection, which
    # made the old total_token=1 here silently truncate to 1 output token.)
    args.top_k, args.total_token, args.tree_depth = 1, args.steps + 1, args.steps
    cmd_tree_hivis(args)


def cmd_round_hivis(args):
    """Full multi-round generation: draft builds a tree, VERIFY against the
    real target with a SINGLE tree-mask forward per round, accept the
    longest matching prefix + one fresh bonus token, append to the running
    generation, and repeat until `--max-new-tokens` or EOS -- matching
    HiViS's own eagenerate loop shape (and its --max-new-token default of
    200) so the accept_length/throughput statistics are pooled over a
    comparable NUMBER OF ROUNDS, not just one round per sample.

    TRUE incremental continuation across rounds (not the earlier "reprocess
    the whole growing prompt from scratch every round" approach, which was
    confirmed buggy -- see compact_cache's docstring):
      - target side: prefix_cache is built ONCE (hivis_target_inputs) and
        only ever grown via compact_cache -- never rebuilt.
      - draft side: no persistent cache needed (it's a 1-layer network, a
        fresh small forward each round is cheap and carries no correctness
        risk); instead its (hs, ids, pos) inputs grow round over round,
        extended with the target's OWN real hidden states at the accepted
        positions (verify_tree_hivis's `accept_hidden`) -- re-grounding the
        draft to reality every round, matching HiViS's own design, instead
        of ever feeding it back its own (possibly wrong) prior guesses.
    """
    m, proc, tgt, img_tok, is_qwen, samples = setup_hivis(args)
    lm_head = tgt.lm_head
    eos_id = getattr(proc.tokenizer, "eos_token_id", None)
    print(f"round   : top-k={args.top_k} tree-depth={args.tree_depth} total-token={args.total_token} "
          f"max-new-tokens={args.max_new_tokens} draft+verify+accept, multi-round per sample, "
          f"vs. naive-autoregressive baseline")
    all_accepts, all_new_tokens, all_wall_s = [], [], []
    baseline_tok_s_list = []
    for k, (img, q) in enumerate(samples):
        hs, ids, pos, inp, first, prefix_cache, rope_deltas = hivis_target_inputs(
            tgt, proc, img, q, img_tok, is_qwen)
        prev_len = inp["input_ids"].shape[1]
        last_draft_pos = int(pos[0, -1].item())

        generated, round_accepts, wall_s = [], [], 0.0
        first_round = True
        # Draft-side incremental continuation: draft_cache/draft_cache_len
        # persist round over round (the root-prefill-only cache, grounded to
        # the target's real hidden states each time -- see
        # make_hivis_tree_roll's docstring). Round 1 passes the full initial
        # (hs, ids, pos); every later round passes ONLY the newly-accepted
        # portion, since the rest already lives inside draft_cache.
        draft_cache, draft_cache_len = None, 0
        while len(generated) < args.max_new_tokens:
            roll, kv = make_hivis_tree_roll(
                m, hs, ids, pos, lm_head, args.tree_depth, args.top_k, args.total_token,
                draft_cache=draft_cache, cache_len=draft_cache_len)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            frontier, all_nodes, root_cache = roll()
            torch.cuda.synchronize()
            draft_s = time.perf_counter() - t0

            accept_with_bonus, best_chain, best_bonus, accept_hidden, select_indices, verify_s = (
                verify_tree_hivis(tgt, prev_len, prefix_cache, rope_deltas, first, all_nodes, is_qwen))

            compact_cache(prefix_cache, prev_len, select_indices)
            prev_len += accept_with_bonus
            # root_cache/kv is the draft's cache right after JUST this
            # round's root-prefill step (built from accept_hidden, i.e. the
            # target's real hidden states) -- carries forward as next
            # round's draft_cache. The tree-exploration extension on top of
            # it (the draft's own ungrounded guesses) is simply dropped by
            # not keeping a reference to `cache` (roll()'s internal var).
            draft_cache, draft_cache_len = root_cache, kv

            # best_chain[0] IS `first` -- already reported as output on a
            # PRIOR round (or, on round 1, not yet reported at all).
            new_hist_tokens = best_chain[1:] + [best_bonus]
            new_output_tokens = (best_chain + [best_bonus]) if first_round else new_hist_tokens
            generated.extend(new_output_tokens)
            round_accepts.append(accept_with_bonus - 1)  # exclude the trivial `first` self-match
            wall_s += draft_s + verify_s
            if eos_id is not None and eos_id in new_output_tokens:
                # A whole round's accepted chain can run past the real stop
                # point (the tree explored speculative continuations beyond
                # where generation should have logically ended) -- trim
                # anything after the first EOS so token counts/throughput
                # reflect what a real generation would actually emit.
                generated = generated[:generated.index(eos_id) + 1]
                break

            hs = accept_hidden
            ids = torch.tensor([new_hist_tokens], device=DEV, dtype=torch.long)
            new_pos = torch.arange(
                last_draft_pos + 1, last_draft_pos + 1 + len(new_hist_tokens), device=DEV).unsqueeze(0)
            pos = new_pos
            last_draft_pos = int(pos[0, -1].item())
            first = best_bonus
            first_round = False

        all_accepts.extend(round_accepts)
        all_new_tokens.append(len(generated))
        all_wall_s.append(wall_s)
        tok_s = len(generated) / wall_s
        print(f"  [{k}] new_tokens={len(generated):4d}  rounds={len(round_accepts):3d}  "
              f"mean_accept_this_sample={sum(round_accepts)/len(round_accepts):.3f}  "
              f"wall={wall_s:.3f}s  {tok_s:7.2f} tok/s")

        # One baseline call per sample, generating the SAME number of tokens
        # this sample's speculative loop produced -- matches HiViS's own
        # speed.py convention of pooling per-sample new_tokens/wall_time
        # ratios rather than per-round baseline calls.
        _, _, _, base_inp, _, _, _ = hivis_target_inputs(tgt, proc, img, q, img_tok, is_qwen)
        baseline_s = naive_baseline_hivis(tgt, base_inp, len(generated))
        baseline_tok_s_list.append(len(generated) / baseline_s)

    mean_accept = sum(all_accepts) / len(all_accepts)
    mean_round = sum(all_new_tokens) / sum(all_wall_s)
    mean_baseline = sum(baseline_tok_s_list) / len(baseline_tok_s_list)
    print(f"\nMEAN  accept_length={mean_accept:.3f} (over {len(all_accepts)} rounds)  "
          f"round={mean_round:.2f} tok/s  baseline={mean_baseline:.2f} tok/s  "
          f"speedup={mean_round/mean_baseline:.2f}x  (n={len(samples)})")
    if args.json:
        res = dict(ckpt=args.ckpt, dataset=args.dataset, n=len(samples), mode="round", draft_arch="hivis",
                   top_k=args.top_k, tree_depth=args.tree_depth, total_token=args.total_token,
                   max_new_tokens=args.max_new_tokens, mean_accept_length=mean_accept,
                   mean_round_tok_s=mean_round, mean_baseline_tok_s=mean_baseline,
                   speedup=mean_round/mean_baseline, num_rounds=len(all_accepts),
                   per_round_accept=all_accepts)
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(res, open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")


def setup(args):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    m, aux_ids, cfg = load_draft(args.ckpt, args.depth, target_model_name_or_path=args.target)
    proc = AutoProcessor.from_pretrained(args.target)
    tgt = AutoModelForImageTextToText.from_pretrained(
        args.target, dtype=DT, device_map=DEV).eval()
    img_tok = getattr(tgt.config, "image_token_id", None)
    samples = load_samples(args.dataset, args.n, getattr(args, "prompt_style", "raw"))
    print(f"ckpt    : {args.ckpt}")
    print(f"draft   : mode={cfg.get('eagle_aux_injection_mode')} layers={len(m.layers)} "
          f"aux={aux_ids} (n={len(aux_ids)})")
    print(f"data    : {args.dataset}  n={len(samples)}  image_token_id={img_tok}")
    print(f"ablation: prefill={args.prefill} sampler={args.sampler} steps={args.steps}")
    return m, aux_ids, proc, tgt, img_tok, samples


def cmd_run(args):
    if args.draft_arch == "hivis":
        return cmd_run_hivis(args)
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


def cmd_tree(args):
    if args.draft_arch == "hivis":
        return cmd_tree_hivis(args)
    m, aux_ids, proc, tgt, img_tok, samples = setup(args)
    print(f"tree    : top-k={args.top_k} tree-depth={args.tree_depth} total-token={args.total_token} "
          f"(top-k=1 degenerates to a linear chain, like make_roll)")
    per, kvs, sizes = [], [], []
    for k, (img, q) in enumerate(samples):
        aux, ids, mask, first = target_inputs(tgt, proc, img, q, aux_ids, img_tok)
        roll, kv = make_tree_roll(m, aux, ids, mask, first, args.prefill, args.sampler,
                                   args.tree_depth, args.top_k, args.total_token)
        frontier = roll()  # untimed warmup + correctness peek
        ms = cuda_time(roll, args.iters, args.warmup)
        per.append(ms); kvs.append(kv); sizes.append(len(frontier))
        print(f"  [{k}] S={ids.shape[1]:5d} kv={kv:5d} tree_nodes={len(frontier):4d}  "
              f"{ms:8.3f} ms/tree  {ms/args.tree_depth:7.4f} ms/depth-step")
    mean = sum(per) / len(per)
    res = dict(ckpt=args.ckpt, dataset=args.dataset, n=len(samples), mode="tree",
               top_k=args.top_k, tree_depth=args.tree_depth, total_token=args.total_token,
               layers=len(m.layers), aux_layer_ids=aux_ids, mean_kv=sum(kvs)/len(kvs),
               mean_tree_nodes=sum(sizes)/len(sizes),
               ms_per_tree=mean, ms_per_depth_step=mean/args.tree_depth,
               per_sample_ms=[round(p, 4) for p in per])
    print(f"\nMEAN  kv={res['mean_kv']:.0f}  nodes={res['mean_tree_nodes']:.1f}  "
          f"{mean:.3f} ms/tree  {res['ms_per_depth_step']:.4f} ms/depth-step")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(res, open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")


def cmd_round_eagle3(args):
    """Multi-round tree speculative decoding for EAGLE3 (SmolVLM), matching
    cmd_round_hivis's design one-for-one:
      - target side: prefix_cache is built ONCE (eagle3_target_prefill) and
        only ever grown via compact_cache -- never rebuilt.
      - draft side: draft_cache/draft_h persist round over round
        (make_eagle3_tree_roll), re-grounded every round to the target's
        REAL multi-layer aux hidden states at the accepted positions
        (verify_tree_eagle3's accept_aux) -- never the draft's own guess.
    """
    m, aux_ids, proc, tgt, img_tok, samples = setup(args)
    eos_id = getattr(proc.tokenizer, "eos_token_id", None)
    print(f"round   : top-k={args.top_k} tree-depth={args.tree_depth} total-token={args.total_token} "
          f"max-new-tokens={args.max_new_tokens} draft+verify+accept, multi-round per sample (eagle3), "
          f"vs. naive-autoregressive baseline")
    all_accepts, all_new_tokens, all_wall_s = [], [], []
    baseline_tok_s_list = []
    for k, (img, q) in enumerate(samples):
        aux, ids, first, prefix_cache, prev_len, mask, mrope = eagle3_target_prefill(
            tgt, proc, img, q, aux_ids, img_tok=img_tok if args.prefill == "noimg" else None)
        # The TARGET always sees every image row regardless of --prefill, so
        # verify_tree_eagle3 always gets the real mrope. The DRAFT only gets
        # it while its own sequence still has the image span in original
        # order (qwen_mrope_prefix's docstring / draft_mrope=None once
        # --prefill noimg has pruned it -- see make_eagle3_tree_roll).
        draft_mrope = None if args.prefill == "noimg" else mrope
        if args.prefill == "noimg":
            # Same ablation as make_roll's --prefill noimg: drop image rows from
            # the DRAFT's initial (aux, ids) only -- the target (prefix_cache)
            # keeps the full prompt untouched. keep[-1]=True preserves the final
            # (shifted-in `first`) position make_eagle3_tree_roll roots from.
            keep = ~mask
            keep[-1] = True
            aux, ids = aux[:, keep], ids[:, keep]

        generated, round_accepts, wall_s = [], [], 0.0
        first_round = True
        draft_cache, draft_h, draft_cache_len = None, None, 0
        while len(generated) < args.max_new_tokens:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            roll = make_eagle3_tree_roll(m, aux, ids, args.tree_depth, args.top_k, args.total_token,
                                          draft_cache=draft_cache, draft_h=draft_h, cache_len=draft_cache_len,
                                          mrope=draft_mrope)
            all_nodes, root_cache, root_h, root_kv = roll()
            torch.cuda.synchronize()
            draft_s = time.perf_counter() - t0

            accept_with_bonus, best_chain, best_bonus, accept_aux, select_indices, verify_s = (
                verify_tree_eagle3(tgt, prev_len, prefix_cache, first, all_nodes, aux_ids, mrope=mrope))

            compact_cache(prefix_cache, prev_len, select_indices)
            prev_len += accept_with_bonus
            draft_cache, draft_h, draft_cache_len = root_cache, root_h, root_kv

            new_hist_tokens = best_chain[1:] + [best_bonus]
            new_output_tokens = (best_chain + [best_bonus]) if first_round else new_hist_tokens
            generated.extend(new_output_tokens)
            round_accepts.append(accept_with_bonus - 1)
            wall_s += draft_s + verify_s
            if eos_id is not None and eos_id in new_output_tokens:
                generated = generated[:generated.index(eos_id) + 1]
                break

            aux = accept_aux
            ids = torch.tensor([new_hist_tokens], device=DEV, dtype=torch.long)
            first = best_bonus
            first_round = False

        all_accepts.extend(round_accepts)
        all_new_tokens.append(len(generated))
        all_wall_s.append(wall_s)
        tok_s = len(generated) / wall_s
        print(f"  [{k}] new_tokens={len(generated):4d}  rounds={len(round_accepts):3d}  "
              f"mean_accept_this_sample={sum(round_accepts)/len(round_accepts):.3f}  "
              f"wall={wall_s:.3f}s  {tok_s:7.2f} tok/s")

        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        base_inp = proc(text=prompt, images=[img], return_tensors="pt").to(DEV)
        torch.cuda.synchronize()
        tb0 = time.perf_counter()
        tgt.generate(**base_inp, max_new_tokens=len(generated), min_new_tokens=len(generated),
                     do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        baseline_s = time.perf_counter() - tb0
        baseline_tok_s_list.append(len(generated) / baseline_s)

    mean_accept = sum(all_accepts) / len(all_accepts)
    mean_round = sum(all_new_tokens) / sum(all_wall_s)
    mean_baseline = sum(baseline_tok_s_list) / len(baseline_tok_s_list)
    print(f"\nMEAN  accept_length={mean_accept:.3f} (over {len(all_accepts)} rounds)  "
          f"round={mean_round:.2f} tok/s  baseline={mean_baseline:.2f} tok/s  "
          f"speedup={mean_round/mean_baseline:.2f}x  (n={len(samples)})")
    if args.json:
        res = dict(ckpt=args.ckpt, dataset=args.dataset, n=len(samples), mode="round", draft_arch="eagle3",
                   top_k=args.top_k, tree_depth=args.tree_depth, total_token=args.total_token,
                   max_new_tokens=args.max_new_tokens, mean_accept_length=mean_accept,
                   mean_round_tok_s=mean_round, mean_baseline_tok_s=mean_baseline,
                   speedup=mean_round / mean_baseline, num_rounds=len(all_accepts),
                   per_round_accept=all_accepts)
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(res, open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")


def cmd_round(args):
    if args.draft_arch == "hivis":
        return cmd_round_hivis(args)
    return cmd_round_eagle3(args)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("run", cmd_run), ("profile", cmd_profile), ("tree", cmd_tree), ("round", cmd_round)):
        b = sub.add_parser(nm)
        b.add_argument("--ckpt", required=True)
        b.add_argument("--draft-arch", choices=["eagle3", "hivis"], default="eagle3", dest="draft_arch",
                        help="eagle3: Eagle3LlamaForCausalLM (multi-aux-hidden-state, draft-vocab remap). "
                             "hivis: HiViS-way EAGLE2-style (single last-hidden-state, no vocab remap).")
        b.add_argument("--target", default=DEFAULT_TARGET)
        b.add_argument("--dataset", default="MMMU/MMMU")
        b.add_argument("--prompt-style", choices=["raw", "answer_then_describe"], default="raw",
                        dest="prompt_style",
                        help="answer_then_describe lengthens VQA-style questions for more decode rounds.")
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
        if nm in ("tree", "round"):
            b.add_argument("--top-k", type=int, default=10, dest="top_k",
                            help="branching factor per tree-depth step (HiViS's --top-k; 1 = linear chain)")
            b.add_argument("--tree-depth", type=int, default=5, dest="tree_depth",
                            help="tree depth in speculative steps (HiViS's --depth)")
            b.add_argument("--total-token", type=int, default=60, dest="total_token",
                            help="beam width: nodes kept per depth step after pruning by cumulative log-prob "
                                 "(HiViS's --total-token, but pruned by score here instead of a fixed sparse topology)")
        if nm == "round":
            b.add_argument("--max-new-tokens", type=int, default=200, dest="max_new_tokens",
                            help="tokens to generate per sample across all rounds (HiViS's own "
                                 "--max-new-token default is also 200)")
        b.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
