#!/usr/bin/env python3
"""Drafting-speed harness — per-stage, per-component report.

Fixed verification-round count for every variant, so acceptance length never enters:
  target prefills once (UNTIMED) -> 32 rounds x K=4 drafted tokens, advanced
  unconditionally. No verification, no acceptance, no output text.

Per run it reports, averaged over all examples:
    PREFILL          total + per-component
    ROUND / token 1  total + per-component     (first token of a round: re-seeds
    ROUND / token 2  ...                        from the target hidden state)
    ROUND / token 3  ...
    ROUND / token 4  ...

Two timing passes:
  clean  - no hooks, CUDA-event timed  -> trustworthy totals
  hooked - per-module sync             -> component shares (absolute ms inflated)
"""
from __future__ import annotations
import argparse, json, os, warnings, collections, torch
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TARGET = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEV, DT = "cuda:0", torch.bfloat16
K, ROUNDS = 4, 32
OMNI_Q = "提取并识别图片中的文本。"
STAGE = {"cur": None}
ACC = collections.defaultdict(lambda: {"n": 0, "ms": 0.0})


def ev(): return torch.cuda.Event(enable_timing=True)


# ---- component classification -------------------------------------------------
def comp_of(name, mod):
    c = mod.__class__.__name__
    if "embed_tokens" in name: return "embed_tokens"
    if "lm_head" in name: return "lm_head (576x32k)"
    if name.startswith("fc") or name == "fc": return "fc (aux fuse)"
    if "fc_norm" in name: return "fc_norm (3.1)"
    if "q_proj" in name or "k_proj" in name or "v_proj" in name: return "attn qkv_proj"
    if "o_proj" in name: return "attn o_proj"
    if "rotary" in name: return "attn rotary"
    if c == "LlamaAttention": return "attn (total)"
    if "gate_proj" in name or "up_proj" in name: return "mlp gate/up"
    if "down_proj" in name: return "mlp down"
    if c == "LlamaMLP": return "mlp (total)"
    if c == "LlamaRMSNorm": return "rmsnorm"
    return None


def hook_all(m):
    hs = []
    def mk(tag, mod):
        st = {}
        def pre(_m, _i):
            if STAGE["cur"] is None: return
            torch.cuda.synchronize(); st["t"] = torch.cuda.Event(True); st["t"].record()
        def post(_m, _i, _o):
            if STAGE["cur"] is None or "t" not in st: return
            e = torch.cuda.Event(True); e.record(); torch.cuda.synchronize()
            a = ACC[(STAGE["cur"], tag)]; a["n"] += 1; a["ms"] += st["t"].elapsed_time(e)
        hs.append(mod.register_forward_pre_hook(pre))
        hs.append(mod.register_forward_hook(post))
    for name, mod in m.named_modules():
        t = comp_of(name, mod)
        if t and not any(ch for ch in mod.children()) or t in ("attn (total)", "mlp (total)"):
            if t: mk(t, mod)
    return hs


# ---- data / models ------------------------------------------------------------
def load_prompts(n):
    from datasets import load_dataset
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(TARGET)
    ds = load_dataset("opendatalab/OmniDocBench", split="train")
    out = []
    for i in range(n):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": OMNI_Q}]}]
        p = proc.apply_chat_template(msgs, add_generation_prompt=True)
        out.append(proc(text=p, images=[ds[i]["image"].convert("RGB")], return_tensors="pt").to(DEV))
    return out


@torch.no_grad()
def target_aux(tgt, inp, aux_ids):
    o = tgt(**inp, output_hidden_states=True)
    hs = o.hidden_states
    return torch.cat([hs[j + 1] for j in aux_ids], -1).to(DT), inp["input_ids"]


def build_draft(base, n_streams, fc_norm, norm_output):
    from angelslim.compressor.speculative.train.models.draft.llama_eagle3 import (
        Eagle3LlamaForCausalLM)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(base, trust_remote_code=True)
    cfg.num_hidden_layers = 1
    cfg.eagle_aux_injection_mode = "fused_fc"
    cfg.aux_hidden_states_layer_ids = list(range(n_streams))
    cfg.eagle_aux_hidden_state_layer_ids = [i + 1 for i in range(n_streams)]
    cfg.fc_norm, cfg.norm_output = fc_norm, norm_output
    for k in ("progressive_per_layer_fc", "progressive_fc_draft_feedback"):
        if hasattr(cfg, k): setattr(cfg, k, False)
    for k in ("eagle_aux_layer_bands", "multi_depth_ce_weights"):
        if hasattr(cfg, k): setattr(cfg, k, None)
    m = Eagle3LlamaForCausalLM(cfg).to(DEV, DT).eval()
    # Fresh construction leaves params/buffers uninitialised. Values don't matter for
    # timing, but garbage in d2t makes tok+d2t[tok] index out of range and the
    # embedding lookup device-asserts (surfaces as CUBLAS_STATUS_EXECUTION_FAILED).
    with torch.no_grad():
        for _, p in m.named_parameters():
            if not torch.isfinite(p).all(): p.normal_(0, 0.02)
        for _, b in m.named_buffers():
            if b.dtype in (torch.int32, torch.int64, torch.long): b.zero_()
            elif b.is_floating_point() and not torch.isfinite(b).all(): b.zero_()
    for l in m.layers:
        r = l.self_attn.rotary_emb
        r._set_cos_sin_cache(r.max_position_embeddings, DEV, DT)
    m._early_exit_threshold = -1.0
    return m


class Mixer(torch.nn.Module):
    def __init__(self, n_bands, per_band, H):
        super().__init__()
        self.nb, self.pb = n_bands, per_band
        self.w = torch.nn.Parameter(torch.randn(n_bands, per_band, dtype=DT) / per_band)
    def forward(self, aux):
        B, S, _ = aux.shape
        H = aux.shape[-1] // (self.nb * self.pb)
        return (aux.view(B, S, self.nb, self.pb, H) *
                self.w[None, None, :, :, None]).sum(3).reshape(B, S, self.nb * H)


class QFormer(torch.nn.Module):
    def __init__(self, n_q, dim, heads=9, mlp=True):
        super().__init__()
        self.n_q = n_q
        self.q = torch.nn.Parameter(torch.randn(1, n_q, dim, dtype=DT) * 0.02)
        self.ln_q = torch.nn.LayerNorm(dim, dtype=DT); self.ln_kv = torch.nn.LayerNorm(dim, dtype=DT)
        self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True, dtype=DT)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(dim, dtype=DT), torch.nn.Linear(dim, dim * 8 // 3, dtype=DT),
            torch.nn.GELU(), torch.nn.Linear(dim * 8 // 3, dim, dtype=DT)) if mlp else None
    def forward(self, img):
        q = self.ln_q(self.q.expand(img.shape[0], -1, -1))
        kv = self.ln_kv(img)
        o, _ = self.attn(q, kv, kv, need_weights=False)
        o = q + o
        return o + self.mlp(o) if self.mlp else o


# ---- the mock speculative loop -------------------------------------------------
@torch.no_grad()
def run_one(m, aux, ids, img_mask, mixer, qformer, drop_img, timed_stages):
    """One example: prefill + ROUNDS rounds of K drafted tokens.
    timed_stages=True -> set STAGE so hooks attribute per stage; else STAGE stays None.
    Returns dict of stage -> ms (clean CUDA-event timing)."""
    out = {}

    def prep(a, i):
        if drop_img:
            keep = ~img_mask.clone(); keep[-1] = True
            return a[:, keep], i[:, keep]
        if qformer is not None:
            a_img, a_txt = a[:, img_mask], a[:, ~img_mask]
            i_img, i_txt = i[:, img_mask][:, :qformer.n_q], i[:, ~img_mask]
            return torch.cat([qformer(a_img), a_txt], 1), torch.cat([i_img, i_txt], 1)
        return a, i

    # ---- PREFILL
    STAGE["cur"] = "prefill" if timed_stages else None
    s, e = ev(), ev(); torch.cuda.synchronize(); s.record()
    a, i = prep(aux, ids)
    if mixer is not None: a = mixer(a)
    h = m.combine_hidden_states(a)
    c = m.init_cache_hidden()
    n = a.shape[1]
    pos = torch.arange(n, device=DEV).unsqueeze(0)
    msk = torch.full((1, 1, n, n), float("-inf"), device=DEV, dtype=h.dtype).triu(1)
    h, c = m.encode_layers(m.embed_tokens(i).to(h.dtype), h, c, msk, pos, True)
    h0 = h[:, -1:]
    e.record(); torch.cuda.synchronize(); out["prefill"] = s.elapsed_time(e)

    # ---- ROUNDS
    # The draft KV persists from prefill and grows by K each round -- that IS
    # "advance 4 unconditionally". position_ids must be RELATIVE: attention adds
    # len(cache_hidden[0]) internally (`position_ids + lck`), so absolute positions
    # double-count and index the rotary table out of range.
    per_tok = [0.0] * K
    cache = c
    for r in range(ROUNDS):
        h = h0                                   # re-seed from the target hidden state
        tok = torch.zeros(1, 1, dtype=torch.long, device=DEV)
        for j in range(K):
            STAGE["cur"] = f"token {j+1}" if timed_stages else None
            s, e = ev(), ev(); torch.cuda.synchronize(); s.record()
            p = torch.zeros(1, 1, dtype=torch.long, device=DEV)
            mk = torch.zeros(1, 1, 1, 1, device=DEV, dtype=h.dtype)
            h, cache = m.encode_layers(m.embed_tokens(tok).to(h.dtype), h, cache, mk, p, True)
            tok = m.compute_logits(h).argmax(-1)
            tok = (tok + m.d2t[tok]).clamp_(0, m.embed_tokens.num_embeddings - 1)
            h = m.next_hidden_from_encode(h)
            if h.shape[1] > 1: h = h[:, -1:]
            e.record(); torch.cuda.synchronize(); per_tok[j] += s.elapsed_time(e)
    STAGE["cur"] = None
    for j in range(K): out[f"token {j+1}"] = per_tok[j] / ROUNDS
    out["kv"] = n
    return out


def report(label, clean, hooked, n_ex, fh):
    def w(x=""): print(x); fh.write(x + "\n")
    w(f"\n{'='*78}\n{label}\n{'='*78}")
    w(f"averaged over {n_ex} examples x {ROUNDS} rounds x K={K}   draft KV = {clean['kv']:.0f} tokens\n")
    round_ms = sum(clean[f"token {j+1}"] for j in range(K))
    w(f"{'stage':22s} {'ms':>9}   components (share of that stage)")
    w("-"*78)
    for st in ["prefill"] + [f"token {j+1}" for j in range(K)]:
        comps = {t: v for (s_, t), v in hooked.items() if s_ == st}
        tot = sum(v["ms"] for v in comps.values()) or 1.0
        top = sorted(comps.items(), key=lambda kv: -kv[1]["ms"])[:5]
        cs = "  ".join(f"{t} {100*v['ms']/tot:.0f}%" for t, v in top)
        w(f"{st:22s} {clean[st]:9.4f}   {cs}")
    w("-"*78)
    w(f"{'ROUND total':22s} {round_ms:9.4f}   ({round_ms/K:.4f} ms per drafted token)")
    w(f"{'128 drafted tokens':22s} {clean['prefill'] + ROUNDS*round_ms:9.4f}   "
      f"(prefill {clean['prefill']:.3f} + {ROUNDS} rounds)")
    return dict(variant=label, kv=clean["kv"],
                prefill_ms=clean["prefill"],
                per_token_ms={f"token {j+1}": clean[f"token {j+1}"] for j in range(K)},
                round_ms=round_ms, ms_per_drafted_token=round_ms / K,
                total_128_ms=clean["prefill"] + ROUNDS * round_ms,
                components={f"{s_}|{t}": v["ms"] for (s_, t), v in hooked.items()})


VARIANTS = [
    # label,                          fc_streams, layers/band, fc_norm, norm_out, mode
    ("1  plain eagle3 (3 layers, 3H)",        3, 1, False, False, "plain"),
    ("2  bands 3 -> 3H  (1 layer/band)",      3, 1, False, False, "mix"),
    ("2  bands 6 -> 3H  (2 layers/band)",     3, 2, False, False, "mix"),
    ("2  bands 9 -> 3H  (3 layers/band)",     3, 3, False, False, "mix"),
    ("2b 1 band  -> H",                       1, 1, False, False, "plain"),
    ("2b 2 bands -> 2H",                      2, 1, False, False, "plain"),
    ("2b 3 bands -> 3H  (= plain)",           3, 1, False, False, "plain"),
    ("3  HiViS: text-only draft prefill",     3, 1, False, False, "noimg"),
    ("4  QFormer 4x  (832 -> 208 img)",       3, 1, False, False, "qf4"),
    ("4  QFormer 8x  (832 -> 104 img)",       3, 1, False, False, "qf8"),
    ("4  QFormer 16x (832 ->  52 img)",       3, 1, False, False, "qf16"),
    ("4  QFormer 4x, no MLP",                 3, 1, False, False, "qf4nm"),
    ("5  eagle 3.1 (fc_norm + norm_output)",  3, 1, True,  True,  "plain"),
]
QF = {"qf4": (208, True), "qf8": (104, True), "qf16": (52, True), "qf4nm": (208, False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--only", default=None, help="substring filter on variant label")
    ap.add_argument("--outdir", default="/home/hyang/AngelSlim/my_angel/plan")
    ap.add_argument("--base", default="/home/hyang/AngelSlim/my_angel/eagle/baseline_1layer/checkpoint-66466")
    a = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoConfig
    prompts = load_prompts(a.n)
    tgt = AutoModelForImageTextToText.from_pretrained(TARGET, dtype=DT, device_map=DEV).eval()
    img_tok = getattr(tgt.config, "image_token_id", 49190)
    H = AutoConfig.from_pretrained(a.base, trust_remote_code=True).hidden_size

    os.makedirs(a.outdir, exist_ok=True)
    rows = []
    with open(f"{a.outdir}/report.txt", "w") as fh:
        fh.write(f"Draft-speed report  |  OmniDocBench  |  {a.n} examples  |  "
                 f"{ROUNDS} rounds x K={K} = {ROUNDS*K} drafted tokens per example\n"
                 f"Acceptance is not measured and does not affect anything: every variant "
                 f"runs the SAME number of rounds.\n"
                 f"'ms' columns are clean CUDA-event timings. Component shares come from a "
                 f"second hooked pass (absolute ms inflated by per-module sync).\n")
        for label, ns, pb, fcn, no, mode in VARIANTS:
            if a.only and a.only not in label: continue
            n_src = ns * pb
            m = build_draft(a.base, ns, fcn, no)
            mixer = Mixer(ns, pb, H).to(DEV, DT).eval() if (mode == "mix" and pb > 1) else None
            qf = None
            if mode in QF:
                nq, mlp = QF[mode]
                qf = QFormer(nq, ns * H, mlp=mlp).to(DEV, DT).eval()
            aux_ids = list(range(1, 1 + n_src))
            cache = [(target_aux(tgt, inp, aux_ids), (inp["input_ids"] == img_tok)[0]) for inp in prompts]

            for (aux, ids), mask in cache[:1]:                       # warmup
                run_one(m, aux, ids, mask, mixer, qf, mode == "noimg", False)
            clean = collections.defaultdict(float)
            for (aux, ids), mask in cache:
                r = run_one(m, aux, ids, mask, mixer, qf, mode == "noimg", False)
                for k, v in r.items(): clean[k] += v / len(cache)
            ACC.clear()
            for (aux, ids), mask in cache:
                run_one(m, aux, ids, mask, mixer, qf, mode == "noimg", True)
            hooked = {k: dict(v) for k, v in ACC.items()}
            rows.append(report(label, clean, hooked, len(cache), fh))
            del m, mixer, qf; torch.cuda.empty_cache()

        base = next((r["total_128_ms"] for r in rows if r["variant"].startswith("1 ")), None)
        if base:
            fh.write(f"\n\n{'='*78}\nSUMMARY vs plain eagle3\n{'='*78}\n")
            hdr = f"{'variant':40s} {'kv':>6} {'prefill':>9} {'round':>9} {'/tok':>8} {'128tok':>9} {'vs #1':>8}"
            print("\n" + hdr); fh.write(hdr + "\n" + "-"*90 + "\n"); print("-"*90)
            for r in rows:
                r["vs_plain"] = r["total_128_ms"] / base
                ln = (f"{r['variant']:40s} {r['kv']:6.0f} {r['prefill_ms']:9.4f} "
                      f"{r['round_ms']:9.4f} {r['ms_per_drafted_token']:8.4f} "
                      f"{r['total_128_ms']:9.3f} {r['vs_plain']:7.3f}x")
                print(ln); fh.write(ln + "\n")
    json.dump(rows, open(f"{a.outdir}/results.json", "w"), indent=1)
    print(f"\nwrote {a.outdir}/report.txt and results.json")


if __name__ == "__main__":
    main()
