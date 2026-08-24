#!/usr/bin/env python3
"""Per-run draft-speed report from the vLLM latency instrumentation."""
import json, re, os, sys, collections
PAT = re.compile(r"\[ANGELSLIM_EAGLE_LATENCY\] (\{.*\})")
NP, ROUNDS, K = 20, 32, 4

def parse(d):
    E = [json.loads(m.group(1)) for m in
         (PAT.search(l) for l in open(f"{d}/run.log", errors="replace")) if m]
    E = [e for e in E if e.get("num_tokens") != 8192]
    pre = collections.defaultdict(lambda: {"n": 0, "ms": 0.0})
    dec = collections.defaultdict(lambda: {"n": 0, "ms": 0.0})
    for e in E:
        t = pre if e.get("num_tokens", 0) > 100 else dec
        k = (e["event"], e.get("num_tokens"))
        t[k]["n"] += 1; t[k]["ms"] += e["ms"]
    return pre, dec

def report(name, d, cfgdesc, fh):
    pre, dec = parse(d)
    CY = dec[("draft_layer_forward", 5)]["n"] or 1
    w = lambda s: (print(s), fh.write(s + "\n"))
    w("=" * 74); w(f"{name}"); w("=" * 74)
    w(f"  {cfgdesc}")
    w(f"  OmniDocBench, {NP} prompts, 887 prompt tokens (832 image), K={K}, vLLM eager")
    w(f"  {CY} cycles observed -> normalised to {ROUNDS} rounds "
      f"(acceptance does not affect any number here)")
    w("")
    w("  PER PROMPT — draft prefill (887-token KV)")
    for (ev, nt), v in sorted(pre.items(), key=lambda kv: -kv[1]["ms"]):
        w(f"    {ev:24s} {v['ms']/NP:8.4f} ms")
    P = sum(v["ms"] for v in pre.values()) / NP
    w(f"    {'PREFILL TOTAL':24s} {P:8.4f} ms")
    w("")
    w("  PER ROUND (K=4 drafted tokens)")
    tot = 0.0
    for (ev, nt), v in sorted(dec.items(), key=lambda kv: -kv[1]["ms"]):
        if v["n"] / CY < 0.01: continue
        per = v["ms"] / CY; tot += per
        lab = "token 1 (seed+draft)" if nt == 5 else ("tokens 2-4" if nt == 1 else f"ntok={nt}")
        w(f"    {ev:24s} {lab:21s} x{v['n']/CY:4.2f} {per:8.4f} ms  ({1000*v['ms']/v['n']:6.1f} us/call)")
    w(f"    {'ROUND TOTAL':46s} {tot:8.4f} ms")
    w("")
    T = P + ROUNDS * tot
    w(f"  128 DRAFTED TOKENS   prefill {P:.4f} + {ROUNDS} rounds {ROUNDS*tot:.4f} = {T:.4f} ms")
    w(f"  per drafted token    {ROUNDS*tot/(ROUNDS*K):.4f} ms")
    w("")
    return dict(name=name, prefill_ms=P, round_ms=tot, total_128_ms=T,
                ms_per_tok=ROUNDS*tot/(ROUNDS*K), cycles=CY)

RUNS = [
    ("RUN 1   plain eagle3 (3 aux, 3H)",        "runs/01_plain_eagle3", "1-layer, fused_fc, fc_in=1728"),
    ("RUN 2   3 layers -> 3 bands (1/band)",    "runs/2_bands3_1per",   "aux=3, aux_band_mix=1 (no mix), fc_in=1728"),
    ("RUN 2   6 layers -> 3 bands (2/band)",    "runs/2_bands6_2per",   "aux=6, aux_band_mix=2, fc_in=1728"),
    ("RUN 2   9 layers -> 3 bands (3/band)",    "runs/2_bands9_3per",   "aux=9, aux_band_mix=3, fc_in=1728"),
    ("RUN 2b  1 band  -> H",                    "runs/2b_1band_H",      "aux=1, fc_in= 576"),
    ("RUN 2b  2 bands -> 2H",                   "runs/2b_2band_2H",     "aux=2, fc_in=1152"),
    ("RUN 2b  3 bands -> 3H (== run 1)",        "runs/2b_3band_3H",     "aux=3, fc_in=1728"),
    ("RUN 3   HiViS flag (NO EFFECT - see note)","runs/3_hivis",        "hivis_remove_image_tokens=true, ignored in fused_fc path"),
    ("RUN 5   eagle 3.1",                       "runs/5_eagle31",       "aux=3, fc_in=1728, fc_norm=ON, norm_output=ON"),
]
rows = []
with open("my_angel/plan/report.txt", "w") as fh:
    for name, d, desc in RUNS:
        p = f"my_angel/plan/{d}"
        if not os.path.exists(f"{p}/run.log"): continue
        r = report(name, p, desc, fh)
        rows.append(r)
        with open(f"{p}/report.txt", "w") as f2:
            report(name, p, desc, f2)
    base = rows[0]["total_128_ms"]
    hdr = f"{'variant':42s} {'prefill':>9} {'round':>9} {'/tok':>8} {'128tok':>9} {'vs run1':>9}"
    print("\n" + hdr); fh.write("\n" + hdr + "\n")
    print("-" * 90); fh.write("-" * 90 + "\n")
    for r in rows:
        ln = (f"{r['name']:42s} {r['prefill_ms']:9.4f} {r['round_ms']:9.4f} "
              f"{r['ms_per_tok']:8.4f} {r['total_128_ms']:9.3f} {r['total_128_ms']/base:8.3f}x")
        print(ln); fh.write(ln + "\n")
json.dump(rows, open("my_angel/plan/results.json", "w"), indent=1)
print("\nwrote my_angel/plan/report.txt  and  per-run report.txt")
