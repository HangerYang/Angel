#!/usr/bin/env python3
"""Per-1k train-CE comparison over the 33233 -> 38233 window.

Every arm here resumed the SAME banded_mix_fc_3.1 checkpoint-33233 and sees the
same data in the same order, so differences at a matched step are paired and the
per-batch noise -- which is far larger than the between-arm gap -- cancels.

Judge on train CE = mean(train/ploss_0..6), NOT the reported `loss`: that folds
in branch_weight * branch_loss and so is not comparable across different w.

Across the 8 architecturally comparable runs evaluated so far,
    tau_eval = 3.8242 - 2.0743 * trainCE     (pearson -0.993, resid RMS 0.0052)
so a CE delta of -0.010 is worth about +0.021 acceptance length. The
`tau_pred` column applies that fit; it is family-local and does not transfer to
other draft architectures.
"""
import json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LO = 33233
HI = int(os.environ.get("HI", 0)) or None   # default: deepest step every arm reached
BAND = int(os.environ.get("BAND", 0)) or None  # default: ~5 bands across the window
FIT_A, FIT_B = 3.8242, -2.0743

# label -> run dir under my_angel/eagle
ARMS = [
    ("baseline (no branch)", "smolvlm-256m-eagle3-banded-mix-fc-3.1"),
    ("top2 w.1 RAMP",        "branch-change-top2-curr-r33k"),
    ("top2 w.1 ramp+synth",  "branch-change-top2-curr-synth-r33k"),
    ("A top2 w.1 no-ramp",   "branch-change-probeA-top2-w01-r33k"),
    ("B top3 w.1 no-ramp",   "branch-change-probeB-top3-w01-r33k"),
    ("C top3 w.3 no-ramp",   "branch-change-probeC-top3-w03-r33k"),
]


def history(run):
    """Newest checkpoint's trainer_state carries the whole run's log_history."""
    d = os.path.join(REPO, "my_angel", "eagle", run)
    ckpts = sorted(
        (int(x.split("-")[1]) for x in os.listdir(d) if x.startswith("checkpoint-")),
        reverse=True,
    ) if os.path.isdir(d) else []
    for c in ckpts:
        p = os.path.join(d, f"checkpoint-{c}", "trainer_state.json")
        if os.path.exists(p):
            h = {e["step"]: e for e in json.load(open(p))["log_history"]}
            if any(s > LO for s in h):
                return h
    return None


def ce(e):
    v = [e[f"train/ploss_{i}"] for i in range(7) if f"train/ploss_{i}" in e]
    return sum(v) / len(v) if v else None


H = [(lab, run, history(run)) for lab, run in ARMS]
missing = [run for lab, run, h in H if h is None]
H = [(lab, run, h) for lab, run, h in H if h is not None]
if missing:
    print(f"not yet trained, skipped: {', '.join(missing)}\n", file=sys.stderr)

base_lab, base_run, base_h = H[0]

# Compare only where EVERY arm has data, so the bands stay paired.
probe_max = [max(s for s in h) for lab, run, h in H[1:]] or [max(base_h)]
HI = HI or min(min(probe_max), max(base_h))
steps = sorted(s for s in base_h if LO < s <= HI)
span = HI - LO
BAND = BAND or max(1000, round(span / 5 / 1000) * 1000)


def window(h, lo, hi):
    ss = [s for s in steps if lo < s <= hi and s in h]
    if not ss:
        return None, 0
    return sum(ce(h[s]) for s in ss) / len(ss), len(ss)


nb = max(1, -(-span // BAND))
bands = [(LO + i * BAND, min(LO + (i + 1) * BAND, HI)) for i in range(nb)]
bands = [(lo, hi) for lo, hi in bands if hi > lo]
w = max(len(l) for l, _, _ in H) + 1

print(f"absolute train CE, mean(ploss_0..6), {BAND//1000}k bands, "
      f"{LO} -> {HI} ({span} branch-on steps)\n")
print(" " * w + "".join(f"{f'{lo//1000}-{hi//1000}k':>10}" for lo, hi in bands) + f"{'ALL':>10}")
for lab, run, h in H:
    row = "".join((f"{window(h,lo,hi)[0]:>10.4f}" if window(h, lo, hi)[0] else f"{'-':>10}")
                  for lo, hi in bands)
    a, _ = window(h, LO, HI)
    print(f"{lab:<{w}}{row}{a:>10.4f}" if a else f"{lab:<{w}}{row}{'-':>10}")

print(f"\npaired delta vs {base_lab} (negative = better)\n")
print(" " * w + "".join(f"{f'{lo//1000}-{hi//1000}k':>10}" for lo, hi in bands)
      + f"{'ALL':>10}{'tau_pred':>10}")
for lab, run, h in H[1:]:
    cells = ""
    for lo, hi in bands:
        a, _ = window(h, lo, hi)
        b, _ = window(base_h, lo, hi)
        cells += f"{a-b:>+10.4f}" if (a and b) else f"{'-':>10}"
    a, _ = window(h, LO, HI)
    b, _ = window(base_h, LO, HI)
    if a and b:
        print(f"{lab:<{w}}{cells}{a-b:>+10.4f}{FIT_B*(a-b):>+10.4f}")
    else:
        print(f"{lab:<{w}}{cells}{'-':>10}{'-':>10}")

print("\ntau_pred = projected acceptance-length gain at this CE gap, from the n=8 fit.")
print(f"It reads {LO}-{HI} only. The paired gap is not stationary -- on top2-curr-r33k")
print("it ran -0.0058 over 33k-38k and -0.0277 over 63k-66k -- so a window short of")
print("66466 understates where the arm would land. Compare arms, not absolutes.")
