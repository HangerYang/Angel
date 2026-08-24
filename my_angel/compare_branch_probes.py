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
LO, HI = 33233, 38233
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
steps = sorted(s for s in base_h if LO < s <= HI)


def window(h, lo, hi):
    ss = [s for s in steps if lo < s <= hi and s in h]
    if not ss:
        return None, 0
    return sum(ce(h[s]) for s in ss) / len(ss), len(ss)


bands = [(LO + i * 1000, min(LO + (i + 1) * 1000, HI)) for i in range(5)]
w = max(len(l) for l, _, _ in H) + 1

print(f"absolute train CE, mean(ploss_0..6), per 1k steps from {LO}\n")
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
print("It reads the 5k window only; the full-run gap was still widening at 66k.")
