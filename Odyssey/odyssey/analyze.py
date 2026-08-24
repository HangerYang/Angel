"""End-of-run analysis for the Odyssey branch sweep.

Produces, per branch:
  - tau (accepted tokens per verification round), mean and distribution
  - salvage yield: how many post-rejection draft tokens each branch kept
  - wall clock from the eval driver's acceptance_metrics.json
  - entropy-vs-outcome scatter data for threshold tuning
  - task correctness delta vs. the baseline branch
  - drift vs. baseline: exact-match rate and token-level agreement

Reads the JSONL emitted by odyssey.events plus the per-benchmark
acceptance_metrics.json / results.json the eval driver already writes.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path


def load_rounds(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "round":
                    rows.append(r)
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rounds: list[dict]) -> dict:
    """Per-branch aggregates. tau is the throughput number; salvage is the
    thing the stale branches exist to produce."""
    by = defaultdict(list)
    for r in rounds:
        by[r.get("branch", "?")].append(r)

    out = {}
    for branch, rs in sorted(by.items()):
        rejected = [r for r in rs if r.get("first_reject", -1) >= 0]
        # Rounds where a tail actually existed to salvage from -- the only ones
        # where the branches can differ at all.
        salvageable = [
            r for r in rejected
            if r.get("first_reject", -1) + 1 < r.get("num_spec", 0)
        ]
        out[branch] = {
            "rounds": len(rs),
            "mean_tau": _mean([r["tau"] for r in rs]),
            "reject_rate": len(rejected) / len(rs) if rs else float("nan"),
            "mean_first_reject": _mean([r["first_reject"] for r in rejected]),
            "salvageable_rounds": len(salvageable),
            "mean_salvage": _mean([r.get("salvaged", 0) for r in salvageable]),
            "total_salvaged": sum(r.get("salvaged", 0) for r in rs),
            "mean_reject_entropy": _mean([r.get("reject_entropy") for r in rejected]),
            "exact": bool(rs[0].get("exact", False)) if rs else None,
        }
    return out


def entropy_scatter(rounds: list[dict], bins: int = 8) -> dict:
    """Entropy at the rejected position vs. whether the stale tail survived.

    This is the threshold-tuning evidence: if salvage yield falls off with
    entropy, gate stale-reuse on entropy; if it is flat, entropy gating buys
    nothing.
    """
    by = defaultdict(list)
    for r in rounds:
        if r.get("first_reject", -1) < 0:
            continue
        h = r.get("reject_entropy")
        if h is None or (isinstance(h, float) and math.isnan(h)):
            continue
        by[r.get("branch", "?")].append((float(h), int(r.get("salvaged", 0))))

    out = {}
    for branch, pairs in sorted(by.items()):
        if not pairs:
            continue
        hs = [h for h, _ in pairs]
        lo, hi = min(hs), max(hs)
        width = (hi - lo) / bins if hi > lo else 1.0
        buckets = defaultdict(list)
        for h, s in pairs:
            b = min(int((h - lo) / width), bins - 1)
            buckets[b].append(s)
        out[branch] = [
            {
                "entropy_lo": round(lo + b * width, 4),
                "entropy_hi": round(lo + (b + 1) * width, 4),
                "n": len(buckets[b]),
                "mean_salvage": _mean(buckets[b]),
            }
            for b in sorted(buckets)
        ]
    return out


def compare_to_recompute(rounds: list[dict], rescored_path: str | None) -> dict:
    """How well does stale-reuse approximate a fresh target forward?

    Joins the stale branches' salvage counts against the offline recompute
    control on the same (round, req) key. Over-salvage means the stale
    distribution accepted tokens the target would have rejected -- the actual
    correctness risk.
    """
    if not rescored_path or not os.path.exists(rescored_path):
        return {}
    fresh = {}
    with open(rescored_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fresh[(r.get("round"), r.get("req"))] = r

    out = {}
    by = defaultdict(list)
    for r in rounds:
        if r.get("first_reject", -1) < 0:
            continue
        key = (r.get("round"), r.get("req"))
        if key not in fresh:
            continue
        by[r.get("branch", "?")].append((r.get("salvaged", 0), fresh[key]["fresh_salvage"]))

    for branch, pairs in sorted(by.items()):
        if not pairs:
            continue
        over = sum(1 for s, f in pairs if s > f)
        under = sum(1 for s, f in pairs if s < f)
        exact = sum(1 for s, f in pairs if s == f)
        out[branch] = {
            "matched_events": len(pairs),
            "mean_stale_salvage": _mean([s for s, _ in pairs]),
            "mean_fresh_salvage": _mean([f for _, f in pairs]),
            "over_salvage_rate": over / len(pairs),
            "under_salvage_rate": under / len(pairs),
            "exact_agreement_rate": exact / len(pairs),
        }
    return out


def entropy_vs_control(rounds: list[dict], rescored_path: str | None, bins: int = 6) -> dict:
    """Does entropy predict when stale-reuse goes WRONG?

    `entropy_scatter` bins salvage yield, which says how much a branch keeps --
    not whether keeping it was correct. This bins the disagreement against the
    recompute control instead: over_salvage is the rate at which the stale
    distribution accepted tokens a fresh target forward would have rejected.
    That is the quantity an entropy gate would have to predict to be worth
    adding.
    """
    if not rescored_path or not os.path.exists(rescored_path):
        return {}
    fresh = {}
    with open(rescored_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fresh[(r.get("round"), r.get("req"))] = r["fresh_salvage"]

    by = defaultdict(list)
    for r in rounds:
        if r.get("first_reject", -1) < 0:
            continue
        key = (r.get("round"), r.get("req"))
        if key not in fresh:
            continue
        h = r.get("reject_entropy")
        if h is None or (isinstance(h, float) and math.isnan(h)):
            continue
        by[r.get("branch", "?")].append(
            (float(h), int(r.get("salvaged", 0)), int(fresh[key]))
        )

    out = {}
    for branch, triples in sorted(by.items()):
        if not triples:
            continue
        hs = [h for h, _, _ in triples]
        lo, hi = min(hs), max(hs)
        width = (hi - lo) / bins if hi > lo else 1.0
        buckets = defaultdict(list)
        for h, s, f in triples:
            b = min(int((h - lo) / width), bins - 1)
            buckets[b].append((s, f))
        out[branch] = [
            {
                "entropy_lo": round(lo + b * width, 4),
                "entropy_hi": round(lo + (b + 1) * width, 4),
                "n": len(buckets[b]),
                "mean_stale_salvage": _mean([s for s, _ in buckets[b]]),
                "mean_fresh_salvage": _mean([f for _, f in buckets[b]]),
                "over_salvage_rate": sum(1 for s, f in buckets[b] if s > f)
                / len(buckets[b]),
            }
            for b in sorted(buckets)
        ]
    return out


def load_bench_metrics(root: str) -> dict:
    """Wall clock and vLLM's own acceptance numbers, per branch per benchmark."""
    out = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "acceptance_metrics.json"))):
        parts = Path(path).parts
        branch, bench = parts[-3], parts[-2]
        try:
            m = json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            continue
        out[branch][bench] = {
            "mean_acceptance_length": m.get("mean_acceptance_length"),
            "draft_acceptance_rate": m.get("draft_acceptance_rate"),
            "total_time": m.get("total_time"),
            "output_throughput": m.get("output_throughput"),
            "avg_output_tokens": m.get("avg_output_tokens"),
        }
    return dict(out)


def drift_vs_baseline(root: str, baseline: str = "baseline") -> dict:
    """Token-level drift of each branch's generations against the baseline's.

    The exactness claim these schemes need is distributional, which needs
    repeated sampling at temp > 0. This is the cheaper greedy proxy: at temp 0
    the baseline output IS the target's own output, so any divergence is drift
    introduced by the branch.
    """
    def _texts(branch):
        got = {}
        for path in sorted(glob.glob(os.path.join(root, branch, "*", "results.json"))):
            bench = Path(path).parts[-2]
            try:
                data = json.loads(Path(path).read_text())
            except (json.JSONDecodeError, OSError):
                continue
            rows = data.get("results", data) if isinstance(data, dict) else data
            if isinstance(rows, list):
                got[bench] = [str(r.get("generated_text", "")) for r in rows]
        return got

    base = _texts(baseline)
    if not base:
        return {}
    out = {}
    for branch_dir in sorted(glob.glob(os.path.join(root, "*"))):
        branch = os.path.basename(branch_dir)
        if branch == baseline or not os.path.isdir(branch_dir):
            continue
        cur = _texts(branch)
        per_bench = {}
        for bench, ref in base.items():
            hyp = cur.get(bench)
            if not hyp:
                continue
            n = min(len(ref), len(hyp))
            if n == 0:
                continue
            exact = sum(1 for i in range(n) if ref[i] == hyp[i])
            # Prefix agreement in characters: where does the branch first
            # diverge from what the target would have said?
            pref = []
            for i in range(n):
                a, b = ref[i], hyp[i]
                j = 0
                while j < min(len(a), len(b)) and a[j] == b[j]:
                    j += 1
                pref.append(j / max(len(a), 1))
            per_bench[bench] = {
                "n": n,
                "exact_match_rate": exact / n,
                "mean_prefix_agreement": _mean(pref),
            }
        if per_bench:
            out[branch] = per_bench
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="glob for round JSONL, e.g. 'logs/*.jsonl.*'")
    ap.add_argument("--results_root", default=None, help="dir holding <branch>/<bench>/")
    ap.add_argument("--rescored", default=None, help="rescore.py output for branch 3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rounds = load_rounds(args.events)
    report = {
        "num_round_events": len(rounds),
        "per_branch": summarize(rounds),
        "entropy_scatter": entropy_scatter(rounds),
        "vs_recompute": compare_to_recompute(rounds, args.rescored),
        "entropy_vs_control": entropy_vs_control(rounds, args.rescored),
    }
    if args.results_root:
        report["bench_metrics"] = load_bench_metrics(args.results_root)
        report["drift_vs_baseline"] = drift_vs_baseline(args.results_root)

    Path(os.path.dirname(os.path.abspath(args.out))).mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["per_branch"], indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
