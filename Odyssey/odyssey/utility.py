"""Utility matching: does a branch still get the ANSWER right?

`analyze.drift_vs_baseline` compares generated strings, which is exact-match
(prefix agreement is just a graded exact-match). That answers "did the branch
reproduce the target's tokens", not "is the branch still useful". Those come
apart: a branch can diverge from the target's wording on the first token and
still land on the same answer, or track it for 90% of the string and then emit
a different final number.

This module scores the extracted answer instead:

  accuracy            branch answer vs. the benchmark's ground truth
  utility_agreement   branch answer vs. the BASELINE's answer -- how often the
                      resync changes the decision, regardless of correctness
  delta               accuracy(branch) - accuracy(baseline)

`delta` is the number that decides whether a branch is worth it. Agreement
matters separately: a branch could match baseline accuracy while flipping
individual answers in both directions, which is a different risk profile from
one that tracks it.

`--embed` adds a third view: cosine similarity between the branch's generation
and the baseline's, under a sentence encoder. That sits between the two -- it
is insensitive to wording the way exact match is not, but unlike answer
extraction it reads the whole generation, so it catches a branch that keeps the
answer and wrecks the justification after it. It cannot stand alone: two texts
that differ only in the final number score near 1.0.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

# Leading option letter, e.g. " D: The suitcase..." -> "D". MMStar answers are
# single letters, and the model echoes the option it picked.
_CHOICE = re.compile(r"\b([A-E])\s*[:.\)]")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def extract_choice(text: str) -> str | None:
    m = _CHOICE.search(text or "")
    if m:
        return m.group(1)
    # Fall back to a bare leading letter ("A", "B.") with nothing after it.
    m = re.match(r"\s*([A-E])\b", text or "")
    return m.group(1) if m else None


def extract_math(text: str) -> str | None:
    """Final answer for MATH-500: \\boxed{...} if present, else the last number.

    Both are weak extractors, but they are applied identically to every branch,
    so the *delta* stays meaningful even where the absolute accuracy is not.
    """
    t = text or ""
    boxed = _BOXED.findall(t)
    if boxed:
        return boxed[-1].strip()
    nums = _NUM.findall(t)
    return nums[-1] if nums else None


def _norm(s: str | None) -> str | None:
    if s is None:
        return None
    return re.sub(r"[\s$,]", "", s).strip().rstrip(".").lower() or None


def degenerate(text: str, window: int = 60, repeats: int = 3) -> bool:
    """Flag looping output: the same line repeated `repeats`+ times.

    Worth reporting separately -- if the target itself degenerates on a
    benchmark, no branch has utility there and the delta is measuring noise.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < repeats:
        return False
    counts = defaultdict(int)
    for ln in lines:
        counts[ln[:window]] += 1
    return max(counts.values()) >= repeats


def score_bench(root: str, branch: str, bench: str) -> dict | None:
    path = os.path.join(root, branch, bench, "results.json")
    if not os.path.exists(path):
        return None
    try:
        data = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return None
    rows = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return None

    extract = extract_math if "MATH" in bench.upper() else extract_choice
    out = []
    for r in rows:
        gen = str(r.get("generated_text", ""))
        gold = r.get("answer") or r.get("solution") or r.get("gt")
        out.append(
            {
                "pred": _norm(extract(gen)),
                "gold": _norm(str(gold)) if gold is not None else None,
                "degenerate": degenerate(gen),
            }
        )
    return {"rows": out}


def embed_similarity(
    root: str, branches: list[str], benches: list[str], baseline: str, model_id: str
) -> dict:
    """Cosine similarity of each branch's generations against the baseline's.

    Encodes once per (branch, bench) and pairs by prompt index. Reports the mean
    and the low tail: a high mean with a fat low tail means most prompts track
    and a few blow up, which a mean alone hides.
    """
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model_id)

    def _texts(branch, bench):
        path = os.path.join(root, branch, bench, "results.json")
        if not os.path.exists(path):
            return None
        try:
            data = json.loads(Path(path).read_text())
        except (json.JSONDecodeError, OSError):
            return None
        rows = data.get("results", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            return None
        return [str(r.get("generated_text", "")) for r in rows]

    out: dict = {}
    for bench in benches:
        base = _texts(baseline, bench)
        if not base:
            continue
        base_emb = enc.encode(base, normalize_embeddings=True, show_progress_bar=False)
        per = {}
        for br in branches:
            cur = _texts(br, bench)
            if not cur:
                continue
            n = min(len(cur), len(base))
            emb = enc.encode(
                cur[:n], normalize_embeddings=True, show_progress_bar=False
            )
            sims = [float((emb[i] * base_emb[i]).sum()) for i in range(n)]
            sims_sorted = sorted(sims)
            per[br] = {
                "n": n,
                "mean_cosine": round(sum(sims) / n, 4),
                "min_cosine": round(sims_sorted[0], 4),
                "p10_cosine": round(sims_sorted[max(0, n // 10 - 1)], 4),
                "median_cosine": round(sims_sorted[n // 2], 4),
                "frac_below_0.8": round(sum(1 for x in sims if x < 0.8) / n, 4),
                "frac_below_0.5": round(sum(1 for x in sims if x < 0.5) / n, 4),
            }
        out[bench] = per
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed", action="store_true", help="add embedding similarity vs baseline")
    ap.add_argument(
        "--embed_model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    args = ap.parse_args()

    branches = sorted(
        os.path.basename(d)
        for d in glob.glob(os.path.join(args.results_root, "*"))
        if os.path.isdir(d)
    )
    benches = sorted(
        {
            os.path.basename(os.path.dirname(p))
            for p in glob.glob(os.path.join(args.results_root, "*", "*", "results.json"))
        }
    )

    report: dict = {}
    for bench in benches:
        base = score_bench(args.results_root, args.baseline, bench)
        if base is None:
            continue
        base_rows = base["rows"]
        per_branch = {}
        for br in branches:
            cur = score_bench(args.results_root, br, bench)
            if cur is None:
                continue
            rows = cur["rows"]
            n = min(len(rows), len(base_rows))
            if n == 0:
                continue
            scored = [i for i in range(n) if rows[i]["gold"] is not None]
            correct = sum(
                1 for i in scored if rows[i]["pred"] is not None
                and rows[i]["pred"] == rows[i]["gold"]
            )
            agree = sum(
                1 for i in range(n)
                if rows[i]["pred"] == base_rows[i]["pred"]
            )
            no_answer = sum(1 for i in range(n) if rows[i]["pred"] is None)
            degen = sum(1 for i in range(n) if rows[i]["degenerate"])
            per_branch[br] = {
                "n": n,
                "scored": len(scored),
                "accuracy": correct / len(scored) if scored else None,
                "utility_agreement": agree / n,
                "no_answer_rate": no_answer / n,
                "degenerate_rate": degen / n,
            }
        b_acc = per_branch.get(args.baseline, {}).get("accuracy")
        for br, m in per_branch.items():
            m["accuracy_delta"] = (
                None if (b_acc is None or m["accuracy"] is None)
                else round(m["accuracy"] - b_acc, 4)
            )
        report[bench] = per_branch

    if args.embed:
        emb = embed_similarity(
            args.results_root, branches, benches, args.baseline, args.embed_model
        )
        for bench, per in emb.items():
            for br, m in per.items():
                if bench in report and br in report[bench]:
                    report[bench][br].update(m)

    Path(os.path.dirname(os.path.abspath(args.out))).mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for bench, per in report.items():
        print(f"\n=== {bench} ===")
        has_emb = any("mean_cosine" in m for m in per.values())
        hdr = f"{'branch':<12} {'acc':>7} {'delta':>7} {'agree':>7} {'no_ans':>7} {'degen':>7}"
        if has_emb:
            hdr += f" {'cos':>7} {'cos_p10':>8} {'<0.8':>6}"
        print(hdr)
        for br in sorted(per):
            m = per[br]
            acc = "n/a" if m["accuracy"] is None else f"{m['accuracy']:.3f}"
            dl = "n/a" if m["accuracy_delta"] is None else f"{m['accuracy_delta']:+.3f}"
            line = (
                f"{br:<12} {acc:>7} {dl:>7} {m['utility_agreement']:>7.3f} "
                f"{m['no_answer_rate']:>7.3f} {m['degenerate_rate']:>7.3f}"
            )
            if has_emb and "mean_cosine" in m:
                line += (
                    f" {m['mean_cosine']:>7.3f} {m['p10_cosine']:>8.3f} "
                    f"{m['frac_below_0.8']:>6.3f}"
                )
            print(line)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
