"""Render the 64-token comparison matrix; blank cells are runs not yet done."""
import csv, glob, json, os, sys

OUT = "/home/hyang/tmp/vek/out"
ARMS = [
    ("img832",      "832 tok  full resolution"),
    ("img320",      "320 tok  edge 1024"),
    ("global_only", " 64 tok  <global-img> thumbnail (surgery)"),
    ("img64",       " 64 tok  pixel downscale to 512px"),
    ("e1024_dp64",  " 64 tok  DivPrune @ edge 1024  (25% cover)"),
    ("dp64",        " 64 tok  DivPrune @ edge 2048  (8.3% cover)"),
]
DS = ["ChartQA_TEST", "OCRBench", "MMMU_DEV_VAL", "MMStar", "MME",
      "ScienceQA_VAL", "AI2D_TEST", "TextVQA_VAL", "POPE"]
SHORT = {"ChartQA_TEST": "ChartQA", "MMMU_DEV_VAL": "MMMU", "ScienceQA_VAL": "SciQA",
         "AI2D_TEST": "AI2D", "TextVQA_VAL": "TextVQA", "OCRBench": "OCRBench",
         "MMStar": "MMStar", "POPE": "POPE", "MME": "MME*"}


def score(arm, ds):
    """Percent for every benchmark, whatever shape its score file happens to be.

    VLMEvalKit is not consistent here: ChartQA/TextVQA write *_acc.csv, POPE and
    MME write *_score.csv, OCRBench writes JSON, and the MCQ sets report a
    fraction rather than a percentage. MME has no single overall column at all --
    its headline is perception + reasoning, out of 2800.
    """
    d = os.path.join(OUT, arm, "SmolVLM-256M")

    if ds == "OCRBench":
        for f in glob.glob(os.path.join(d, "*OCRBench*.json")):
            obj = json.load(open(f))
            v = obj.get("Final Score") or obj.get("Final Score Norm")
            if v is not None:
                return float(v) / 10.0        # OCRBench is scored out of 1000
        return None

    for pat in (f"*{ds}*acc.csv", f"*{ds}*score.csv"):
        for f in glob.glob(os.path.join(d, pat)):
            rows = list(csv.DictReader(open(f)))
            if not rows:
                continue
            if ds == "MME":
                r = rows[0]
                if "perception" in r and "reasoning" in r:
                    return (float(r["perception"]) + float(r["reasoning"])) / 28.0
            for row in rows:
                for k in ("Overall", "overall", "acc", "accuracy"):
                    if k in row:
                        try:
                            v = float(row[k])
                        except (TypeError, ValueError):
                            continue
                        return v * 100 if v <= 1.0 else v
    return None


def main():
    w = 9
    head = f"{'arm':<46}" + "".join(f"{SHORT[d]:>{w}}" for d in DS)
    print(head)
    print("-" * len(head))
    done = total = 0
    for arm, label in ARMS:
        line = f"{label:<46}"
        for d in DS:
            v = score(arm, d)
            total += 1
            done += v is not None
            line += f"{v:>{w}.2f}" if v is not None else f"{'·':>{w}}"
        print(line)
    print("-" * len(head))
    print(f"{done}/{total} cells  ({100*done//total}%)     · = not run yet")
    print("\nMMMU = validation split only. POPE/SciQA/AI2D/MMStar/MMMU are MCQ or Y/N")
    print("scored by exact match; ChartQA relaxed_accuracy, TextVQA vqa_score,")
    print("OCRBench its own metric. No judge model anywhere.")


if __name__ == "__main__":
    sys.exit(main())
