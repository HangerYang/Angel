#!/usr/bin/env python3
"""Canonical table for the answer_then_describe (ATD) acceptance experiment.

Reads the `rerun_atd/` roots written by
scripts/speculative/smolvlm/run_atd_acceptance.sh and pairs every cell with its
raw-prompt counterpart from the temp0 sweep, so each number carries the
population (N) and the arm that produced it.

Fails loudly: a missing cell or an unexpected N aborts instead of silently
printing a dash, because this file is meant to be an experiment record.
"""
import json, os, sys, datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "my_angel", "atd-results.md")

EXPECTED_N = 80
TEMP = 0
STYLE = "answer_then_describe"
# Only the benchmarks whose raw output was too short for speculation to pay and
# that ATD actually lengthened; chartqa is excluded (no prompt moved it).
DS = ["MMStar", "MMMU", "textvqa", "mathvista"]

BASELINE = "no_eagle_baseline"
# name -> (raw-prompt root, ATD root, draft checkpoint or None)
RUNS = {
    BASELINE: (
        "my_angel/no_eagle_baseline/temp0",
        "my_angel/no_eagle_baseline/atd_temp0",
        None,
    ),
    "baseline_1layer": (
        "my_angel/eagle/baseline_1layer/rerun/temp0",
        "my_angel/eagle/baseline_1layer/rerun_atd/temp0",
        "my_angel/eagle/baseline_1layer/checkpoint-66466",
    ),
    "banded_mix_fc_3.1": (
        "my_angel/smolvlm-256m-eagle3-banded-mix-fc-3.1/rerun/temp0",
        "my_angel/smolvlm-256m-eagle3-banded-mix-fc-3.1/rerun_atd/temp0",
        "my_angel/smolvlm-256m-eagle3-banded-mix-fc-3.1/checkpoint-66466",
    ),
    "banded_mix_wide_3.1": (
        "my_angel/smolvlm-256m-eagle3-banded-mix-wide-3.1/rerun/temp0",
        "my_angel/smolvlm-256m-eagle3-banded-mix-wide-3.1/rerun_atd/temp0",
        "my_angel/smolvlm-256m-eagle3-banded-mix-wide-3.1/checkpoint-66466",
    ),
}

errors = []


def metrics(root, ds, what):
    f = os.path.join(REPO, root, ds, "results.json")
    if not os.path.exists(f):
        errors.append(f"missing cell: {what} -> {root}/{ds}/results.json")
        return None
    m = json.load(open(f))
    m = m.get("metrics", m)
    n = m.get("num_prompts")
    if n != EXPECTED_N:
        errors.append(f"unexpected N={n} (expected {EXPECTED_N}) for {what} -> {root}/{ds}")
    for key in ("avg_output_tokens", "output_throughput"):
        if m.get(key) is None:
            errors.append(f"incomplete cell: {what} -> {root}/{ds} has no {key}")
    return m


cells = {}
for name, (raw_root, atd_root, _) in RUNS.items():
    for ds in DS:
        cells[(name, "raw", ds)] = metrics(raw_root, ds, f"{name}/raw/{ds}")
        cells[(name, "atd", ds)] = metrics(atd_root, ds, f"{name}/atd/{ds}")

if errors:
    print("ATD results are incomplete — refusing to write a partial record:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)


def draft_layers(ckpt):
    if not ckpt:
        return "—"
    f = os.path.join(REPO, ckpt, "config.json")
    if os.path.exists(f):
        return str(json.load(open(f)).get("num_hidden_layers", "?"))
    return "?"


L = [
    "# answer_then_describe: does lengthening the output recover the speedup?", "",
    f"Generated {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}.", "",
    f"**Population**: N={EXPECTED_N} prompts per benchmark, temp {TEMP}, `output_len=1024`,",
    "`max_num_seqs=1`, K=4, `enforce_eager`, gpu-mem 0.8, one vLLM job at a time on GPU 0.", "",
    f"**Prompt styles**: `raw` (dataset question verbatim, image placeholders stripped as in",
    f"the raw harness) vs `{STYLE}`",
    '(`"Answer this question: {q} Then describe the image in detail to justify your answer."`).', "",
    "**Arms**: `target-only` = no draft (`USE_EAGLE=0`); `with-draft` = EAGLE-3 speculative",
    "decoding with the named draft. Note the two arms are **not** bit-identical at temp 0 in",
    "this build — on long outputs the speculative path diverges mid-sequence (COCO-Caption",
    "63/80 samples, chartqa 13/80; MMStar and MATH-500 0/80) — so a length taken from the",
    "target-only arm is not interchangeable with one taken from a draft arm.", "",
    "Benchmarks are the four whose raw output was too short for speculation to pay and that",
    "`answer_then_describe` lengthened. chartqa is excluded: no prompt style moved it.", "",
    "## Output length (target-only arm)", "",
    "| benchmark | N | arm | raw out tok | ATD out tok | length ratio |",
    "|---|---:|---|---:|---:|---:|",
]
for ds in DS:
    r = cells[(BASELINE, "raw", ds)]
    a = cells[(BASELINE, "atd", ds)]
    L.append(f"| {ds} | {EXPECTED_N} | target-only | {r['avg_output_tokens']:.1f} | "
             f"{a['avg_output_tokens']:.1f} | **{a['avg_output_tokens'] / r['avg_output_tokens']:.2f}x** |")

L += ["", "## Throughput and speedup (with-draft arm)", "",
      "Speedup is against `no_eagle_baseline` measured under the *same* prompt style, so the",
      "raw and ATD columns are each internally consistent.", "",
      "| benchmark | N | draft | layers | raw tok/s | ATD tok/s | raw speedup | ATD speedup | ATD accept |",
      "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
for ds in DS:
    for name, (_, _, ckpt) in RUNS.items():
        if name == BASELINE:
            continue
        r, a = cells[(name, "raw", ds)], cells[(name, "atd", ds)]
        br, ba = cells[(BASELINE, "raw", ds)], cells[(BASELINE, "atd", ds)]
        L.append(
            f"| {ds} | {EXPECTED_N} | `{name}` | {draft_layers(ckpt)} | "
            f"{r['output_throughput']:.1f} | {a['output_throughput']:.1f} | "
            f"{r['output_throughput'] / br['output_throughput']:.3f}x | "
            f"**{a['output_throughput'] / ba['output_throughput']:.3f}x** | "
            f"{a.get('mean_acceptance_length', float('nan')):.3f} |"
        )

L += ["", "## Baseline (target-only) throughput", "",
      "| benchmark | N | raw tok/s | ATD tok/s |", "|---|---:|---:|---:|"]
for ds in DS:
    L.append(f"| {ds} | {EXPECTED_N} | {cells[(BASELINE, 'raw', ds)]['output_throughput']:.1f} | "
             f"{cells[(BASELINE, 'atd', ds)]['output_throughput']:.1f} |")

L += ["", "Produced by `scripts/speculative/smolvlm/run_atd_acceptance.sh`;",
      "regenerate with `python my_angel/make_atd_results.py`."]

open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT} ({len(DS) * len(RUNS) * 2} cells, all present at N={EXPECTED_N})")
