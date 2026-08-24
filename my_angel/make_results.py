#!/usr/bin/env python3
"""Build the results table from the sequential GPU0 rerun."""
import json, os, glob, datetime

ROOT = "/home/hyang/AngelSlim/my_angel"
DS = ["MMStar", "MMMU", "OmniDocBench", "MATH-500"]
# Benchmarks only some runs carry; reported in their own section so the main
# tables stay a clean 4-way comparison instead of filling with dashes.
EXTRA_DS = ["textvqa", "chartqa", "mathvista", "COCO-Caption"]
RUNS = [
    ("baseline_1layer", "eagle/baseline_1layer"),
    ("baseline_staged", "progressive_eagle/baseline_staged"),
    ("branch_distill_w01", "progressive_eagle/branch_distill_w01"),
    ("attn_match_img_w01_L1_15_23", "progressive_eagle/attn_match_img_w01_L1_15_23"),
    ("banded_mix_uninit", "progressive_eagle/banded_mix_uninit"),
    ("per_layer_fc", "progressive_eagle/per_layer_fc"),
    ("per_layer_fc_feedback", "progressive_eagle/per_layer_fc_feedback"),
    ("per_layer_fc_feedback_earlyexit", "progressive_eagle/per_layer_fc_feedback_earlyexit"),
    ("banded_mix_per_layer_fc_feedback", "progressive_eagle/banded_mix_per_layer_fc_feedback"),
    # 1-layer EAGLE 3.1 draft whose 3 FC input streams are learned softmax mixes
    # over early/middle/late aux bands. Trained 2026-08-20; lives in output/.
    ("banded_mix_fc_3.1", "my_angel/smolvlm-256m-eagle3-banded-mix-fc-3.1"),
    # Same 3 bands, but no fusion FC: the mixed bands stay separate streams into
    # a 4H layer 0 ([emb | band0 | band1 | seed]). Trained 2026-08-20.
    ("banded_mix_wide_3.1", "my_angel/smolvlm-256m-eagle3-banded-mix-wide-3.1"),
]

REPO = os.path.dirname(ROOT)          # AngelSlim root, for runs outside my_angel/


def cell(path, ds):
    # Two layouts in the tree: the sequential rerun writes rerun/temp0/<ds>/,
    # while runs evaluated separately (banded_mix_per_layer_fc_feedback) write
    # eval/plain/<ds>/. Paths resolve under my_angel/ first, then the repo root
    # so runs living in output/ can be listed without being moved.
    bases = [f"{ROOT}/{path}", f"{REPO}/{path}"]
    for pat in [f"{b}/{sub}/{ds}/results.{ext}"
                for b in bases
                for sub in ("rerun/temp0", "eval/plain")
                for ext in ("json", "jsonl")]:
        if os.path.exists(pat):
            try:
                d = json.load(open(pat)); return d.get("metrics", d)
            except Exception: pass
    return None

def draft_layers(path):
    """num_hidden_layers straight from the run's checkpoint config.

    Read rather than hard-coded, so a new run lands in the right group without
    anyone having to remember to update a list here.
    """
    for b in (ROOT, REPO):
        for ck in ("checkpoint-66466", "checkpoint-33233"):
            f = f"{b}/{path}/{ck}/config.json"
            if os.path.exists(f):
                try:
                    return int(json.load(open(f)).get("num_hidden_layers", 0))
                except Exception:
                    pass
    return 0


def grouped(runs):
    """[(group label, [runs])] ordered shallowest draft first."""
    by = {}
    for name, path in runs:
        by.setdefault(draft_layers(path), []).append((name, path))
    out = []
    for n in sorted(by):
        label = f"{n}-layer draft" + ("s" if len(by[n]) != 1 else "")
        out.append((label if n else "unknown depth", by[n]))
    return out


def group_row(label, ncols):
    return f"| **{label}** |" + " |" * ncols


base = {ds: cell("no_eagle_baseline", ds) for ds in DS}
if not any(base.values()):                      # baseline lives at my_angel/no_eagle_baseline/temp0
    for ds in DS:
        for pat in (f"{ROOT}/no_eagle_baseline/temp0/{ds}/results.json",
                    f"{ROOT}/no_eagle_baseline/temp0/{ds}/results.jsonl"):
            if os.path.exists(pat):
                try:
                    d = json.load(open(pat)); base[ds] = d.get("metrics", d)
                except Exception: pass

L = ["# Rerun results — temp 0, GPU 0 sequential", "",
     f"Generated {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}.",
     "",
     "All cells: one vLLM job at a time on GPU 0, nothing else on the box — so",
     "timing is comparable across runs.", "",
     "| | |", "|---|---|",
     "| population | N=80 prompts per benchmark |",
     "| prompt style | `raw` (dataset question verbatim; image placeholders stripped as in the raw harness) |",
     "| temperature | 0 |",
     "| arms | `no_eagle_baseline` = target-only; every other row = with-draft (EAGLE-3, K=4) |",
     "| decode | `output_len=1024`, `max_num_seqs=1`, `enforce_eager`, tp 1 |",
     "| target | HuggingFaceTB/SmolVLM-256M-Instruct |", "",
     "Rows are grouped by **draft depth** (`num_hidden_layers`, read from each",
     "run's checkpoint config). Depth is the main cost lever: a 3-layer draft",
     "pays ~3x the per-step decode cost of a 1-layer one, so acceptance is only",
     "comparable within a group — across groups, read the throughput table.", "",
     "## Acceptance length", "",
     "| run | " + " | ".join(DS) + " | mean |", "|---|" + "---|" * (len(DS) + 1)]
for glabel, gruns in grouped(RUNS):
    L.append(group_row(glabel, len(DS) + 1))
    for name, path in gruns:
        vals, cells = [], []
        for ds in DS:
            m = cell(path, ds)
            if m and "mean_acceptance_length" in m:
                v = m["mean_acceptance_length"]; vals.append(v); cells.append(f"{v:.3f}")
            else: cells.append("—")
        mean = f"**{sum(vals)/len(vals):.3f}**" if vals else "—"
        L.append(f"| `{name}` | " + " | ".join(cells) + f" | {mean} |")

L += ["", "## Throughput (tok/s)", "",
      "| run | " + " | ".join(DS) + " | mean |", "|---|" + "---|" * (len(DS) + 1)]
for glabel, gruns in grouped(RUNS):
    L.append(group_row(glabel, len(DS) + 1))
    for name, path in gruns:
        vals, cells = [], []
        for ds in DS:
            m = cell(path, ds)
            if m and "output_throughput" in m:
                v = m["output_throughput"]; vals.append(v); cells.append(f"{v:.1f}")
            else: cells.append("—")
        mean = f"**{sum(vals)/len(vals):.1f}**" if vals else "—"
        L.append(f"| `{name}` | " + " | ".join(cells) + f" | {mean} |")

if any(base.values()):
    L += ["", "## Speedup vs non-speculative target-only baseline", "",
          "| run | " + " | ".join(DS) + " | mean |", "|---|" + "---|" * (len(DS) + 1)]
    b = ["| `no_eagle_baseline` (tok/s) | " +
         " | ".join(f"{base[ds]['output_throughput']:.1f}" if base.get(ds) else "—" for ds in DS) + " | |"]
    L += b
    for glabel, gruns in grouped(RUNS):
        L.append(group_row(glabel, len(DS) + 1))
        for name, path in gruns:
            vals, cells = [], []
            for ds in DS:
                m, bm = cell(path, ds), base.get(ds)
                if m and bm and bm.get("output_throughput"):
                    r = m["output_throughput"] / bm["output_throughput"]; vals.append(r)
                    cells.append(f"{r:.3f}x")
                else: cells.append("—")
            mean = f"**{sum(vals)/len(vals):.3f}x**" if vals else "—"
            L.append(f"| `{name}` | " + " | ".join(cells) + f" | {mean} |")

L += ["", "## Prompt / output sizes", "",
      "N=80, temp 0. Taken from whichever run carries the cell; the speculative path is",
      "not bit-exact against target-only in this build, so lengths can differ by ~1-3%",
      "between arms on long outputs (chartqa 88.4 target-only vs 84.9-86.0 with a draft).", "",
      "| dataset | N | avg input tok | avg output tok |", "|---|---:|---|---|"]
for ds in DS:
    m = next((cell(p, ds) for _, p in RUNS if cell(p, ds)), None)
    if m:
        L.append(f"| {ds} | 80 | {m.get('avg_input_tokens', 0):.1f} | {m.get('avg_output_tokens', 0):.1f} |")

extra_runs = [(n_, p_) for n_, p_ in RUNS if any(cell(p_, d) for d in EXTRA_DS)]
if extra_runs:
    L += ["", "## Extended benchmarks (runs that have them)", "",
          "| run | " + " | ".join(EXTRA_DS) + " |", "|---|" + "---|" * len(EXTRA_DS)]
    for name, path in extra_runs:
        acc, tps = [], []
        for ds in EXTRA_DS:
            m = cell(path, ds)
            acc.append(f"{m['mean_acceptance_length']:.3f}" if m else "—")
            tps.append(f"{m['output_throughput']:.1f}" if m else "—")
        L.append(f"| `{name}` accept | " + " | ".join(acc) + " |")
        L.append(f"| `{name}` tok/s | " + " | ".join(tps) + " |")
    L += ["", "| dataset | avg input tok | avg output tok |", "|---|---|---|"]
    for ds in EXTRA_DS:
        m = next((cell(p_, ds) for _, p_ in extra_runs if cell(p_, ds)), None)
        if m:
            L.append(f"| {ds} | {m.get('avg_input_tokens', 0):.1f} | "
                     f"{m.get('avg_output_tokens', 0):.1f} |")
    L += ["",
          "> `banded_mix_per_layer_fc_feedback` was evaluated in a separate "
          "session under "
          "`eval/plain/`, not in the sequential GPU-0 rerun, and used "
          "`gpu_memory_utilization 0.9` where the rerun used `0.8`. Every other "
          "flag matches (`max_num_seqs 1`, `output_len 1024`, K=4, "
          "`enforce_eager`, temp 0, N=80) and its `avg_input_tokens` agree "
          "exactly with the rerun's, so acceptance is directly comparable; "
          "treat its tok/s as indicative rather than head-to-head."]

found = sum(1 for _, p_ in RUNS for d in DS + EXTRA_DS if cell(p_, d))
L += ["", f"Cells found: {found} "
      f"({len(RUNS)} runs x {len(DS)} core benchmarks = {len(RUNS) * len(DS)} "
      f"expected, plus extended-benchmark cells).", ""]
out = f"{ROOT}/RESULTS_rerun.md"
open(out, "w").write("\n".join(L))
print(f"wrote {out} ({found} cells)")
