#!/usr/bin/env python3
"""One-layer draft comparison: 6 runs x 8 benchmarks x temp 0/1.

Canonical source = the answer_then_describe (ATD, "long prompt") sweep written by
scripts/speculative/smolvlm/run_atd_acceptance.sh into the `atd_temp{t}` /
`rerun_atd/temp{t}` roots. The older short-prompt sweep (`temp{t}` /
`rerun/temp{t}`) is kept only for the output-length comparison that motivated
the prompt change; it is no longer reported as a result.

Separate from make_results.py (the depth-grouped rerun table).
"""
import json, os, datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "my_angel", "one-layer-results.md")

CORE = ["MMStar", "MMMU", "OmniDocBench", "MATH-500"]
EXTRA = ["textvqa", "chartqa", "mathvista", "COCO-Caption"]
DS = CORE + EXTRA
TEMPS = [0, 1]
EXPECTED_N = 80

BASELINE = "no_eagle_baseline"
# Canonical (long-prompt) roots. name -> out root, {t} substituted.
RUNS = [
    (BASELINE, "my_angel/no_eagle_baseline/atd_temp{t}"),
    ("baseline_1layer", "my_angel/eagle/baseline_1layer/rerun_atd/temp{t}"),
    ("banded_mix_fc_3.1", "my_angel/eagle/smolvlm-256m-eagle3-banded-mix-fc-3.1/rerun_atd/temp{t}"),
    ("banded_mix_wide_3.1", "my_angel/eagle/smolvlm-256m-eagle3-banded-mix-wide-3.1/rerun_atd/temp{t}"),
    ("branch_distill_top1_w01", "my_angel/eagle/branch-distill-top1-w01/rerun_atd/temp{t}"),
    ("vistoken_k1", "my_angel/eagle/vistoken-k1/rerun_atd/temp{t}"),
]
DRAFTS = [r for r in RUNS if r[0] != BASELINE]

# Superseded short-prompt roots: used only for the output-length ratio table.
RAW_BASELINE_ROOT = "my_angel/no_eagle_baseline/temp{t}"

# How answer_then_describe touches each benchmark's prompt.
ATD_TREATMENT = {
    "MMStar": "wrapper", "MMMU": "wrapper", "textvqa": "wrapper",
    "chartqa": "wrapper", "mathvista": "wrapper",
    "OmniDocBench": "prompt replaced (OCR)", "COCO-Caption": "prompt replaced (caption)",
    "MATH-500": "**unmodified** — reruns the raw prompt",
}


def cell(root, temp, ds):
    base = os.path.join(REPO, root.format(t=temp), ds)
    for ext in ("json", "jsonl"):
        f = f"{base}/results.{ext}"
        if os.path.exists(f):
            try:
                d = json.load(open(f))
                return d.get("metrics", d)
            except Exception:
                pass
    return None


def fmt(v, spec):
    return format(v, spec) if v is not None else "—"


def table(header, rows):
    return [f"| {header} | " + " | ".join(DS) + " | mean |",
            "|---|" + "---|" * (len(DS) + 1)] + rows


def metric_rows(temp, key, spec, runs):
    out = []
    for name, root in runs:
        vals, cells = [], []
        for ds in DS:
            m = cell(root, temp, ds)
            v = m.get(key) if m else None
            if v is not None:
                vals.append(v)
            cells.append(fmt(v, spec))
        mean = f"**{format(sum(vals)/len(vals), spec)}**" if vals else "—"
        out.append(f"| `{name}` | " + " | ".join(cells) + f" | {mean} |")
    return out


def speedup_rows(temp):
    base_root = dict(RUNS)[BASELINE]
    out = []
    for name, root in DRAFTS:
        vals, cells = [], []
        for ds in DS:
            m, b = cell(root, temp, ds), cell(base_root, temp, ds)
            if m and b and b.get("output_throughput"):
                r = m["output_throughput"] / b["output_throughput"]
                vals.append(r); cells.append(f"{r:.3f}x")
            else:
                cells.append("—")
        mean = f"**{sum(vals)/len(vals):.3f}x**" if vals else "—"
        out.append(f"| `{name}` | " + " | ".join(cells) + f" | {mean} |")
    return out


def missing_cells(temp):
    return [f"{name} {ds}" for name, root in RUNS for ds in DS
            if cell(root, temp, ds) is None]


L = [
    "# One-layer drafts — 8 benchmarks, GPU 0 sequential", "",
    f"Generated {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}.", "",
    "Every 1-layer draft plus the non-speculative target, over 8 benchmarks.",
    "One vLLM job at a time on GPU 0 with nothing else on the box, so tok/s is",
    "comparable across every cell.", "",
    "| | |", "|---|---|",
    f"| population | N={EXPECTED_N} prompts per benchmark |",
    "| prompt | long prompt (`answer_then_describe`): question unchanged, model answers then justifies |",
    "| temperature | 0 and 1 |",
    "| arms | `no_eagle_baseline` = target-only; every other row = with-draft (EAGLE-3, K=4) |",
    "| decode | `output_len=1024`, `max_num_seqs=1`, `enforce_eager`, gpu-mem 0.8, tp 1 |",
    "| target | HuggingFaceTB/SmolVLM-256M-Instruct |",
    "| drafts | `checkpoint-66466` of each run (1 draft layer) |", "",
    "Produced by `scripts/speculative/smolvlm/run_atd_acceptance.sh`:", "",
    "```bash",
    'DATASETS_ATD="Lin-Chen/MMStar MMMU/MMMU opendatalab/OmniDocBench HuggingFaceH4/MATH-500 \\',
    '  lmms-lab/textvqa lmms-lab/chartqa ai4math/mathvista lmms-lab/COCO-Caption" \\',
    "  TEMP=0 bash scripts/speculative/smolvlm/run_atd_acceptance.sh   # then TEMP=1",
    "python my_angel/make_one_layer_results.py",
    "```", "",
    "**Engine / how this differs from the default eval.** Same harness every arm uses:",
    "`run_atd_acceptance.sh` -> `scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh`",
    "-> `tools/vllm_offline_eagle3_vlm_batch.py` -> vLLM V1 offline `LLM(...)` from",
    "`third_party/vllm` (v0.25.0) with `speculative_config={method: eagle3, ...}` and",
    "`enforce_eager`. The wrapper overrides exactly three things, none of them the engine:", "",
    "| | default suite | this sweep |", "|---|---|---|",
    "| prompt style | `raw` (`eval_acceptance_suite_dp.sh:114`) | `answer_then_describe` |",
    "| benchmarks | 10 others: ChartQA, VQAv2, GQA, ScienceQA, MME, SEED-Bench, MMVet, MMBench, … (`:38-48`) | the 8 below |",
    "| GPUs | round-robins one job per GPU | pinned `CUDA_VISIBLE_DEVICES=0`, serial, so tok/s is comparable |", "",
    "`raw` left MMStar / textvqa / mathvista at 9-18 output tokens — too short for",
    "speculation to pay — which is why this sweep, not the default one, is the record.", "",
    "| run | what it is |", "|---|---|",
    "| `no_eagle_baseline` | target only, no speculative decoding |",
    "| `baseline_1layer` | stock EAGLE-3: 3 aux layers, 3H→H fusion FC, 2H layer 0 |",
    "| `banded_mix_fc_3.1` | 9 aux layers → 3 learned band mixes → 3H→H FC, EAGLE 3.1 |",
    "| `banded_mix_wide_3.1` | same 3 band mixes, **no** FC: 4H layer 0 `[emb｜band0｜band1｜seed]` |",
    "| `branch_distill_top1_w01` | `banded_mix_fc_3.1` + branch-aware distillation (draft top-1 vs teacher top-3, w=0.1, 1 step) |",
    "| `vistoken_k1` | `banded_mix_fc_3.1` + learned-query row compression: each tile's 64 visual rows → k=1 summary |",
    "", "## Prompt", "",
    "Every benchmark uses the long prompt: the model answers first and then justifies,",
    "so outputs are long enough for speculative decoding to pay. The question text",
    "itself is unchanged. How each benchmark is treated:", "",
    "| benchmark | treatment |", "|---|---|",
]
for ds in DS:
    L.append(f"| {ds} | {ATD_TREATMENT.get(ds, '?')} |")

for temp in TEMPS:
    miss = missing_cells(temp)
    L += ["", f"## Results — temp {temp}", ""]
    if len(miss) == len(RUNS) * len(DS):
        L += [f"> Not run yet: all {len(miss)} cells missing.", ""]
        continue
    L += ["### Acceptance length", ""]
    L += table("run", metric_rows(temp, "mean_acceptance_length", ".3f", DRAFTS))
    L += ["", "### Throughput (tok/s)", ""]
    L += table("run", metric_rows(temp, "output_throughput", ".1f", RUNS))
    L += ["", "### Speedup vs `no_eagle_baseline`", ""]
    L += table("run", speedup_rows(temp))
    if miss:
        L += ["", f"> **{len(miss)} of {len(RUNS) * len(DS)} cells missing** "
              "(sweep in flight or failed): "
              + ", ".join(f"`{m}`" for m in miss[:12])
              + (" …" if len(miss) > 12 else "") + "."]

# Length depends on the arm: speculative decoding is not bit-exact against
# target-only at temp 0 in this build, so quote the target-only arm and say so
# rather than whichever run happens to be found first.
L += ["", "## Prompt / output sizes", "",
      f"Target-only arm (`{BASELINE}`), long prompt, N={EXPECTED_N}. The with-draft arms",
      "differ slightly on long outputs because the speculative path is not bit-exact",
      "here; do not mix the two.", "",
      "| dataset | N | temp | avg input tok | avg output tok |", "|---|---:|---:|---:|---:|"]
_base_root = dict(RUNS)[BASELINE]
for temp in TEMPS:
    for ds in DS:
        m = cell(_base_root, temp, ds)
        if m:
            L.append(f"| {ds} | {EXPECTED_N} | {temp} | {m.get('avg_input_tokens', 0):.1f} | "
                     f"{m.get('avg_output_tokens', 0):.1f} |")

L += ["", "### Why the long prompt (target-only, temp 0)", "",
      "The short prompt this sweep replaced left several benchmarks at 9-45 output",
      "tokens — too short for speculation to pay. Superseded numbers, kept as the",
      "reason for the change:", "",
      "| benchmark | short-prompt out tok | long-prompt out tok | ratio |", "|---|---:|---:|---:|"]
for ds in DS:
    r, a = cell(RAW_BASELINE_ROOT, 0, ds), cell(_base_root, 0, ds)
    if r and a:
        L.append(f"| {ds} | {r['avg_output_tokens']:.1f} | {a['avg_output_tokens']:.1f} | "
                 f"**{a['avg_output_tokens'] / r['avg_output_tokens']:.2f}x** |")

open(OUT, "w").write("\n".join(L) + "\n")
total = len(RUNS) * len(DS) * len(TEMPS)
done = total - sum(len(missing_cells(t)) for t in TEMPS)
print(f"wrote {OUT} ({done}/{total} cells)")
