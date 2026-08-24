#!/usr/bin/env python3
"""Summarise the prompt-style experiments.

Arm 1 (run_prompt_style_ab.sh):  raw vs the papers' prompts (`verbose`).
Arm 2 (run_prompt_variants.sh):  whole-prompt variants on the VQA benchmarks
                                 where `verbose` did nothing.
"""
import json, os, datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "my_angel", "prompt_style_ab")
OUT = os.path.join(REPO, "my_angel", "prompt-style-ab.md")

VQA = ["MMStar", "MMMU", "textvqa", "chartqa", "mathvista"]
LONG = ["OmniDocBench", "COCO-Caption", "MATH-500"]
STYLES = ["raw", "verbose", "detail_prefix", "cot", "describe_first", "min_words",
          "answer_then_describe"]

TEMPLATE = {
    "verbose": "question + 'Please answer with an explanation.'",
    "detail_prefix": "'Answer the following question in detail, explaining your reasoning: ' + question",
    "cot": "question + newline + \"Let's think step by step and explain the reasoning before giving the answer.\"",
    "describe_first": "'Describe what you see in the image in detail, then answer this question: ' + question",
    "min_words": "question + 'Please answer with at least 100 words.'",
    "answer_then_describe": "'Answer this question: ' + question + ' Then describe the image in detail to justify your answer.'",
}
RULE = {
    "OmniDocBench": "whole prompt replaced with the OCR instruction",
    "COCO-Caption": "whole prompt replaced with the caption instruction",
    "MATH-500": "never modified — control",
}


def metrics(style, ds):
    for ext in ("json", "jsonl"):
        f = f"{BASE}/{style}/{ds}/results.{ext}"
        if os.path.exists(f):
            d = json.load(open(f))
            return d.get("metrics", d)
    return None


def rows(style, ds):
    for ext in ("json", "jsonl"):
        f = f"{BASE}/{style}/{ds}/results.{ext}"
        if os.path.exists(f):
            d = json.load(open(f))
            return d.get("results") or d.get("rows") or []
    return []


def out_tok(style, ds):
    m = metrics(style, ds)
    return m.get("avg_output_tokens") if m else None


have = [s for s in STYLES if any(out_tok(s, d) is not None for d in VQA + LONG)]

L = ["# Generation prompts: what actually makes SmolVLM-256M answer at length", "",
     f"Generated {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}.", "",
     "| | |", "|---|---|",
     "| population | N=10 prompts per benchmark — the first 10 of the same ordered subset the N=80 sweeps use |",
     "| arm | target-only (`USE_EAGLE=0`, no draft): this measures prompt compliance, not acceptance |",
     "| prompt styles | `raw` (default) plus the styles below, via `--prompt_style` |",
     "| temperature | 0 |",
     "| decode | `output_len=1024`, `max_num_seqs=1`, `enforce_eager`, gpu-mem 0.8, tp 1 |",
     "| target | HuggingFaceTB/SmolVLM-256M-Instruct |", "",
     "N=10 means single samples move a mean: chartqa is 138.3 here but 86.0 at N=80,",
     "purely from one 522-word outlier. The first 10 generations are character-identical",
     "between the two populations, so this is subsampling, not a different configuration.", "",
     "| style | prompt |", "|---|---|",
     "| `raw` | the dataset question, verbatim |"]
for s in have:
    if s in TEMPLATE:
        L.append(f"| `{s}` | {TEMPLATE[s]} |")

L += ["", "## VQA benchmarks — avg output tokens", "",
      "| benchmark | " + " | ".join(f"`{s}`" for s in have) + " |",
      "|---|" + "---|" * len(have)]
for ds in VQA:
    cells = []
    base = out_tok("raw", ds)
    for s in have:
        v = out_tok(s, ds)
        if v is None:
            cells.append("—")
        elif s == "raw" or not base:
            cells.append(f"{v:.1f}")
        else:
            cells.append(f"{v:.1f} ({v / base:.2f}x)")
    L.append(f"| {ds} | " + " | ".join(cells) + " |")

means = []
for s in have:
    vals = [out_tok(s, d) for d in VQA if out_tok(s, d) is not None]
    means.append(f"**{sum(vals) / len(vals):.1f}**" if vals else "—")
L.append("| **mean** | " + " | ".join(means) + " |")

# Does the styled answer still contain what the raw arm answered? Crude proxy
# for "the benchmark still measures VQA" -- it says the answer survived, not
# that it is correct (the raw answer may itself be wrong).
def retained(style, ds):
    import re
    def norm(t):
        return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()
    a, b = rows("raw", ds), rows(style, ds)
    if not a or not b:
        return None
    hits = 0
    for x, y in zip(a, b):
        ra, sa = norm(x.get("generated_text")), norm(y.get("generated_text"))
        if ra and (ra in sa or all(w in sa for w in ra.split()[:3])):
            hits += 1
    return hits, min(len(a), len(b))


styled = [s_ for s_ in have if s_ != "raw"]
if styled:
    L += ["", "## Answer retention (raw answer still present in the styled answer)", "",
          "| benchmark | " + " | ".join(f"`{s_}`" for s_ in styled) + " |",
          "|---|" + "---|" * len(styled)]
    for ds in VQA:
        cells = []
        for s_ in styled:
            r = retained(s_, ds)
            cells.append(f"{r[0]}/{r[1]}" if r else "—")
        L.append(f"| {ds} | " + " | ".join(cells) + " |")

L += ["", "## Long-output benchmarks (raw vs the papers' prompts)", "",
      "| benchmark | change | raw | verbose | ratio |", "|---|---|---:|---:|---:|"]
for ds in LONG:
    r, v = out_tok("raw", ds), out_tok("verbose", ds)
    ratio = f"**{v / r:.2f}x**" if (r and v) else "—"
    L.append(f"| {ds} | {RULE.get(ds, '')} | "
             f"{'—' if r is None else format(r, '.1f')} | "
             f"{'—' if v is None else format(v, '.1f')} | {ratio} |")

L += ["", "## Sample answers (first prompt of each VQA benchmark)", "",
      "> `results.json` records the raw question, not the styled prompt, so only",
      "> the answers are shown — the applied prompt is echoed in each run's",
      "> `_logs/<bench>.log` as `prompt_style=... -> ...`.", ""]
for ds in VQA:
    got = [(s, rows(s, ds)) for s in have]
    got = [(s, r) for s, r in got if r]
    if not got:
        continue
    L += [f"### {ds}", "", f"Question: `{(got[0][1][0].get('question') or '')[:160]}`", ""]
    for s, r in got:
        L.append(f"- `{s}`: {(r[0].get('generated_text') or '')[:220]!r}")
    L.append("")

open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT} (styles: {', '.join(have)})")
