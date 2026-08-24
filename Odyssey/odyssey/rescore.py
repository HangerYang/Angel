"""Branch 3: ground-truth re-score of the corrected tail.

Branches 1/2 verify the tail against distributions the target computed under
the *rejected* token. The honest question is how much that staleness costs, and
the only way to answer it is to run the target over the *corrected* prefix and
compare.

Doing that inside the verification loop means an extra target forward mid-round
(~3.9 ms on SmolVLM-256M, i.e. roughly the whole round it is trying to save),
so it is not a shippable policy. It is a control: run it offline over the
events the online `recompute` pass logged.

Fidelity note. A rescore is only meaningful if it conditions on the same prefix
the online target saw. That means:

  * text-only benchmarks only. SmolVLM is a VLM; for a benchmark with images
    the online distribution is conditioned on image embeddings this pass cannot
    reconstruct, so any number it produced would be measuring something else.
    MATH-500 is text-only, which is why the sweep uses it.
  * the prefix is rebuilt as (prompt tokens) + (tokens emitted in earlier
    rounds of the same request), and checked against the `first_pos` the
    sampler logged. Prompts whose reconstruction does not line up are skipped
    and counted, rather than silently rescored against the wrong context.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch


def load_events(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "round":
                    rows.append(rec)
    return rows


def group_into_requests(rounds: list[dict]) -> list[list[dict]]:
    """Split a flat round stream into per-prompt sequences.

    The sweep runs MAX_NUM_SEQS=1, so requests are strictly sequential and a
    drop in `first_pos` marks the start of the next prompt.
    """
    groups: list[list[dict]] = []
    cur: list[dict] = []
    last = None
    for r in rounds:
        fp = r.get("first_pos")
        if fp is None:
            continue
        if last is not None and fp <= last:
            if cur:
                groups.append(cur)
            cur = []
        cur.append(r)
        last = fp
    if cur:
        groups.append(cur)
    return groups


def rescore(
    model_id: str,
    groups: list[list[dict]],
    prompt_ids: list[list[int]],
    device: str = "cuda",
    max_events: int | None = None,
) -> tuple[list[dict], dict]:
    """Re-run the target over each corrected prefix.

    For a rejection at index i of a round whose prefix is P, the corrected
    continuation is P + draft[:i] + [correction]. We run one target forward
    over that and greedily walk the remaining proposals, counting how many the
    fresh distribution accepts. That is the number the stale branches
    approximate.

    Greedy only: at temperature 0 acceptance is exactly "does the argmax
    match", which makes the comparison against the stale branches deterministic
    and needs no sampling.
    """
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    full = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16)
    # Text-only rescore: drive the language model directly so no image inputs
    # are required (and none are silently defaulted in).
    lm = full.model.text_model.to(device).eval()
    head = full.lm_head.to(device).eval()

    out: list[dict] = []
    stats = {
        "groups": len(groups),
        "groups_matched": 0,
        "groups_skipped_prefix_mismatch": 0,
        "events_scored": 0,
    }

    # Match a group to its prompt by the length the engine reported rather than
    # by position: some prompts finish without ever hitting a rejection, so the
    # group index and the prompt index drift apart.
    by_len: dict[int, list[int]] = {}
    for pi, ids in enumerate(prompt_ids):
        by_len.setdefault(len(ids), []).append(pi)
    used: set[int] = set()

    n_scored = 0
    for g in groups:
        if not g:
            continue
        want = int(g[0]["first_pos"])
        cands = [pi for pi in by_len.get(want, []) if pi not in used]
        if not cands:
            stats["groups_skipped_prefix_mismatch"] += 1
            continue
        pi = cands[0]
        used.add(pi)
        prompt = list(prompt_ids[pi])
        stats["groups_matched"] += 1

        prefix = list(prompt)
        for r in g:
            i = int(r.get("first_reject", -1))
            draft = list(r.get("draft_tokens", []))
            if i >= 0 and i + 1 < len(draft) and (
                max_events is None or n_scored < max_events
            ):
                corrected = prefix + draft[:i] + [int(r["correction"])]
                ids = torch.tensor([corrected], device=device)
                with torch.no_grad():
                    h = lm(input_ids=ids).last_hidden_state
                    logits = head(h)[0]

                # Walk the remaining proposals against fresh distributions.
                fresh = 0
                cur = list(corrected)
                nxt = int(logits[-1].argmax())
                for j in range(i + 1, len(draft)):
                    if nxt != draft[j]:
                        break
                    fresh += 1
                    cur.append(draft[j])
                    with torch.no_grad():
                        h = lm(input_ids=torch.tensor([cur], device=device))
                        nxt = int(head(h.last_hidden_state)[0, -1].argmax())

                out.append(
                    {
                        "round": r.get("round"),
                        "req": r.get("req"),
                        "first_reject": i,
                        "fresh_salvage": fresh,
                        "tail_len": len(draft) - (i + 1),
                        "reject_entropy": r.get("reject_entropy"),
                    }
                )
                n_scored += 1
                stats["events_scored"] = n_scored

            # Advance the prefix by what the engine actually emitted, so later
            # rounds in this request condition on the real context.
            prefix.extend(int(t) for t in r.get("emitted", []))

    return out, stats


def load_prompt_ids(dataset: str, num_prompts: int, model_id: str) -> list[list[int]]:
    """Tokenize the benchmark prompts the same way the eval driver does.

    Only the text-only path is supported; see the module docstring.
    """
    from datasets import load_dataset
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    ds = load_dataset(dataset, split="test")
    ds = ds.select(range(min(num_prompts, len(ds))))

    out = []
    for row in ds:
        q = row.get("problem") or row.get("question") or ""
        # SmolVLM's chat template only renders a content-part list; a bare
        # string collapses to a 9-token prompt, which matches nothing.
        msgs = [{"role": "user", "content": [{"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        ids = proc.tokenizer(text, add_special_tokens=False)["input_ids"]
        out.append(list(ids))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="glob for the recompute run's JSONL")
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    ap.add_argument("--num_prompts", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_events", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ev = load_events(args.events)
    groups = group_into_requests(ev)
    print(f"loaded {len(ev)} round events -> {len(groups)} requests")

    prompts = load_prompt_ids(args.dataset, args.num_prompts, args.model)
    print(f"tokenized {len(prompts)} prompts from {args.dataset}")

    rows, stats = rescore(
        args.model, groups, prompts, device=args.device, max_events=args.max_events
    )
    Path(os.path.dirname(os.path.abspath(args.out))).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"wrote {len(rows)} rescored events -> {args.out}")
    if stats["groups_matched"] == 0:
        print(
            "WARNING: no request prefix matched. The control did not run; do "
            "not read vs_recompute numbers from this sweep."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
