"""Smoke-test every arm against real rows of every dataset, one process.

Exercises the real vlmeval path -- SmolVLM.generate_inner, including the
per-dataset prompt builders -- rather than a reimplementation, because that is
where an arm breaks: a dataset whose prompt routes somewhere unexpected, an
image that will not decode, a selector that returns the wrong row count.

Loads the model once and mutates processor size / row_selector per arm, so the
whole matrix costs one model load.
"""

import os
import sys
import traceback

import torch

N_ROWS = int(os.environ.get("SMOKE_ROWS", "2"))


def main(datasets, arms):
    from vlmeval.config import supported_VLM
    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.smolvlm_select import build_selector

    model = supported_VLM["SmolVLM-256M"]()
    model.kwargs["max_new_tokens"] = 32          # smoke only; the sweep uses 2048
    default_size = dict(model.processor.image_processor.size)
    print(f"model loaded, default processor size = {default_size}\n", flush=True)

    results = {}
    for ds_name in datasets:
        try:
            ds = build_dataset(ds_name)
            rows = [ds.data.iloc[i] for i in range(min(N_ROWS, len(ds.data)))]
            msgs = [ds.build_prompt(r) for r in rows]
        except Exception as e:                                    # noqa: BLE001
            print(f"[{ds_name}] DATASET FAILED: {type(e).__name__}: {e}", flush=True)
            for a, _, _ in arms:
                results[(ds_name, a)] = "dataset-fail"
            continue

        for tag, edge, select in arms:
            os.environ.pop("SMOLVLM_TOKEN_SELECT", None)
            model.processor.image_processor.size = (
                {"longest_edge": int(edge)} if edge else dict(default_size)
            )
            if select:
                os.environ["SMOLVLM_TOKEN_SELECT"] = select
                os.environ["SMOLVLM_GLOBAL_POLICY"] = "exclude"
                os.environ.pop("SMOLVLM_SELECT_STATS", None)
                model.row_selector = build_selector()
            else:
                model.row_selector = None

            try:
                outs = [model.generate_inner(m, dataset=ds_name) for m in msgs]
                sel = model.row_selector
                kept = sel.n_selected // max(sel.n_calls, 1) if sel else None
                frm_g = sel.n_from_global if sel else 0
                bad = [o for o in outs if not str(o).strip()]
                if bad:
                    results[(ds_name, tag)] = "EMPTY-OUTPUT"
                elif sel and kept != sel.budget:
                    results[(ds_name, tag)] = f"BAD-ROWS {kept}!={sel.budget}"
                elif sel and select != "global_only" and frm_g:
                    results[(ds_name, tag)] = f"GLOBAL-LEAK {frm_g}"
                else:
                    k = f"{kept}r" if kept else "-"
                    results[(ds_name, tag)] = f"ok {k}"
                print(f"[{ds_name}] {tag:<12} {results[(ds_name, tag)]:<16} "
                      f"{str(outs[0]).strip()[:38]!r}", flush=True)
            except Exception as e:                                # noqa: BLE001
                results[(ds_name, tag)] = f"FAIL {type(e).__name__}"
                print(f"[{ds_name}] {tag:<12} FAIL {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
        print("", flush=True)

    print("\n===== SUMMARY =====")
    tags = [a[0] for a in arms]
    print(f"{'dataset':<20}" + "".join(f"{t:<16}" for t in tags))
    nbad = 0
    for d in datasets:
        line = f"{d:<20}"
        for t in tags:
            v = results.get((d, t), "?")
            if not v.startswith("ok"):
                nbad += 1
            line += f"{v:<16}"
        print(line)
    print(f"\n{nbad} problem cell(s)")
    return 1 if nbad else 0


if __name__ == "__main__":
    ARMS = [
        ("img832",      None,  None),
        ("img64",       512,   None),
        ("global_only", None,  "global_only"),
        ("stride64",    None,  "stride:64"),
        ("random64",    None,  "random:64"),
        ("norm64",      None,  "norm:64"),
        ("divprune64",  None,  "divprune:64"),
        ("cdpruner64",  None,  "cdpruner:64"),
    ]
    sys.exit(main(sys.argv[1:], ARMS))
