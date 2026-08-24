"""Filter the mixed text+VL jsonl down to rows that actually carry an image.

The mixed_70k70k set interleaves ShareGPT (text-only) with LLaVA-Instruct (VL).
Q-Sampler training only has signal on rows with pixels, so text-only rows are
dropped here once rather than skipped every epoch.

Rows are kept only when every referenced image exists on this machine -- a
missing path would otherwise blow up mid-epoch inside the collator.

    python scripts/qsampler/filter_image_rows.py \
        --in  dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl \
        --out dataset/smolvlm_256m_target_gen_mixed_70k70k/train_images_only.jsonl
"""

import argparse
import json
import os
import sys


def image_refs(row):
    """Image references in a row, in order, as ``(kind, value)``.

    Two schemas appear in this corpus:
      * ``{"type": "image", "image": "/abs/path.jpg"}``      -> ("path", path)
      * ``{"type": "image_url", "image_url": {"url": ...}}`` -> ("data", uri)
        where the url is a ``data:image/...;base64,`` payload (this is what the
        mixed_70k70k set actually uses -- images are embedded, not on disk).
    """
    refs = []
    for turn in row.get("conversations", []):
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "image":
                v = part.get("image")
                if v:
                    refs.append(("path", v))
            elif kind == "image_url":
                v = part.get("image_url")
                if isinstance(v, dict):
                    v = v.get("url")
                if v:
                    refs.append(("data" if str(v).startswith("data:") else "path", v))
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="keep rows whose image files are not present on this machine",
    )
    ap.add_argument("--limit", type=int, default=0, help="stop after N kept rows (0 = all)")
    args = ap.parse_args()

    total = kept = no_image = missing = bad = 0
    seen_dirs = {}
    with open(args.src) as fin, open(args.dst, "w") as fout:
        for line in fin:
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            refs = image_refs(row)
            if not refs:
                no_image += 1
                continue
            if not args.allow_missing_images:
                # Embedded data URIs are self-contained; only on-disk paths can
                # go missing. Cache by directory so a 130k-row scan does not
                # stat the same tree thousands of times.
                ok = True
                for kind, v in refs:
                    if kind == "data":
                        continue
                    d = os.path.dirname(v)
                    if d not in seen_dirs:
                        seen_dirs[d] = os.path.isdir(d)
                    if not seen_dirs[d] or not os.path.exists(v):
                        ok = False
                        break
                if not ok:
                    missing += 1
                    continue
            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                break
            if kept % 5000 == 0:
                print(f"  kept {kept} / scanned {total}", file=sys.stderr, flush=True)

    print(
        f"{args.src} -> {args.dst}\n"
        f"  scanned={total} kept={kept} text_only={no_image} "
        f"missing_images={missing} unparseable={bad}"
    )


if __name__ == "__main__":
    main()
