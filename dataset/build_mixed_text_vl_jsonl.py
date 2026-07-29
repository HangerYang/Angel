#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""
Build a mixed text / vision-language JSONL from ShareGPT + LLaVA-NeXT-Data.

Output format (one JSON object per line):

  # 文生文
  {"id": "0", "conversations": [
      {"role": "user", "content": [{"type": "text", "text": "..."}]},
      {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
  ]}

  # 图生文
  {"id": "1", "conversations": [
      {"role": "user", "content": [
          {"type": "image", "image": "/abs/path/to/images/0001.jpg"},
          {"type": "text", "text": "..."}
      ]},
      {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
  ]}

Example:
  python dataset/build_mixed_text_vl_jsonl.py \\
    --sharegpt dataset/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \\
    --llava-parquet /data1/datasets/lmms-lab/LLaVA-NeXT-Data/data/train-00000-of-00250.parquet \\
    --num-text 18 --num-vl 18 \\
    --output-dir dataset/mixed_text_vl_36
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "user": "user",
    "assistant": "assistant",
}


def text_content(text: str) -> dict:
    return {"type": "text", "text": text}


def strip_image_token(text: str) -> str:
    return text.replace("<image>", "").strip().lstrip("\n").strip()


def normalize_sharegpt(convs: list) -> list | None:
    """Convert ShareGPT turns to role/content; drop leading assistant turns."""
    out: list[dict] = []
    for turn in convs:
        role = ROLE_MAP.get(turn.get("from") or turn.get("role"))
        if role is None:
            continue
        val = turn.get("value") or ""
        if not str(val).strip():
            continue
        out.append({"role": role, "content": [text_content(str(val))]})

    while out and out[0]["role"] != "user":
        out.pop(0)
    if not out or out[0]["role"] != "user":
        return None
    if not any(t["role"] == "assistant" for t in out):
        return None
    return out


def load_pil_from_parquet_image(image_obj) -> Image.Image | None:
    if image_obj is None:
        return None
    if isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            return Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
        if image_obj.get("path"):
            try:
                return Image.open(image_obj["path"]).convert("RGB")
            except OSError:
                return None
    if hasattr(image_obj, "convert"):
        return image_obj.convert("RGB")
    return None


def sample_sharegpt(path: Path, n: int) -> list[dict]:
    print(f"Loading ShareGPT from {path} ...")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict] = []
    for item in data:
        convs = normalize_sharegpt(item.get("conversations") or [])
        if not convs:
            continue
        records.append({"conversations": convs})
        if len(records) >= n:
            break
    print(f"  selected {len(records)} text samples")
    return records


def sample_llava_parquet(path: Path, n: int, img_dir: Path) -> list[dict]:
    print(f"Reading LLaVA-NeXT parquet from {path} ...")
    img_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(path)

    records: list[dict] = []
    for batch in pf.iter_batches(batch_size=64):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            if len(records) >= n:
                break

            img = load_pil_from_parquet_image(row.get("image"))
            if img is None:
                continue

            convs_raw = row["conversations"]
            if hasattr(convs_raw, "tolist"):
                convs_raw = convs_raw.tolist()

            sid = str(row.get("id", f"llava_{len(records)}"))
            safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", sid)
            img_path = img_dir / f"{len(records):04d}_{safe}.jpg"
            img.save(img_path, format="JPEG", quality=95)
            abs_img = str(img_path.resolve())

            out_convs: list[dict] = []
            first_user = True
            for turn in convs_raw:
                if not isinstance(turn, dict):
                    continue
                role = ROLE_MAP.get(turn.get("from") or turn.get("role"))
                if role is None:
                    continue
                text = strip_image_token(str(turn.get("value") or ""))
                if role == "user" and first_user:
                    out_convs.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": abs_img},
                                text_content(text),
                            ],
                        }
                    )
                    first_user = False
                else:
                    if not text and role == "user":
                        continue
                    out_convs.append({"role": role, "content": [text_content(text)]})

            while out_convs and out_convs[0]["role"] != "user":
                out_convs.pop(0)
            if not out_convs or out_convs[0]["role"] != "user":
                img_path.unlink(missing_ok=True)
                continue
            if "image" not in [c.get("type") for c in out_convs[0]["content"]]:
                img_path.unlink(missing_ok=True)
                continue
            if not any(t["role"] == "assistant" for t in out_convs):
                img_path.unlink(missing_ok=True)
                continue

            records.append({"conversations": out_convs})
        if len(records) >= n:
            break

    print(f"  selected {len(records)} VL samples; images -> {img_dir}")
    return records


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sharegpt",
        type=Path,
        default=repo / "dataset/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
        help="ShareGPT JSON list (文生文 source)",
    )
    p.add_argument(
        "--llava-parquet",
        type=Path,
        default=Path(
            "/data1/datasets/lmms-lab/LLaVA-NeXT-Data/data/train-00000-of-00250.parquet"
        ),
        help="LLaVA-NeXT-Data parquet shard (图生文 source)",
    )
    p.add_argument("--num-text", type=int, default=18, help="Number of ShareGPT samples")
    p.add_argument("--num-vl", type=int, default=18, help="Number of LLaVA-NeXT samples")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "dataset/mixed_text_vl_36",
        help="Output directory (writes mixed_text_vl_36.jsonl + images/)",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="mixed_text_vl_36.jsonl",
        help="JSONL filename inside --output-dir",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    img_dir = out_dir / "images"
    out_jsonl = out_dir / args.output_name

    if img_dir.exists():
        for old in img_dir.glob("*"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    text_recs = sample_sharegpt(args.sharegpt, args.num_text)
    vl_recs = sample_llava_parquet(args.llava_parquet, args.num_vl, img_dir)

    records = []
    for i, rec in enumerate(text_recs + vl_recs):
        records.append({"id": str(i), "conversations": rec["conversations"]})

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_vl = sum(
        any(c.get("type") == "image" for t in r["conversations"] for c in t["content"])
        for r in records
    )
    print(f"Wrote {len(records)} lines ({len(records) - n_vl} text + {n_vl} vl) -> {out_jsonl}")


if __name__ == "__main__":
    main()
