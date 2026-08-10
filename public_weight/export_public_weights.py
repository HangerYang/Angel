#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""Export compact packs into public_weight/{hawk_warmup,hawk_nccl,real_hawk_lora}.

Hawk / Eagle: skip embed (bottom; reload from SmolVLM). Keep hot intermediate
layers + fuse + draft lm_head/norm + vocab maps.

real_hawk LoRA: skip embed + frozen base Linear weights. Keep LoRA A/B + fuse
+ draft lm_head/norm + vocab maps. Bases reload from SmolVLM layer init.

Usage:
  python public_weight/export_public_weights.py
  python public_weight/export_public_weights.py \\
      --hawk_ckpt output/smolvlm_256m_hawk_warmup \\
      --hawk_nccl_ckpt output/smolvlm_256m_hawk_nccl \\
      --real_hawk_ckpt output/smolvlm_256m_real_hawk_nccl/checkpoint-66466
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from public_weight._common import (  # noqa: E402
    filter_keys,
    load_state_dict,
    resolve_latest_checkpoint,
    write_pack,
)


def _export_hawk(ckpt: Path, out_dir: Path, *, kind: str = "hawk") -> None:
    ckpt = resolve_latest_checkpoint(ckpt)
    state = load_state_dict(ckpt)
    with (ckpt / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Skip bottom embed (reload from SmolVLM). Keep layers/fuse/head/vocab.
    hot = filter_keys(state, drop_prefixes=("embed_tokens.",))
    write_pack(
        out_dir,
        hot=hot,
        draft_config=cfg,
        meta={
            "kind": kind,
            "source_ckpt": str(ckpt.resolve()),
            "skipped": ["embed_tokens.*  (reload from SmolVLM)"],
            "kept": [
                "layers.*",
                "fuse_w1/fuse_w2",
                "norm",
                "lm_head",
                "t2d/d2t",
            ],
        },
    )


def _export_real_hawk_lora(ckpt: Path, out_dir: Path) -> None:
    ckpt = resolve_latest_checkpoint(ckpt)
    state = load_state_dict(ckpt)
    with (ckpt / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Keep LoRA + fuse + head/vocab. Drop embed and frozen base weights.
    hot = filter_keys(
        state,
        keep_substrings=("lora_A", "lora_B"),
        keep_prefixes=("fuse_w1.", "fuse_w2.", "norm.", "lm_head."),
        keep_exact=("t2d", "d2t"),
    )
    write_pack(
        out_dir,
        hot=hot,
        draft_config=cfg,
        meta={
            "kind": "real_hawk_lora",
            "source_ckpt": str(ckpt.resolve()),
            "skipped": [
                "embed_tokens.*  (reload from SmolVLM)",
                "*.base.weight / plain layer weights (reload from SmolVLM layer init)",
            ],
            "kept": [
                "layers.*.lora_A / lora_B",
                "fuse_w1/fuse_w2",
                "norm",
                "lm_head",
                "t2d/d2t",
            ],
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--hawk_ckpt",
        type=Path,
        default=_ROOT / "output" / "smolvlm_256m_hawk_warmup",
    )
    ap.add_argument(
        "--hawk_nccl_ckpt",
        type=Path,
        default=_ROOT / "output" / "smolvlm_256m_hawk_nccl",
    )
    ap.add_argument(
        "--real_hawk_ckpt",
        type=Path,
        default=_ROOT / "output" / "smolvlm_256m_real_hawk_nccl",
    )
    ap.add_argument(
        "--out_root",
        type=Path,
        default=_HERE,
        help="public_weight/ root (writes hawk_warmup/, hawk_nccl/, real_hawk_lora/)",
    )
    ap.add_argument("--skip_hawk", action="store_true")
    ap.add_argument("--skip_hawk_nccl", action="store_true")
    ap.add_argument("--skip_real_hawk", action="store_true")
    args = ap.parse_args()

    if not args.skip_hawk:
        _export_hawk(args.hawk_ckpt, args.out_root / "hawk_warmup", kind="hawk_warmup")
    if not args.skip_hawk_nccl:
        _export_hawk(args.hawk_nccl_ckpt, args.out_root / "hawk_nccl", kind="hawk_nccl")
    if not args.skip_real_hawk:
        _export_real_hawk_lora(args.real_hawk_ckpt, args.out_root / "real_hawk_lora")


if __name__ == "__main__":
    main()
