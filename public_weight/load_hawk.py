#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""Load compact hawk pack + SmolVLM → full Eagle3 hawk draft.

Pack (public_weight/hawk_warmup/) stores only hot intermediate weights
(layers, fuse, norm, lm_head, vocab maps). Embed is taken from SmolVLM.

Usage:
  python public_weight/load_hawk.py \\
      --smolvlm HuggingFaceTB/SmolVLM-256M-Instruct \\
      --pack public_weight/hawk_warmup \\
      --save_full ~/tmp/hawk_full_reload

  # In code:
  from public_weight.load_hawk import load_hawk_draft
  model = load_hawk_draft(
      smolvlm_path="HuggingFaceTB/SmolVLM-256M-Instruct",
      pack_dir="public_weight/hawk_warmup",
  )
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Union

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from angelslim.compressor.speculative.train.models.draft import (  # noqa: E402
    DraftModelConfig,
    create_draft_model,
)
from public_weight._common import (  # noqa: E402
    DEFAULT_EMBED_KEY,
    read_hot_weights,
    read_pack_config,
)


def load_hawk_draft(
    smolvlm_path: Union[str, Path],
    pack_dir: Union[str, Path] = _HERE / "hawk_warmup",
    *,
    embed_weight_key: str = DEFAULT_EMBED_KEY,
    torch_dtype: Optional[torch.dtype] = None,
):
    """Rebuild a full hawk draft from SmolVLM embed + compact hot pack.

    Returns
    -------
    model : Eagle3LlamaForCausalLM
    """
    pack_dir = Path(pack_dir)
    cfg_dict = read_pack_config(pack_dir)
    # Prefer train JSON for factory registration fields if pack cfg is thin.
    train_cfg_path = (
        _ROOT
        / "angelslim/compressor/speculative/train/configs/smolvlm-256m-hawk.json"
    )
    if train_cfg_path.is_file():
        cfg = DraftModelConfig.from_file(str(train_cfg_path))
        # Overlay pack config (keeps aux ids / mode from the exported run).
        for k, v in cfg_dict.items():
            setattr(cfg, k, v)
    else:
        cfg = DraftModelConfig.from_file(str(pack_dir / "config.json"))

    model = create_draft_model(cfg)
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)

    # Backward compatibility: old packs omitted embed_tokens.weight.
    model.load_embed_weights(str(smolvlm_path), embed_weight_key)
    model.freeze_embed_weights()

    hot = read_hot_weights(pack_dir)
    missing, unexpected = model.load_state_dict(hot, strict=False)
    # Old packs may miss embed_tokens.weight; it was loaded above.
    missing = [k for k in missing if not k.startswith("embed_tokens.")]
    if missing:
        raise RuntimeError(
            f"hawk pack missing required keys after load: {missing[:20]}"
        )
    if unexpected:
        raise RuntimeError(f"hawk pack unexpected keys: {unexpected[:20]}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--smolvlm",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="HF id or local SmolVLM path (provides embed_tokens)",
    )
    ap.add_argument(
        "--pack",
        type=Path,
        default=_HERE / "hawk_warmup",
        help="Compact pack dir (hawk_warmup/ or hawk_nccl/) with hot_weights.safetensors",
    )
    ap.add_argument(
        "--embed_weight_key",
        type=str,
        default=DEFAULT_EMBED_KEY,
    )
    ap.add_argument(
        "--save_full",
        type=Path,
        default=None,
        help="Optional: write a full HF draft checkpoint (for vLLM etc.)",
    )
    args = ap.parse_args()

    model = load_hawk_draft(
        args.smolvlm,
        args.pack,
        embed_weight_key=args.embed_weight_key,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded hawk draft  params={n:,}  mode={model.hawk=}")

    if args.save_full is not None:
        args.save_full.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(args.save_full))
        # Ensure config carries hawk mode from pack.
        cfg = read_pack_config(args.pack)
        with (args.save_full / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        print(f"Wrote full checkpoint → {args.save_full}")


if __name__ == "__main__":
    main()
