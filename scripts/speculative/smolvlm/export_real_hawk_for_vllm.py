#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""Merge real_hawk (or layer_skip_lora) draft weights into a hawk-shaped ckpt for vLLM.

Train keeps frozen base Linear weights + LoRA A/B. vLLM hawk eval expects plain
``*.weight`` tensors. This script:

1. Loads the trained draft
2. Merges LoRA into base weights
3. Writes a new directory with hawk ``eagle_aux_injection_mode``

Usage:
  python scripts/speculative/smolvlm/export_real_hawk_for_vllm.py \\
      --draft_model output/smolvlm_256m_real_hawk/checkpoint-1000 \\
      --output_dir output/smolvlm_256m_real_hawk_merged
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from angelslim.compressor.speculative.train.models.draft import (
    DraftModelConfig,
    create_draft_model,
    merge_lora_into_state_dict,
)
from angelslim.compressor.speculative.train.models.draft.lora_utils import (
    apply_real_hawk_training_setup,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft_model", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument(
        "--draft_model_config_path",
        type=Path,
        default=Path(
            "angelslim/compressor/speculative/train/configs/"
            "smolvlm-256m-real-hawk.json"
        ),
    )
    args = ap.parse_args()

    cfg_path = args.draft_model / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    train_cfg = DraftModelConfig.from_file(str(args.draft_model_config_path))
    # Build same structure as train (layers + LoRA), then load checkpoint.
    model = create_draft_model(train_cfg)
    apply_real_hawk_training_setup(
        model,
        r=int(getattr(train_cfg, "lora_r", 16)),
        alpha=float(getattr(train_cfg, "lora_alpha", 32)),
        dropout=float(getattr(train_cfg, "lora_dropout", 0.0)),
        target_modules=list(
            getattr(
                train_cfg,
                "lora_target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
        ),
    )
    weights = args.draft_model / "model.safetensors"
    bin_weights = args.draft_model / "pytorch_model.bin"
    if weights.is_file():
        from safetensors.torch import load_file

        state = load_file(str(weights))
    elif bin_weights.is_file():
        state = torch.load(bin_weights, map_location="cpu", weights_only=True)
    else:
        state = {}
        for p in sorted(args.draft_model.glob("*.safetensors")):
            from safetensors.torch import load_file

            state.update(load_file(str(p)))
        if not state:
            raise FileNotFoundError(f"No weights under {args.draft_model}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded checkpoint (missing={len(missing)}, unexpected={len(unexpected)})")

    merged = merge_lora_into_state_dict(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["eagle_aux_injection_mode"] = "hawk"
    cfg.pop("lora_r", None)
    cfg.pop("lora_alpha", None)
    cfg.pop("lora_dropout", None)
    cfg.pop("lora_target_modules", None)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    out_bin = args.output_dir / "pytorch_model.bin"
    clean = {k: v for k, v in merged.items() if "lora_A" not in k and "lora_B" not in k}
    torch.save(clean, out_bin)
    for name in ("generation_config.json",):
        src = args.draft_model / name
        if src.is_file():
            shutil.copy2(src, args.output_dir / name)
    print(f"Wrote hawk-shaped merged draft → {args.output_dir}")
    print("Eval with DRAFT_MODEL=<output_dir> and smolvlm-256m-hawk.json")


if __name__ == "__main__":
    main()
