#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""Load compact real_hawk LoRA pack + SmolVLM → full draft (+ LoRA-only dict).

Pack (public_weight/real_hawk_lora/) stores only LoRA A/B + fuse + draft
lm_head/norm + vocab maps. Frozen base layer weights and embed come from
SmolVLM (layer init [1,14,26] + embed copy).

Usage:
  python public_weight/load_real_hawk_lora.py \\
      --smolvlm HuggingFaceTB/SmolVLM-256M-Instruct \\
      --pack public_weight/real_hawk_lora \\
      --save_full /tmp/real_hawk_full \\
      --save_lora_only /tmp/real_hawk_lora_only.safetensors

  # In code:
  from public_weight.load_real_hawk_lora import load_real_hawk_lora
  model, lora_only = load_real_hawk_lora(
      smolvlm_path="HuggingFaceTB/SmolVLM-256M-Instruct",
      pack_dir="public_weight/real_hawk_lora",
  )
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from safetensors.torch import save_file

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from angelslim.compressor.speculative.train.models.draft import (  # noqa: E402
    DraftModelConfig,
    create_draft_model,
    apply_real_hawk_training_setup,
    merge_lora_into_state_dict,
)
from public_weight._common import (  # noqa: E402
    DEFAULT_EMBED_KEY,
    read_hot_weights,
    read_pack_config,
)


def _lora_only_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().contiguous()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def load_real_hawk_lora(
    smolvlm_path: Union[str, Path],
    pack_dir: Union[str, Path] = _HERE / "real_hawk_lora",
    *,
    embed_weight_key: str = DEFAULT_EMBED_KEY,
    torch_dtype: Optional[torch.dtype] = None,
    merge_lora: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
    """Rebuild real_hawk draft from SmolVLM + compact LoRA pack.

    Returns
    -------
    model : Eagle3LlamaForCausalLM with LoRA modules (or merged if merge_lora)
    lora_only : dict of lora_A / lora_B tensors (always the unmerged adapters)
    """
    pack_dir = Path(pack_dir)
    cfg_dict = read_pack_config(pack_dir)
    train_cfg_path = (
        _ROOT
        / "angelslim/compressor/speculative/train/configs/smolvlm-256m-real-hawk.json"
    )
    if train_cfg_path.is_file():
        cfg = DraftModelConfig.from_file(str(train_cfg_path))
        for k, v in cfg_dict.items():
            setattr(cfg, k, v)
    else:
        cfg = DraftModelConfig.from_file(str(pack_dir / "config.json"))

    model = create_draft_model(cfg)
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)

    # Bottom embed from SmolVLM.
    model.load_embed_weights(str(smolvlm_path), embed_weight_key)
    model.freeze_embed_weights()

    # Frozen base layers from SmolVLM target layers (draft_layer_init_from_target).
    layer_ids = list(getattr(cfg, "draft_layer_init_from_target", [1, 14, 26]))
    model.load_layer_weights_from_target(
        str(smolvlm_path),
        layer_ids,
        embed_weight_key=embed_weight_key,
    )

    # Inject LoRA shells, then load hot pack (LoRA + fuse + head + vocab).
    apply_real_hawk_training_setup(
        model,
        r=int(getattr(cfg, "lora_r", 16)),
        alpha=float(getattr(cfg, "lora_alpha", 32)),
        dropout=float(getattr(cfg, "lora_dropout", 0.0)),
        target_modules=list(
            getattr(
                cfg,
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

    hot = read_hot_weights(pack_dir)
    missing, unexpected = model.load_state_dict(hot, strict=False)
    # Allowed missing: embed + all base.weight under LoRALinear.
    allowed_missing = []
    bad_missing = []
    for k in missing:
        if k.startswith("embed_tokens."):
            allowed_missing.append(k)
        elif k.endswith(".base.weight") or (
            k.endswith(".weight")
            and ("self_attn." in k or "mlp." in k)
            and "lora_" not in k
        ):
            # Bases already filled from SmolVLM; LoRALinear uses .base.weight.
            allowed_missing.append(k)
        elif "hidden_norm" in k or "input_layernorm" in k or "post_attention_layernorm" in k:
            # Norms copied from target during layer init; not in LoRA pack.
            allowed_missing.append(k)
        else:
            bad_missing.append(k)
    if bad_missing:
        raise RuntimeError(
            f"real_hawk pack missing required keys: {bad_missing[:30]}"
        )
    if unexpected:
        raise RuntimeError(f"real_hawk pack unexpected keys: {unexpected[:20]}")

    lora_only = _lora_only_state(model)
    if merge_lora:
        # Replace LoRA modules with merged Linear weights (hawk-shaped for vLLM).
        merged = merge_lora_into_state_dict(model)
        # Rebuild plain hawk model without LoRA wrappers.
        cfg.eagle_aux_injection_mode = "hawk"
        plain = create_draft_model(cfg)
        if torch_dtype is not None:
            plain = plain.to(dtype=torch_dtype)
        plain.load_state_dict(merged, strict=False)
        plain.freeze_embed_weights()
        return plain, lora_only
    return model, lora_only


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--smolvlm",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="HF id or local SmolVLM path (embed + base layer weights)",
    )
    ap.add_argument(
        "--pack",
        type=Path,
        default=_HERE / "real_hawk_lora",
    )
    ap.add_argument("--embed_weight_key", type=str, default=DEFAULT_EMBED_KEY)
    ap.add_argument(
        "--merge_lora",
        action="store_true",
        help="Return hawk-shaped merged weights (for vLLM) instead of LoRA modules",
    )
    ap.add_argument(
        "--save_full",
        type=Path,
        default=None,
        help="Optional: write full draft checkpoint directory",
    )
    ap.add_argument(
        "--save_lora_only",
        type=Path,
        default=None,
        help="Optional: write LoRA-only safetensors (A/B only)",
    )
    args = ap.parse_args()

    model, lora_only = load_real_hawk_lora(
        args.smolvlm,
        args.pack,
        embed_weight_key=args.embed_weight_key,
        merge_lora=args.merge_lora,
    )
    n = sum(p.numel() for p in model.parameters())
    n_lora = sum(t.numel() for t in lora_only.values())
    print(
        f"Loaded real_hawk  params={n:,}  lora_only_tensors={len(lora_only)} "
        f"lora_params={n_lora:,}  merge_lora={args.merge_lora}"
    )

    if args.save_lora_only is not None:
        args.save_lora_only.parent.mkdir(parents=True, exist_ok=True)
        save_file(lora_only, str(args.save_lora_only))
        print(f"Wrote LoRA-only → {args.save_lora_only}")

    if args.save_full is not None:
        args.save_full.mkdir(parents=True, exist_ok=True)
        if args.merge_lora:
            model.save_pretrained(str(args.save_full))
            cfg = read_pack_config(args.pack)
            cfg["eagle_aux_injection_mode"] = "hawk"
            with (args.save_full / "config.json").open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
                f.write("\n")
        else:
            # Save full state including LoRA wrappers (train-compatible).
            from safetensors.torch import save_file as _save

            _save(
                {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                str(args.save_full / "model.safetensors"),
            )
            cfg = read_pack_config(args.pack)
            with (args.save_full / "config.json").open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
                f.write("\n")
        print(f"Wrote full checkpoint → {args.save_full}")


if __name__ == "__main__":
    main()
