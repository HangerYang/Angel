#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""Load compact real_hawk LoRA pack + SmolVLM → full draft (+ LoRA-only dict).

Pack (public_weight/real_hawk_lora/) stores only LoRA A/B + fuse + draft
lm_head/norm + vocab maps. New packs include embed_tokens.weight; frozen base
layer weights come from SmolVLM layer init. The loader still initializes embed
from SmolVLM first so older packs that omitted it remain usable.

HF / vLLM eval needs plain ``*.weight`` Linear keys. Use ``--save_full``
(always merged hawk-shaped) or ``load_real_hawk_checkpoint`` for train
checkpoints that still have LoRA module keys.

Usage:
  python public_weight/load_real_hawk_lora.py \\
      --smolvlm HuggingFaceTB/SmolVLM-256M-Instruct \\
      --pack public_weight/real_hawk_lora \\
      --save_full ~/tmp/real_hawk_full \\
      --save_lora_only ~/tmp/real_hawk_lora_only.safetensors

  # In code:
  from public_weight.load_real_hawk_lora import load_real_hawk_lora, load_real_hawk_checkpoint
  model, lora_only = load_real_hawk_lora(
      smolvlm_path="HuggingFaceTB/SmolVLM-256M-Instruct",
      pack_dir="public_weight/real_hawk_lora",
  )
  # HF-compatible (merged Linear) from train ckpt or --save_full dir:
  hf_model = load_real_hawk_checkpoint("output/.../checkpoint-XXXX")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from safetensors.torch import load_file, save_file

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


def _state_has_lora_keys(state: Dict[str, torch.Tensor]) -> bool:
    return any("lora_A" in k or "lora_B" in k for k in state)


def _load_dir_state(path: Path) -> Dict[str, torch.Tensor]:
    path = Path(path)
    single = path / "model.safetensors"
    if single.is_file():
        return dict(load_file(str(single)))
    state: Dict[str, torch.Tensor] = {}
    for p in sorted(path.glob("*.safetensors")):
        state.update(load_file(str(p)))
    if state:
        return state
    bin_path = path / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No weights under {path}")


def _write_hf_hawk_dir(
    out_dir: Path,
    *,
    state: Dict[str, torch.Tensor],
    config: Dict,
) -> None:
    """Write a plain-hawk checkpoint loadable via ``from_pretrained``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    save_file(cpu_state, str(out_dir / "model.safetensors"))
    cfg = dict(config)
    # Merged weights are plain Linear; mode must be hawk for HF/vLLM.
    cfg["eagle_aux_injection_mode"] = "hawk"
    # Drop LoRA-only fields so from_pretrained does not expect adapters.
    for k in ("lora_r", "lora_alpha", "lora_dropout", "lora_target_modules"):
        cfg.pop(k, None)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


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

    # Backward compatibility: old packs omitted embed_tokens.weight.
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
    # Allowed missing: embed for old packs + all base.weight under LoRALinear.
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
        # Replace LoRA modules with merged Linear weights (hawk-shaped for HF/vLLM).
        merged = merge_lora_into_state_dict(model)
        cfg.eagle_aux_injection_mode = "hawk"
        plain = create_draft_model(cfg)
        if torch_dtype is not None:
            plain = plain.to(dtype=torch_dtype)
        missing, unexpected = plain.load_state_dict(merged, strict=False)
        missing = [k for k in missing if not k.startswith("embed_tokens.")]
        if missing or unexpected:
            raise RuntimeError(
                f"merged hawk load failed missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        plain.freeze_embed_weights()
        return plain, lora_only
    return model, lora_only


def load_real_hawk_checkpoint(
    ckpt_dir: Union[str, Path],
    *,
    config_path: Optional[Union[str, Path]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    merge_lora: bool = True,
) -> torch.nn.Module:
    """Load a real_hawk train checkpoint or HF ``--save_full`` dir.

    - If weights contain ``lora_A`` / ``lora_B``: inject LoRA, load, optionally merge.
    - If already plain hawk Linear weights: ``from_pretrained`` / strict load.

    ``merge_lora=True`` (default) returns an HF-evaluable hawk draft.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg_path = Path(config_path) if config_path else ckpt_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    state = _load_dir_state(ckpt_dir)
    cfg = DraftModelConfig.from_file(str(cfg_path))

    if not _state_has_lora_keys(state):
        # Already HF-compatible merged hawk.
        if torch_dtype is not None:
            from angelslim.compressor.speculative.train.models.draft.llama_eagle3 import (
                Eagle3LlamaForCausalLM,
            )

            return Eagle3LlamaForCausalLM.from_pretrained(
                str(ckpt_dir), dtype=torch_dtype, trust_remote_code=True
            )
        from angelslim.compressor.speculative.train.models.draft.llama_eagle3 import (
            Eagle3LlamaForCausalLM,
        )

        return Eagle3LlamaForCausalLM.from_pretrained(
            str(ckpt_dir), trust_remote_code=True
        )

    # Train-style LoRA checkpoint: inject shells then load.
    mode = getattr(cfg, "eagle_aux_injection_mode", "")
    if mode not in ("real_hawk", "layer_skip_lora"):
        # Force real_hawk so fuse/LoRA topology matches training.
        cfg.eagle_aux_injection_mode = "real_hawk"
    model = create_draft_model(cfg)
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
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
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [
        k
        for k in missing
        if "lora_" in k or k.startswith("fuse_") or k.startswith("lm_head") or k == "norm.weight"
    ]
    if bad_missing:
        raise RuntimeError(f"real_hawk ckpt missing required keys: {bad_missing[:20]}")
    if merge_lora:
        merged = merge_lora_into_state_dict(model)
        cfg.eagle_aux_injection_mode = "hawk"
        plain = create_draft_model(cfg)
        if torch_dtype is not None:
            plain = plain.to(dtype=torch_dtype)
        missing, unexpected = plain.load_state_dict(merged, strict=False)
        missing = [k for k in missing if not k.startswith("embed_tokens.")]
        if missing or unexpected:
            raise RuntimeError(
                f"merged hawk load failed missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        return plain
    return model


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
        help="Return hawk-shaped merged model in-process (also implied by --save_full)",
    )
    ap.add_argument(
        "--save_full",
        type=Path,
        default=None,
        help=(
            "Write HF-loadable merged hawk checkpoint dir "
            "(plain *.weight Linear keys; use with from_pretrained)"
        ),
    )
    ap.add_argument(
        "--save_lora_only",
        type=Path,
        default=None,
        help="Optional: write LoRA-only safetensors (A/B only)",
    )
    ap.add_argument(
        "--save_unmerged_full",
        type=Path,
        default=None,
        help="Optional: write train-style LoRA module state (NOT HF from_pretrained)",
    )
    args = ap.parse_args()

    want_merge = bool(args.merge_lora or args.save_full is not None)
    model, lora_only = load_real_hawk_lora(
        args.smolvlm,
        args.pack,
        embed_weight_key=args.embed_weight_key,
        merge_lora=want_merge,
    )
    n = sum(p.numel() for p in model.parameters())
    n_lora = sum(t.numel() for t in lora_only.values())
    print(
        f"Loaded real_hawk  params={n:,}  lora_only_tensors={len(lora_only)} "
        f"lora_params={n_lora:,}  merge_lora={want_merge}"
    )

    if args.save_lora_only is not None:
        args.save_lora_only.parent.mkdir(parents=True, exist_ok=True)
        save_file(lora_only, str(args.save_lora_only))
        print(f"Wrote LoRA-only → {args.save_lora_only}")

    if args.save_unmerged_full is not None:
        # Rebuild unmerged if we already merged for --save_full.
        if want_merge:
            unmerged, _ = load_real_hawk_lora(
                args.smolvlm,
                args.pack,
                embed_weight_key=args.embed_weight_key,
                merge_lora=False,
            )
        else:
            unmerged = model
        args.save_unmerged_full.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in unmerged.state_dict().items()},
            str(args.save_unmerged_full / "model.safetensors"),
        )
        cfg = read_pack_config(args.pack)
        with (args.save_unmerged_full / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        print(
            f"Wrote unmerged LoRA checkpoint → {args.save_unmerged_full} "
            "(use load_real_hawk_checkpoint; NOT plain from_pretrained)"
        )

    if args.save_full is not None:
        cfg = read_pack_config(args.pack)
        _write_hf_hawk_dir(
            args.save_full,
            state=model.state_dict(),
            config=cfg,
        )
        print(f"Wrote HF-loadable merged hawk → {args.save_full}")


if __name__ == "__main__":
    main()
