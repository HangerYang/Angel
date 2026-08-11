# Copyright 2025 Tencent Inc. All Rights Reserved.
"""Shared helpers for compact public_weight packs (hawk / real_hawk LoRA)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBED_KEY = "model.text_model.embed_tokens.weight"


def load_state_dict(ckpt_dir: Path) -> Dict[str, torch.Tensor]:
    """Load a HF-style draft checkpoint directory into a state_dict."""
    ckpt_dir = Path(ckpt_dir)
    single = ckpt_dir / "model.safetensors"
    if single.is_file():
        return dict(load_file(str(single)))
    bin_path = ckpt_dir / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    state: Dict[str, torch.Tensor] = {}
    for p in sorted(ckpt_dir.glob("*.safetensors")):
        state.update(load_file(str(p)))
    if not state:
        raise FileNotFoundError(f"No weights under {ckpt_dir}")
    return state


def bytes_of(state: Dict[str, torch.Tensor]) -> int:
    return int(sum(v.numel() * v.element_size() for v in state.values()))


def filter_keys(
    state: Dict[str, torch.Tensor],
    *,
    keep_prefixes: Optional[Iterable[str]] = None,
    keep_substrings: Optional[Iterable[str]] = None,
    drop_prefixes: Optional[Iterable[str]] = None,
    drop_substrings: Optional[Iterable[str]] = None,
    keep_exact: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    keep_p = tuple(keep_prefixes or ())
    keep_s = tuple(keep_substrings or ())
    drop_p = tuple(drop_prefixes or ())
    drop_s = tuple(drop_substrings or ())
    keep_e: Set[str] = set(keep_exact or ())
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k in keep_e:
            out[k] = v.detach().cpu().contiguous()
            continue
        if drop_p and k.startswith(drop_p):
            continue
        if drop_s and any(s in k for s in drop_s):
            continue
        if keep_p or keep_s:
            if keep_p and k.startswith(keep_p):
                out[k] = v.detach().cpu().contiguous()
            elif keep_s and any(s in k for s in keep_s):
                out[k] = v.detach().cpu().contiguous()
            continue
        out[k] = v.detach().cpu().contiguous()
    return out


def write_pack(
    out_dir: Path,
    *,
    hot: Dict[str, torch.Tensor],
    draft_config: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hot = {k: v.detach().cpu().contiguous() for k, v in hot.items()}
    embed = {}
    if "embed_tokens.weight" in hot:
        embed["embed_tokens.weight"] = hot.pop("embed_tokens.weight")

    save_file(hot, str(out_dir / "hot_weights.safetensors"))
    embed_path = out_dir / "embed_tokens.safetensors"
    if embed:
        save_file(embed, str(embed_path))
    elif embed_path.exists():
        embed_path.unlink()

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(draft_config, f, indent=2)
        f.write("\n")

    all_keys = {**hot, **embed}
    meta = {
        **meta,
        "num_tensors": len(all_keys),
        "bytes": bytes_of(all_keys),
        "files": {
            "hot_weights.safetensors": sorted(hot.keys()),
            "embed_tokens.safetensors": sorted(embed.keys()),
        },
        "keys": sorted(all_keys.keys()),
    }
    with (out_dir / "pack_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(
        f"Wrote {out_dir}  tensors={len(all_keys)}  "
        f"size={meta['bytes'] / 1e6:.2f} MB"
    )


def read_pack_config(pack_dir: Path) -> Dict[str, Any]:
    with (Path(pack_dir) / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def read_hot_weights(pack_dir: Path) -> Dict[str, torch.Tensor]:
    pack_dir = Path(pack_dir)
    path = pack_dir / "hot_weights.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    state = dict(load_file(str(path)))
    embed_path = pack_dir / "embed_tokens.safetensors"
    if embed_path.is_file():
        state.update(load_file(str(embed_path)))
    return state


def resolve_latest_checkpoint(path: Path) -> Path:
    """Prefer warmup_end / dir with config.json; else newest checkpoint-*."""
    path = Path(path)
    if (path / "config.json").is_file() and (
        (path / "model.safetensors").is_file()
        or (path / "pytorch_model.bin").is_file()
        or any(path.glob("*.safetensors"))
    ):
        return path
    warmup = path / "warmup_end"
    if (warmup / "config.json").is_file():
        return warmup
    cands = sorted(
        (
            p
            for p in path.glob("checkpoint-*")
            if (p / "config.json").is_file()
        ),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if not cands:
        raise FileNotFoundError(f"No checkpoint under {path}")
    return cands[-1]
