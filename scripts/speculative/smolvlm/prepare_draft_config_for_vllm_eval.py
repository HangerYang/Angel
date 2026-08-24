#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All Rights Reserved.
"""Ensure a trained SmolVLM Eagle3 draft checkpoint is ready for vLLM eval.

Reads ``num_hidden_layers`` (1 or multi-layer) and aux-layer ids from the draft
checkpoint and/or the AngelSlim train draft JSON, then writes any missing
vLLM-facing fields into ``<draft_model>/config.json``.

vLLM reads:
  - ``num_hidden_layers`` — draft depth (layer 0 = Eagle 2H, rest = standard H)
  - ``eagle_aux_hidden_state_layer_ids`` — target layers that feed fused HS

AngelSlim train uses ``aux_hidden_states_layer_ids`` (HF ``hs[id+1]``). If only
that list is present, this script sets ``eagle_aux = [id+1 for id in aux]``,
matching ``tools/train_eagle3_online.py``.

Usage:
  python scripts/speculative/smolvlm/prepare_draft_config_for_vllm_eval.py \\
      --draft_model output/smolvlm_256m_eagle3_online \\
      --draft_model_config_path \\
          angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _as_int_list(value: Any, field: str) -> Optional[List[int]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"{field} must be a non-empty list of ints, got {value!r}")
    return [int(x) for x in value]


def prepare_draft_config(
    draft_model: Path,
    draft_model_config_path: Optional[Path] = None,
    dry_run: bool = False,
    eagle_miracle_mode: bool = False,
    early_exit_threshold: float = -1.0,
    early_exit_min_layer: int = 0,
    early_exit_max_layer: int = -1,
) -> Dict[str, Any]:
    config_path = draft_model / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Draft checkpoint config not found: {config_path}. "
            "Train with a save strategy that writes the draft (e.g. "
            "SAVE_STRATEGY=epoch) or point --draft_model at a saved dir."
        )

    cfg = _load_json(config_path)
    train_cfg: Dict[str, Any] = {}
    if draft_model_config_path is not None:
        if not draft_model_config_path.is_file():
            raise FileNotFoundError(
                f"Train draft config not found: {draft_model_config_path}"
            )
        train_cfg = _load_json(draft_model_config_path)

    # Prefer checkpoint values; fill gaps from the train JSON used to build it.
    num_layers = cfg.get("num_hidden_layers", train_cfg.get("num_hidden_layers"))
    if num_layers is None:
        raise ValueError(
            "num_hidden_layers missing from draft config.json and train config. "
            "Set it explicitly (1 for single-layer, >1 for multi-layer draft)."
        )
    num_layers = int(num_layers)

    aux_ids = _as_int_list(
        cfg.get("aux_hidden_states_layer_ids", train_cfg.get("aux_hidden_states_layer_ids")),
        "aux_hidden_states_layer_ids",
    )
    eagle_aux_ids = _as_int_list(
        cfg.get(
            "eagle_aux_hidden_state_layer_ids",
            train_cfg.get("eagle_aux_hidden_state_layer_ids"),
        ),
        "eagle_aux_hidden_state_layer_ids",
    )
    if eagle_aux_ids is None and aux_ids is not None:
        eagle_aux_ids = [i + 1 for i in aux_ids]

    if eagle_aux_ids is None:
        raise ValueError(
            "Need eagle_aux_hidden_state_layer_ids (vLLM) or "
            "aux_hidden_states_layer_ids (train) in draft/train config."
        )

    draft_init = cfg.get(
        "draft_layer_init_from_target",
        train_cfg.get("draft_layer_init_from_target"),
    )
    if draft_init is not None:
        draft_init = _as_int_list(draft_init, "draft_layer_init_from_target")
        if len(draft_init) != num_layers:
            raise ValueError(
                f"draft_layer_init_from_target length ({len(draft_init)}) must "
                f"equal num_hidden_layers ({num_layers})"
            )

    updates: Dict[str, Any] = {
        "num_hidden_layers": num_layers,
        "eagle_aux_hidden_state_layer_ids": eagle_aux_ids,
    }
    if aux_ids is not None:
        updates["aux_hidden_states_layer_ids"] = aux_ids
    if draft_init is not None:
        updates["draft_layer_init_from_target"] = draft_init

    injection_mode = cfg.get(
        "eagle_aux_injection_mode",
        train_cfg.get("eagle_aux_injection_mode", "fused_fc"),
    )
    updates["eagle_aux_injection_mode"] = injection_mode
    if injection_mode == "progressive_staged":
        if aux_ids is not None and len(aux_ids) != num_layers:
            raise ValueError(
                "progressive_staged requires len(aux_hidden_states_layer_ids) "
                f"== num_hidden_layers ({num_layers}), got {len(aux_ids)}"
            )
        if eagle_aux_ids is not None and len(eagle_aux_ids) != num_layers:
            raise ValueError(
                "progressive_staged requires len(eagle_aux_hidden_state_layer_ids) "
                f"== num_hidden_layers ({num_layers}), got {len(eagle_aux_ids)}"
            )
        updates["num_aux_hidden_states"] = num_layers
        # progressive_per_layer_fc: each draft layer gets its own independent FC
        # over ALL aux streams. Propagate from checkpoint or train config.
        progressive_per_layer_fc = bool(
            cfg.get("progressive_per_layer_fc", train_cfg.get("progressive_per_layer_fc", False))
        )
        updates["progressive_per_layer_fc"] = progressive_per_layer_fc
        # progressive_fc_draft_feedback: step 1+ mirrors step 0 — concat all
        # draft layer outputs and project through the same per-layer FCs.
        # Requires progressive_per_layer_fc=True and a retrained checkpoint.
        progressive_fc_draft_feedback = bool(
            cfg.get(
                "progressive_fc_draft_feedback",
                train_cfg.get("progressive_fc_draft_feedback", False),
            )
        )
        updates["progressive_fc_draft_feedback"] = progressive_fc_draft_feedback
        # early_exit_bridges: residual MLPs for approximating deeper layers.
        # Trained for robustness to early-exit feedback chains.
        early_exit_bridges = bool(
            cfg.get("early_exit_bridges", train_cfg.get("early_exit_bridges", False))
        )
        updates["early_exit_bridges"] = early_exit_bridges
        if early_exit_bridges:
            # vLLM falls back to hidden_size when bridge_intermediate_size is
            # absent (llama_eagle3.py EarlyExitBridge construction); mirror that.
            default_bridge_mid = cfg.get("hidden_size", train_cfg.get("hidden_size"))
            if default_bridge_mid is None:
                raise ValueError(
                    "early_exit_bridges needs bridge_intermediate_size, or "
                    "hidden_size to fall back to, in the draft/train config."
                )
            bridge_mid = int(
                cfg.get(
                    "bridge_intermediate_size",
                    train_cfg.get("bridge_intermediate_size", default_bridge_mid),
                )
            )
            updates["bridge_intermediate_size"] = bridge_mid
            multi_depth_ce = cfg.get(
                "multi_depth_ce_weights",
                train_cfg.get("multi_depth_ce_weights", []),
            )
            if multi_depth_ce:
                updates["multi_depth_ce_weights"] = multi_depth_ce
    elif injection_mode == "progressive_banded_mix":
        # Banded mix: N aux streams (N > num_layers) are grouped into num_layers
        # bands; each band is softmax-mixed to one stream per draft layer.
        # num_aux_hidden_states = total number of aux streams (= len of flat bands).
        raw_bands = cfg.get("eagle_aux_layer_bands", train_cfg.get("eagle_aux_layer_bands"))
        if not raw_bands:
            raise ValueError(
                "progressive_banded_mix requires eagle_aux_layer_bands in draft "
                "config.json or train config. Run this script with "
                "--draft_model_config_path pointing at the train JSON."
            )
        flat_band_ids = [layer_id for band in raw_bands for layer_id in band]
        num_aux = len(flat_band_ids)
        if eagle_aux_ids is not None and len(eagle_aux_ids) != num_aux:
            raise ValueError(
                "progressive_banded_mix: len(eagle_aux_hidden_state_layer_ids) "
                f"({len(eagle_aux_ids)}) must equal total aux streams in bands "
                f"({num_aux})"
            )
        updates["num_aux_hidden_states"] = num_aux
        # Propagate band fields from checkpoint or train config if missing.
        for key in ("eagle_aux_layer_bands", "eagle_aux_band_init_layer_ids"):
            val = cfg.get(key, train_cfg.get(key))
            if val is not None:
                updates[key] = val
            elif key == "eagle_aux_layer_bands":
                raise ValueError(
                    f"progressive_banded_mix requires {key}; not found in "
                    "checkpoint config or train config."
                )
    elif injection_mode == "banded_mix_fc":
        # Same banding as progressive_banded_mix, but the mixed streams feed the
        # stock nH->H fusion FC of a 1-layer EAGLE 3.1 draft instead of being
        # injected per draft layer -- so the band count is independent of
        # num_hidden_layers and the bands may be unequal in size.
        raw_bands = cfg.get("eagle_aux_layer_bands", train_cfg.get("eagle_aux_layer_bands"))
        if not raw_bands:
            raise ValueError(
                "banded_mix_fc requires eagle_aux_layer_bands in draft "
                "config.json or train config. Run this script with "
                "--draft_model_config_path pointing at the train JSON."
            )
        flat_band_ids = [layer_id for band in raw_bands for layer_id in band]
        num_aux = len(flat_band_ids)
        if eagle_aux_ids is not None and len(eagle_aux_ids) != num_aux:
            raise ValueError(
                "banded_mix_fc: len(eagle_aux_hidden_state_layer_ids) "
                f"({len(eagle_aux_ids)}) must equal total aux streams in bands "
                f"({num_aux})"
            )
        updates["num_aux_hidden_states"] = num_aux
        for key in ("eagle_aux_layer_bands", "eagle_aux_band_init_layer_ids"):
            val = cfg.get(key, train_cfg.get(key))
            if val is not None:
                updates[key] = val
            elif key == "eagle_aux_layer_bands":
                raise ValueError(
                    f"banded_mix_fc requires {key}; not found in checkpoint "
                    "config or train config."
                )
    elif injection_mode in ("hawk", "real_hawk", "layer_skip_lora"):
        # Progressive hawk / real_hawk: one aux stream per draft layer.
        if aux_ids is not None and len(aux_ids) != num_layers:
            raise ValueError(
                f"{injection_mode} requires len(aux_hidden_states_layer_ids) "
                f"== num_hidden_layers ({num_layers}), got {len(aux_ids)}"
            )
        if eagle_aux_ids is not None and len(eagle_aux_ids) != num_layers:
            raise ValueError(
                f"{injection_mode} requires len(eagle_aux_hidden_state_layer_ids) "
                f"== num_hidden_layers ({num_layers}), got {len(eagle_aux_ids)}"
            )
        updates["num_aux_hidden_states"] = num_layers
        if injection_mode in ("real_hawk", "layer_skip_lora"):
            print(
                f"  WARNING: {injection_mode} train ckpts keep LoRA A/B — merge first:\n"
                "    python scripts/speculative/smolvlm/export_real_hawk_for_vllm.py \\\n"
                "      --draft_model <ckpt> --output_dir <merged>\n"
                "  Then eval the merged dir with eagle_aux_injection_mode=hawk."
            )
            # Still stamp hawk so a mistaken direct eval is closer to working
            # only after merge; leave mode as-is here and let export rewrite.
        print(
            "  hawk fuse: vLLM loads fuse_w1/fuse_w2; needs progressive patch "
            "(third_party/patches/10-vllm-v0.25.0-eagle3-progressive-staged.patch)."
        )

    for key in (
        "target_model_type",
        "modal_type",
        "draft_vocab_size",
        "image_token_id",
    ):
        if key not in cfg and key in train_cfg:
            updates[key] = train_cfg[key]

    # EAGLE 3.1: always stamp so vLLM's llama_eagle3.py can read the flags.
    updates["fc_norm"] = bool(cfg.get("fc_norm", train_cfg.get("fc_norm", False)))
    updates["norm_output"] = bool(
        cfg.get("norm_output", train_cfg.get("norm_output", False))
    )
    # QK-norm: vLLM builds self_attn.q_norm/k_norm only when this flag is on.
    updates["qk_norm"] = bool(cfg.get("qk_norm", train_cfg.get("qk_norm", False)))

    # Oracle gist conditioning. Only the "fc" injection point is loadable by
    # vLLM: it fuses the gist as one more FC stream and leaves layer 0 at 2H.
    # "qkv" makes layer 0 3-stream, which vLLM's EAGLE-3 layer cannot build.
    gist_on = bool(cfg.get("gist_conditioning", train_cfg.get("gist_conditioning", False)))
    if gist_on:
        gist_injection = str(
            cfg.get("gist_injection", train_cfg.get("gist_injection", "qkv"))
        )
        if gist_injection not in ("fc", "qkv", "both"):
            raise ValueError(
                f"unknown gist_injection {gist_injection!r} (expected fc/qkv/both)"
            )
        gist_mode = str(cfg.get("gist_mode", train_cfg.get("gist_mode", "remaining")))
        if gist_mode != "whole":
            raise ValueError(
                f"gist_mode={gist_mode!r} cannot be served at decode time: it "
                "refreshes the oracle every few tokens against text that does "
                "not exist yet. Only gist_mode='whole' (one vector per request) "
                "is supportable in vLLM."
            )
        gist_dim = int(cfg.get("gist_embedding_dim", train_cfg.get("gist_embedding_dim", 0)) or 0)
        if gist_dim <= 0:
            raise ValueError("gist_conditioning requires gist_embedding_dim > 0")
        updates["gist_conditioning"] = True
        updates["gist_injection"] = gist_injection
        updates["gist_mode"] = "whole"
        updates["gist_embedding_dim"] = gist_dim
    else:
        updates["gist_conditioning"] = False

    # Per-depth staleness fallback: on an early exit the skipped depths keep the
    # previous step's inject at that depth, matching how the draft was trained
    # (see take_progressive_draft_feedback's sim_exit_depth path). Without this,
    # eval fills them by repeating the exiting layer's output instead.
    updates["stale_depth_fallback"] = bool(
        cfg.get("stale_depth_fallback", train_cfg.get("stale_depth_fallback", False))
    )

    # Early exit: -1 disables; positive value is the min-softmax-confidence threshold.
    if early_exit_threshold > 0:
        updates["early_exit_threshold"] = float(early_exit_threshold)
        updates["early_exit_max_layer"] = int(early_exit_max_layer)
        updates["early_exit_min_layer"] = int(early_exit_min_layer)
    else:
        # Explicitly clear any previously stamped threshold so the checkpoint is
        # comparable to a no-early-exit run without needing to re-prep.
        updates["early_exit_threshold"] = -1.0
        updates.pop("early_exit_min_layer", None)
        cfg.pop("early_exit_min_layer", None)

    if eagle_miracle_mode:
        # Miracle (oracle GT-HS) works for fused_fc, progressive_staged, hawk.
        updates["eagle_miracle_mode"] = True

    changed = {k: v for k, v in updates.items() if cfg.get(k) != v}
    cfg.update(updates)
    if eagle_miracle_mode and "eagle_assistance_mode" in cfg:
        cfg.pop("eagle_assistance_mode", None)
        changed["eagle_assistance_mode"] = None

    print("SmolVLM Eagle3 draft config for vLLM eval:")
    print(f"  draft_model: {draft_model}")
    print(f"  eagle_aux_injection_mode: {injection_mode}")
    print(f"  eagle_miracle_mode: {cfg.get('eagle_miracle_mode', False)}")
    print(f"  fc_norm: {cfg.get('fc_norm', False)}")
    if cfg.get("gist_conditioning"):
        print(
            f"  gist: injection={cfg.get('gist_injection')} "
            f"mode={cfg.get('gist_mode')} dim={cfg.get('gist_embedding_dim')} "
            "(ORACLE -- acceptance is an upper bound, not a deployable speedup)"
        )
    print(f"  norm_output: {cfg.get('norm_output', False)}")
    print(f"  qk_norm: {cfg.get('qk_norm', False)}")
    print(f"  early_exit_threshold: {cfg.get('early_exit_threshold', -1.0)}"
          + (f"  early_exit_min_layer: {cfg.get('early_exit_min_layer', 0)}"
             if cfg.get("early_exit_threshold", -1.0) > 0 else "  (disabled)"))
    if cfg.get("progressive_per_layer_fc"):
        print(f"  progressive_per_layer_fc: True")
    if cfg.get("progressive_fc_draft_feedback"):
        print(f"  progressive_fc_draft_feedback: True (symmetric FC feedback)")
    if cfg.get("early_exit_bridges"):
        print(f"  early_exit_bridges: True (residual MLPs for early-exit approximation)")
        print(f"    bridge_intermediate_size: {cfg.get('bridge_intermediate_size')}")
        if cfg.get("multi_depth_ce_weights"):
            print(f"    multi_depth_ce_weights: {cfg.get('multi_depth_ce_weights')}")
    print(f"  num_hidden_layers: {num_layers} "
          f"({'single-layer' if num_layers == 1 else f'{num_layers}-layer'} draft)")
    print(f"  num_aux_hidden_states: {cfg.get('num_aux_hidden_states')}")
    print(f"  aux_hidden_states_layer_ids (train): {cfg.get('aux_hidden_states_layer_ids')}")
    print(f"  eagle_aux_hidden_state_layer_ids (vLLM): "
          f"{cfg.get('eagle_aux_hidden_state_layer_ids')}")
    if injection_mode in ("progressive_banded_mix", "banded_mix_fc"):
        print(f"  eagle_aux_layer_bands: {cfg.get('eagle_aux_layer_bands')}")
        print(f"  eagle_aux_band_init_layer_ids: {cfg.get('eagle_aux_band_init_layer_ids')}")
    _vt = cfg.get("vistoken_compress")
    if _vt:
        # Carried through untouched; vLLM auto-arms the row compressor on it.
        print(f"  vistoken_compress: {_vt}")
    if draft_init is not None:
        print(f"  draft_layer_init_from_target: {draft_init}")
    if changed:
        print(f"  updating config.json fields: {sorted(changed)}")
        if not dry_run:
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  wrote {config_path}")
    else:
        print("  config.json already has required fields; no write needed")

    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--draft_model",
        type=Path,
        required=True,
        help="Path to trained draft checkpoint directory (contains config.json)",
    )
    p.add_argument(
        "--draft_model_config_path",
        type=Path,
        default=None,
        help="Optional AngelSlim train draft JSON to fill missing fields",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate/print only; do not rewrite config.json",
    )
    p.add_argument(
        "--eagle_miracle_mode",
        action="store_true",
        help=(
            "Stamp eagle_miracle_mode=true into draft config.json "
            "(fused_fc / progressive_staged / hawk)."
        ),
    )
    p.add_argument(
        "--early_exit_threshold",
        type=float,
        default=-1.0,
        help=(
            "Min softmax confidence to exit draft early (skip remaining layers). "
            "-1 disables early exit (default). Typical value: 0.8."
        ),
    )
    p.add_argument(
        "--early_exit_min_layer",
        type=int,
        default=0,
        help=(
            "Earliest draft layer index after which early exit may trigger. "
            "Default 0 = check after every non-final layer."
        ),
    )
    p.add_argument(
        "--early_exit_max_layer",
        type=int,
        default=-1,
        help=(
            "Highest draft layer index at which the confidence check runs. "
            "Each check costs a GPU->CPU sync, so checking a layer that "
            "rarely exits is a net loss. -1 = every non-final layer."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    prepare_draft_config(
        draft_model=args.draft_model,
        draft_model_config_path=args.draft_model_config_path,
        dry_run=args.dry_run,
        eagle_miracle_mode=args.eagle_miracle_mode,
        early_exit_threshold=args.early_exit_threshold,
        early_exit_min_layer=args.early_exit_min_layer,
        early_exit_max_layer=args.early_exit_max_layer,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
