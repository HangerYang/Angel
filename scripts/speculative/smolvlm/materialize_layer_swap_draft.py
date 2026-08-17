#!/usr/bin/env python3
"""Copy a progressive Eagle3 draft into a swap eval dir without mutating the ckpt.

Writes a patched ``config.json`` and symlinks ``model.safetensors``. Train-time
aux ids are stored as ``aux_hidden_states_layer_ids``; vLLM ids are ``id+1``.

Within-band candidates (train/HF ids):
  early {0, 4, 8}   mid {10, 12, 18}   late {20, 23, 25}
Also allowed for OFAT / baseline: early 1, mid 14, late 26 and 28.

Example:
  python scripts/speculative/smolvlm/materialize_layer_swap_draft.py \\
      --src output/smolvlm_256m_eagle3_progressive_nccl/checkpoint-66466 \\
      --dst output/smolvlm_256m_eagle3_progressive_nccl/layer_swap/e4_m12_l23 \\
      --aux_hidden_states_layer_ids 4,12,23
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TARGET_NUM_LAYERS = 30  # SmolVLM-256M text tower
EARLY_IDS: Tuple[int, ...] = (0, 4, 8)
MID_IDS: Tuple[int, ...] = (10, 12, 18)
LATE_IDS: Tuple[int, ...] = (20, 23, 25)
EARLY_ALLOWED: Tuple[int, ...] = EARLY_IDS + (1,)
MID_ALLOWED: Tuple[int, ...] = MID_IDS + (14,)
LATE_ALLOWED: Tuple[int, ...] = LATE_IDS + (26, 28)
BAND_CANDIDATES = {
    "early": EARLY_ALLOWED,
    "mid": MID_ALLOWED,
    "late": LATE_ALLOWED,
}


def _parse_ids(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("aux ids must be a non-empty comma-separated list")
    return [int(p) for p in parts]


def swap_name(aux_ids: Sequence[int]) -> str:
    e, m, l = (int(x) for x in aux_ids)
    return f"e{e}_m{m}_l{l}"


def iter_grid() -> Iterable[Tuple[str, List[int]]]:
    for e in EARLY_IDS:
        for m in MID_IDS:
            for l in LATE_IDS:
                ids = [e, m, l]
                yield swap_name(ids), ids


def iter_ofat() -> Iterable[Tuple[str, List[int]]]:
    """One-slot sweeps around trained [1, 14, 26], plus top-late 28."""
    designated = [1, 14, 26]
    extras = (
        [[e, 14, 26] for e in EARLY_IDS]
        + [[1, m, 26] for m in MID_IDS]
        + [[1, 14, l] for l in LATE_IDS]
        + [designated, [1, 14, 28]]
    )
    seen = set()
    for ids in extras:
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        yield swap_name(ids), ids


def iter_all() -> Iterable[Tuple[str, List[int]]]:
    seen = set()
    for name, ids in list(iter_grid()) + list(iter_ofat()):
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        yield name, ids


# One-slot OFAT on vLLM / eagle_aux indices (baseline 2-15-27).
OFAT_VLLM: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("baseline", (2, 15, 27)),
    ("early_0", (0, 15, 27)),
    ("early_4", (4, 15, 27)),
    ("early_8", (8, 15, 27)),
    ("mid_10", (2, 10, 27)),
    ("mid_12", (2, 12, 27)),
    ("mid_18", (2, 18, 27)),
    ("late_20", (2, 15, 20)),
    ("late_23", (2, 15, 23)),
    ("late_25", (2, 15, 25)),
)


def iter_ofat_vllm() -> Iterable[Tuple[str, List[int]]]:
    for name, ids in OFAT_VLLM:
        yield name, list(ids)


def validate_within_band(aux_ids: Sequence[int]) -> List[str]:
    if len(aux_ids) != 3:
        raise ValueError(f"expected 3 aux ids [early, mid, late], got {list(aux_ids)}")
    e, m, l = (int(x) for x in aux_ids)
    if e not in EARLY_ALLOWED:
        raise ValueError(f"early slot {e} not in {list(EARLY_ALLOWED)}")
    if m not in MID_ALLOWED:
        raise ValueError(f"mid slot {m} not in {list(MID_ALLOWED)}")
    if l not in LATE_ALLOWED:
        raise ValueError(f"late slot {l} not in {list(LATE_ALLOWED)}")
    return ["early", "mid", "late"]


def materialize(
    src: Path,
    dst: Path,
    aux_ids: Sequence[int] | None = None,
    *,
    eagle_aux_ids: Sequence[int] | None = None,
    name: str | None = None,
    designated_aux_ids: Sequence[int] = (1, 14, 26),
    validate_bands: bool = True,
) -> Dict[str, object]:
    src = src.resolve()
    dst = dst.resolve()
    config_src = src / "config.json"
    weight_src = src / "model.safetensors"
    if not config_src.is_file():
        raise FileNotFoundError(f"missing {config_src}")
    if not weight_src.is_file():
        raise FileNotFoundError(f"missing {weight_src}")

    if eagle_aux_ids is not None:
        eagle_aux_ids = [int(x) for x in eagle_aux_ids]
        if aux_ids is None:
            aux_ids = [i - 1 if i > 0 else 0 for i in eagle_aux_ids]
        else:
            aux_ids = [int(x) for x in aux_ids]
        bands = ["early", "mid", "late"]
        validate_bands = False
    else:
        if aux_ids is None:
            raise ValueError("need aux_ids or eagle_aux_ids")
        aux_ids = [int(x) for x in aux_ids]
        if validate_bands:
            bands = validate_within_band(aux_ids)
        else:
            bands = ["early", "mid", "late"]
        eagle_aux_ids = [i + 1 for i in aux_ids]
    designated = [int(x) for x in designated_aux_ids]

    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(config_src.read_text(encoding="utf-8"))
    cfg["aux_hidden_states_layer_ids"] = aux_ids
    cfg["eagle_aux_hidden_state_layer_ids"] = eagle_aux_ids
    (dst / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    weight_dst = dst / "model.safetensors"
    if weight_dst.exists() or weight_dst.is_symlink():
        weight_dst.unlink()
    weight_dst.symlink_to(weight_src)

    meta = {
        "src": str(src),
        "dst": str(dst),
        "name": name or swap_name(aux_ids),
        "designated_aux_hidden_states_layer_ids": designated,
        "aux_hidden_states_layer_ids": aux_ids,
        "eagle_aux_hidden_state_layer_ids": eagle_aux_ids,
        "slot_bands": bands,
        "band_candidates": {k: list(v) for k, v in BAND_CANDIDATES.items()},
    }
    (dst / "swap_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, help="Original draft checkpoint dir")
    p.add_argument("--dst", type=Path, help="Swap eval dir to create")
    p.add_argument(
        "--aux_hidden_states_layer_ids",
        help="Train/HF aux ids, comma-separated (e.g. 4,12,23)",
    )
    p.add_argument(
        "--designated_aux_hidden_states_layer_ids",
        default="1,14,26",
        help="Original trained aux ids (recorded in swap_meta.json)",
    )
    p.add_argument(
        "--list-grid",
        action="store_true",
        help="Print the 3x3x3 within-band names and ids, then exit",
    )
    p.add_argument(
        "--list-all",
        action="store_true",
        help="Print grid + designated OFAT/baseline names and ids, then exit",
    )
    p.add_argument(
        "--list-ofat-vllm",
        action="store_true",
        help="Print one-slot OFAT using vLLM eagle_aux ids (baseline 2,15,27)",
    )
    p.add_argument(
        "--eagle_aux_hidden_state_layer_ids",
        help="vLLM aux ids, comma-separated (e.g. 0,15,27). Skips train+1.",
    )
    p.add_argument("--name", default=None, help="Optional swap dir label")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_grid or args.list_all or args.list_ofat_vllm:
        if args.list_ofat_vllm:
            rows = iter_ofat_vllm()
        elif args.list_all:
            rows = iter_all()
        else:
            rows = iter_grid()
        for name, ids in rows:
            print(f"{name}\t{','.join(str(i) for i in ids)}")
        return 0
    if args.src is None or args.dst is None:
        raise SystemExit("--src and --dst are required")
    if args.eagle_aux_hidden_state_layer_ids:
        meta = materialize(
            args.src,
            args.dst,
            eagle_aux_ids=_parse_ids(args.eagle_aux_hidden_state_layer_ids),
            name=args.name,
            designated_aux_ids=_parse_ids(args.designated_aux_hidden_states_layer_ids),
        )
    elif args.aux_hidden_states_layer_ids:
        meta = materialize(
            args.src,
            args.dst,
            _parse_ids(args.aux_hidden_states_layer_ids),
            name=args.name,
            designated_aux_ids=_parse_ids(args.designated_aux_hidden_states_layer_ids),
        )
    else:
        raise SystemExit(
            "need --aux_hidden_states_layer_ids or --eagle_aux_hidden_state_layer_ids"
        )
    print("Layer-swap draft dir:")
    print(f"  name: {meta['name']}")
    print(f"  src: {meta['src']}")
    print(f"  dst: {meta['dst']}")
    print(f"  aux_hidden_states_layer_ids (train): {meta['aux_hidden_states_layer_ids']}")
    print(f"  eagle_aux_hidden_state_layer_ids (vLLM): {meta['eagle_aux_hidden_state_layer_ids']}")
    print(f"  slot_bands: {meta['slot_bands']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
