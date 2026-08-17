#!/usr/bin/env python3
"""Compare SmolVLM target/draft visual attention on eval samples.

The main score is text-to-image attention mass: for text query tokens, how much
attention probability lands on image-token keys. It also reports a uniform
causal baseline so a score can be read as above/below chance.

Defaults compare:
  - final hawk: output/aux_experiments/hawk_feature_match_from_warmup/checkpoint-66466
  - regular eagle: output/smolvlm_256m_eagle3_nccl/checkpoint-66466
  - progressive eagle: output/aux_experiments/progressive_threshold/checkpoint-66466
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from angelslim.compressor.speculative.train.data.chat_templates import (  # noqa: E402
    ChatTemplateType,
)
from angelslim.compressor.speculative.train.data.dataset_builder.online_dataset_builder import (  # noqa: E402
    OnlineSmolVLMDatasetBuilder,
)
from angelslim.compressor.speculative.train.models.draft import (  # noqa: E402
    DraftModelConfig,
    create_draft_model,
)


@dataclass
class RunSpec:
    name: str
    checkpoint: Path


def _default_runs() -> List[RunSpec]:
    return [
        RunSpec(
            "final_hawk",
            _REPO_ROOT
            / "output/aux_experiments/hawk_feature_match_from_warmup/checkpoint-66466",
        ),
        RunSpec(
            "regular_eagle",
            _REPO_ROOT / "output/smolvlm_256m_eagle3_nccl/checkpoint-66466",
        ),
        RunSpec(
            "progressive_eagle",
            _REPO_ROOT
            / "output/aux_experiments/progressive_threshold/checkpoint-66466",
        ),
    ]


def _load_state_dict(ckpt: Path) -> Dict[str, torch.Tensor]:
    safes = sorted(ckpt.glob("*.safetensors"))
    if safes:
        from safetensors.torch import load_file

        out: Dict[str, torch.Tensor] = {}
        for path in safes:
            out.update(load_file(str(path)))
        return out
    bin_path = ckpt / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No safetensors or pytorch_model.bin under {ckpt}")


def _load_draft(ckpt: Path, device: torch.device, dtype: torch.dtype):
    cfg_path = ckpt / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing draft config: {cfg_path}")
    cfg = DraftModelConfig.from_file(cfg_path)
    draft = create_draft_model(cfg)
    state = _load_state_dict(ckpt)
    missing, unexpected = draft.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  {ckpt.name}: unexpected draft keys={len(unexpected)} e.g. {unexpected[:3]}")
    required_missing = [
        k for k in missing
        if not (k.startswith("t2d") or k.startswith("d2t"))
    ]
    if required_missing:
        print(f"  {ckpt.name}: missing draft keys={len(required_missing)} e.g. {required_missing[:5]}")
    draft.to(device=device, dtype=dtype)
    draft.eval()
    for p in draft.parameters():
        p.requires_grad_(False)
    return draft, cfg


def _build_loss_query_mask(
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> torch.Tensor:
    q = (loss_mask > 0) & (attention_mask > 0) & (~image_mask)
    if not q.any():
        q = (attention_mask > 0) & (~image_mask)
    return q


def _image_attention_score(
    attn: torch.Tensor,
    query_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> Optional[float]:
    profile = _image_attention_profile(attn, query_mask, image_mask)
    return None if profile is None else float(profile["mass"])


def _image_attention_profile(
    attn: torch.Tensor,
    query_mask: torch.Tensor,
    image_mask: torch.Tensor,
    eps: float = 1e-12,
) -> Optional[Dict[str, float]]:
    """Text-query → image-key attention profile (head-averaged).

    mass: total probability on image keys (not renormalized).
    Remaining stats are over the renormalized distribution on image keys.
    """
    if not image_mask.any() or not query_mask.any():
        return None
    attn_mean = attn.float().mean(dim=1)  # [B, S, S]
    masses: List[float] = []
    entropies: List[float] = []
    top1s: List[float] = []
    top5s: List[float] = []
    effs: List[float] = []
    for b in range(attn_mean.shape[0]):
        q = query_mask[b].bool()
        k = image_mask[b].bool()
        if not q.any() or not k.any():
            continue
        # [Nq, Nk]
        img = attn_mean[b, q][:, k]
        mass_q = img.sum(dim=-1)
        masses.append(float(mass_q.mean().item()))
        n_k = int(k.sum().item())
        log_n = math.log(max(n_k, 2))
        for row, m in zip(img, mass_q):
            if float(m.item()) <= eps:
                continue
            p = (row / m.clamp_min(eps)).clamp_min(eps)
            p = p / p.sum()
            H = float((-(p * p.log()).sum()).item())
            H_norm = H / log_n
            entropies.append(H_norm)
            top1s.append(float(p.max().item()))
            topk = min(5, n_k)
            top5s.append(float(p.topk(topk).values.sum().item()))
            effs.append(float(math.exp(H) / max(n_k, 1)))
    if not masses or not entropies:
        return None
    H_mean = sum(entropies) / len(entropies)
    return {
        "mass": sum(masses) / len(masses),
        "norm_entropy": H_mean,
        "concentration": 1.0 - H_mean,
        "mass_x_concentration": (sum(masses) / len(masses)) * (1.0 - H_mean),
        "top1_image_share": sum(top1s) / len(top1s),
        "top5_image_share": sum(top5s) / len(top5s),
        "effective_image_fraction": sum(effs) / len(effs),
    }


def _image_attention_map_similarity(
    draft_attn: torch.Tensor,
    target_attn: torch.Tensor,
    query_mask: torch.Tensor,
    image_mask: torch.Tensor,
    eps: float = 1e-8,
) -> Optional[Dict[str, float]]:
    """Draft↔teacher similarity on renormalized image-key attention maps."""
    if not image_mask.any() or not query_mask.any():
        return None
    d = draft_attn.float().mean(dim=1)  # [B, S, S]
    t = target_attn.float().mean(dim=1)
    s = min(d.shape[-1], t.shape[-1])
    d = d[..., :s]
    t = t[..., :s]
    cos_vals: List[float] = []
    kl_dt: List[float] = []  # KL(draft || target) — matches training
    kl_td: List[float] = []
    l1_vals: List[float] = []
    for b in range(d.shape[0]):
        q = query_mask[b].bool()
        k = image_mask[b].bool()
        if not q.any() or not k.any():
            continue
        q_idx = torch.nonzero(q, as_tuple=False).flatten()
        k_idx = torch.nonzero(k, as_tuple=False).flatten()
        # Keep queries whose positions exist in both maps.
        q_idx = q_idx[q_idx < s]
        k_idx = k_idx[k_idx < s]
        if q_idx.numel() == 0 or k_idx.numel() == 0:
            continue
        for qi in q_idx.tolist():
            d_row = d[b, qi, k_idx]
            t_row = t[b, qi, k_idx]
            d_sum = float(d_row.sum().item())
            t_sum = float(t_row.sum().item())
            if d_sum <= eps or t_sum <= eps:
                continue
            pd = (d_row / d_sum).clamp_min(eps)
            pt = (t_row / t_sum).clamp_min(eps)
            pd = pd / pd.sum()
            pt = pt / pt.sum()
            cos_vals.append(float(F.cosine_similarity(pd, pt, dim=0).item()))
            kl_dt.append(float((pd * (pd.log() - pt.log())).sum().item()))
            kl_td.append(float((pt * (pt.log() - pd.log())).sum().item()))
            l1_vals.append(float((pd - pt).abs().sum().item()))
    if not cos_vals:
        return None
    return {
        "cosine": sum(cos_vals) / len(cos_vals),
        "kl_draft_target": sum(kl_dt) / len(kl_dt),
        "kl_target_draft": sum(kl_td) / len(kl_td),
        "l1": sum(l1_vals) / len(l1_vals),
        "n_query_pairs": float(len(cos_vals)),
    }


def _accum_profile(
    store: Dict[int, Dict[str, List[float]]],
    layer_id: int,
    profile: Optional[Dict[str, float]],
) -> None:
    if profile is None:
        return
    bucket = store.setdefault(layer_id, {k: [] for k in profile if k != "n_query_pairs"})
    for k, v in profile.items():
        if k == "n_query_pairs":
            continue
        bucket.setdefault(k, []).append(float(v))


def _finalize_profile(store: Dict[int, Dict[str, List[float]]]) -> Dict[int, Dict[str, float]]:
    return {
        lid: {k: float(sum(vs) / len(vs)) for k, vs in vals.items() if vs}
        for lid, vals in store.items()
    }


def _uniform_visible_image_baseline(
    attention_mask: torch.Tensor,
    query_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> Optional[float]:
    if not image_mask.any() or not query_mask.any():
        return None
    vals = []
    bsz, seq_len = attention_mask.shape
    for b in range(bsz):
        valid = attention_mask[b].bool()
        images = image_mask[b].bool()
        for q in torch.nonzero(query_mask[b], as_tuple=False).flatten().tolist():
            visible = valid.clone()
            visible[q + 1 :] = False
            denom = int(visible.sum().item())
            if denom <= 0:
                continue
            vals.append(float((visible & images).sum().item()) / float(denom))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _modality_gap(
    hidden: torch.Tensor,
    query_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> Optional[Dict[str, float]]:
    if not image_mask.any() or not query_mask.any():
        return None
    rows = []
    for b in range(hidden.shape[0]):
        im = image_mask[b]
        tx = query_mask[b]
        if not im.any() or not tx.any():
            continue
        h = hidden[b].float()
        im_mean = h[im].mean(dim=0)
        tx_mean = h[tx].mean(dim=0)
        cos = F.cosine_similarity(im_mean, tx_mean, dim=0).item()
        rows.append(
            {
                "cosine": float(cos),
                "cosine_distance": float(1.0 - cos),
                "image_norm": float(im_mean.norm().item()),
                "text_norm": float(tx_mean.norm().item()),
            }
        )
    if not rows:
        return None
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def _mean(vals: Iterable[float]) -> Optional[float]:
    vals = list(vals)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _accum_scalar(store: Dict[int, List[float]], layer_id: int, value: Optional[float]) -> None:
    if value is None:
        return
    store.setdefault(layer_id, []).append(float(value))


def _accum_gap(
    store: Dict[int, Dict[str, List[float]]],
    layer_id: int,
    gap: Optional[Dict[str, float]],
) -> None:
    if gap is None:
        return
    bucket = store.setdefault(layer_id, {k: [] for k in gap})
    for k, v in gap.items():
        bucket.setdefault(k, []).append(float(v))


def _finalize_gap(store: Dict[int, Dict[str, List[float]]]) -> Dict[int, Dict[str, float]]:
    return {
        lid: {k: float(sum(vs) / len(vs)) for k, vs in vals.items() if vs}
        for lid, vals in store.items()
    }


def _rank(score_map: Dict[int, float], reverse: bool = True) -> List[Dict[str, Any]]:
    rows = [
        {"layer_id": int(lid), "score": float(score)}
        for lid, score in sorted(score_map.items())
    ]
    rows.sort(key=lambda r: r["score"], reverse=reverse)
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def _closest_target_layers(
    draft_scores: Dict[int, float],
    target_scores: Dict[int, float],
    top_k: int = 5,
) -> Dict[int, List[Dict[str, float]]]:
    out = {}
    for dlid, dscore in draft_scores.items():
        rows = []
        for tlid, tscore in target_scores.items():
            rows.append(
                {
                    "target_layer_id": int(tlid),
                    "target_score": float(tscore),
                    "abs_gap": float(abs(dscore - tscore)),
                }
            )
        rows.sort(key=lambda r: r["abs_gap"])
        out[int(dlid)] = rows[:top_k]
    return out


def _sense_label(score: Optional[float], baseline: Optional[float]) -> str:
    if score is None:
        return "no_image_samples"
    if baseline is None or baseline <= 0:
        return "no_visible_image_baseline"
    ratio = score / baseline
    if ratio >= 1.25:
        return "above_uniform"
    if ratio <= 0.75:
        return "below_uniform"
    return "near_uniform"


def _parse_runs(spec: Optional[str]) -> List[RunSpec]:
    if not spec:
        return _default_runs()
    runs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).name
        runs.append(RunSpec(name.strip(), Path(path).expanduser().resolve()))
    return runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    p.add_argument(
        "--data-path",
        default=str(_REPO_ROOT / "dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl"),
    )
    p.add_argument(
        "--runs",
        default=None,
        help="Comma list name=/path/to/checkpoint. Defaults to final hawk, regular eagle, progressive eagle.",
    )
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "draft_visual_attention"),
    )
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-only", action="store_true")
    p.add_argument("--no-shuffle", action="store_true", help="Use the first N rows instead of shuffled eval sampling")
    p.add_argument("--image-only", action="store_true", help="Analyze only examples with images")
    p.add_argument(
        "--sample-pool-size",
        type=int,
        default=None,
        help="Raw shuffled examples to preprocess before image-only filtering; default=max_samples*3",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"Loading target {args.model_path} on {device} dtype={args.dtype}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    target = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
        attn_implementation="eager",
    )
    target.to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    image_token_id = int(getattr(target.config, "image_token_id", 49190))
    num_target_layers = int(target.config.text_config.num_hidden_layers)

    print(f"Building dataset {args.data_path} max_samples={args.max_samples}")
    builder = OnlineSmolVLMDatasetBuilder(
        tokenizer=processor,
        max_length=args.max_length,
        shuffle_seed=args.seed,
        chat_template_type=ChatTemplateType.SMOLVLM,
        display=False,
    )
    sample_num = args.max_samples
    if args.image_only and args.max_samples is not None:
        sample_num = args.sample_pool_size or max(args.max_samples * 3, args.max_samples)
    ds = builder.build_dataset(
        args.data_path,
        num_proc=args.num_proc,
        shuffle=not args.no_shuffle,
        sample_num=sample_num,
        load_from_cache_file=False,
    )
    if args.image_only:
        ds = ds.filter(
            lambda batch: [bool(p) and p != "[]" for p in batch["image_paths"]],
            batched=True,
            num_proc=args.num_proc,
            load_from_cache_file=False,
            desc="Filtering image-only samples",
        )
        if args.max_samples is not None and 0 < args.max_samples < len(ds):
            ds = ds.select(range(args.max_samples))
        if args.max_samples is not None and len(ds) < args.max_samples:
            print(
                f"WARNING: requested {args.max_samples} image samples but only "
                f"found {len(ds)} in sample_pool_size={sample_num}."
            )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=builder.get_data_collator())

    run_specs = [] if args.target_only else _parse_runs(args.runs)
    drafts = []
    for spec in run_specs:
        print(f"Loading draft {spec.name}: {spec.checkpoint}")
        draft, cfg = _load_draft(spec.checkpoint, device=device, dtype=dtype)
        drafts.append((spec, draft, cfg))

    target_attn_accum: Dict[int, List[float]] = {}
    target_gap_accum: Dict[int, Dict[str, List[float]]] = {}
    target_profile_accum: Dict[int, Dict[str, List[float]]] = {}
    baseline_vals: List[float] = []
    draft_attn_accum: Dict[str, Dict[int, List[float]]] = {s.name: {} for s, _, _ in drafts}
    draft_gap_accum: Dict[str, Dict[int, Dict[str, List[float]]]] = {s.name: {} for s, _, _ in drafts}
    draft_profile_accum: Dict[str, Dict[int, Dict[str, List[float]]]] = {
        s.name: {} for s, _, _ in drafts
    }
    draft_sim_accum: Dict[str, Dict[int, Dict[str, List[float]]]] = {
        s.name: {} for s, _, _ in drafts
    }
    n_image = 0
    n_text = 0

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        image_mask = input_ids == image_token_id
        query_mask = _build_loss_query_mask(loss_mask, attention_mask, image_mask)
        has_image = bool(image_mask.any().item())
        n_image += int(has_image)
        n_text += int(not has_image)

        fwd: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "output_attentions": has_image,
            "use_cache": False,
        }
        if "pixel_values" in batch and batch["pixel_values"] is not None:
            fwd["pixel_values"] = batch["pixel_values"].to(device)
        if "pixel_attention_mask" in batch and batch["pixel_attention_mask"] is not None:
            fwd["pixel_attention_mask"] = batch["pixel_attention_mask"].to(device)

        with torch.no_grad():
            target_out = target(**fwd)

        if has_image:
            base = _uniform_visible_image_baseline(attention_mask, query_mask, image_mask)
            if base is not None:
                baseline_vals.append(base)
            if target_out.attentions is not None:
                for lid, attn in enumerate(target_out.attentions):
                    profile = _image_attention_profile(attn, query_mask, image_mask)
                    _accum_scalar(
                        target_attn_accum,
                        lid,
                        None if profile is None else profile["mass"],
                    )
                    _accum_profile(target_profile_accum, lid, profile)

        for lid in range(num_target_layers):
            _accum_gap(
                target_gap_accum,
                lid,
                _modality_gap(target_out.hidden_states[lid + 1], query_mask, image_mask),
            )

        aux_cache: Dict[Tuple[int, ...], torch.Tensor] = {}
        for spec, draft, cfg in drafts:
            aux_ids = tuple(int(x) for x in getattr(cfg, "aux_hidden_states_layer_ids", [1, 14, 26]))
            if aux_ids not in aux_cache:
                aux_cache[aux_ids] = torch.cat(
                    [target_out.hidden_states[i + 1] for i in aux_ids], dim=-1
                )
            with torch.no_grad():
                d_out = draft(
                    hidden_states=aux_cache[aux_ids].to(dtype=next(draft.parameters()).dtype),
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=has_image,
                    output_hidden_states=True,
                )
            if has_image and target_out.attentions is not None:
                for dlid, attn in enumerate(d_out["attentions"]):
                    profile = _image_attention_profile(attn, query_mask, image_mask)
                    _accum_scalar(
                        draft_attn_accum[spec.name],
                        dlid,
                        None if profile is None else profile["mass"],
                    )
                    _accum_profile(draft_profile_accum[spec.name], dlid, profile)
                    intended = aux_ids[dlid] if dlid < len(aux_ids) else None
                    if intended is not None and 0 <= intended < len(target_out.attentions):
                        sim = _image_attention_map_similarity(
                            attn,
                            target_out.attentions[intended],
                            query_mask,
                            image_mask,
                        )
                        _accum_profile(draft_sim_accum[spec.name], dlid, sim)
            for dlid, hs in enumerate(d_out["hidden_states"]):
                _accum_gap(
                    draft_gap_accum[spec.name],
                    dlid,
                    _modality_gap(hs, query_mask, image_mask),
                )

        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (step + 1) % 25 == 0 or (step + 1) == len(ds):
            print(f"  processed {step + 1}/{len(ds)} image={n_image} text_only={n_text}")

    target_scores = {
        lid: v for lid, vals in target_attn_accum.items() if (v := _mean(vals)) is not None
    }
    target_profiles = _finalize_profile(target_profile_accum)
    baseline = _mean(baseline_vals)
    target_ranking = _rank(target_scores)

    runs_summary: Dict[str, Any] = {}
    quality_runs: Dict[str, Any] = {}
    rows = []
    for spec, draft, cfg in drafts:
        aux_ids = [int(x) for x in getattr(cfg, "aux_hidden_states_layer_ids", [])]
        dscores = {
            lid: v
            for lid, vals in draft_attn_accum[spec.name].items()
            if (v := _mean(vals)) is not None
        }
        dprofiles = _finalize_profile(draft_profile_accum[spec.name])
        dsims = _finalize_profile(draft_sim_accum[spec.name])
        closest = _closest_target_layers(dscores, target_scores)
        gaps = _finalize_gap(draft_gap_accum[spec.name])
        rows_for_run = []
        draft_quality_layers: Dict[str, Any] = {}
        for dlid, score in sorted(dscores.items()):
            closest0 = closest.get(dlid, [{}])[0]
            ratio = None if baseline in (None, 0) else score / baseline
            row = {
                "run": spec.name,
                "draft_layer_id": dlid,
                "draft_visual_attention": score,
                "uniform_visible_image_baseline": baseline,
                "attention_over_uniform": ratio,
                "sense_label": _sense_label(score, baseline),
                "closest_target_layer": closest0.get("target_layer_id"),
                "closest_target_score": closest0.get("target_score"),
                "closest_abs_gap": closest0.get("abs_gap"),
            }
            intended = aux_ids[dlid] if dlid < len(aux_ids) else None
            prof = dprofiles.get(dlid, {})
            sim = dsims.get(dlid, {})
            tprof = target_profiles.get(intended, {}) if intended is not None else {}
            for k, v in prof.items():
                row[k] = v
            for k, v in sim.items():
                row[f"map_{k}"] = v
            if intended is not None:
                row["intended_target_layer"] = intended
                for k in (
                    "mass",
                    "norm_entropy",
                    "concentration",
                    "mass_x_concentration",
                    "top1_image_share",
                    "top5_image_share",
                    "effective_image_fraction",
                ):
                    if k in tprof:
                        row[f"target_{k}"] = tprof[k]
            rows.append(row)
            rows_for_run.append(row)
            q_entry = dict(prof)
            if intended is not None:
                q_entry["intended_target_layer"] = intended
                for k, v in tprof.items():
                    q_entry[f"target_{k}"] = v
            q_entry.update({f"map_{k}": v for k, v in sim.items()})
            draft_quality_layers[str(dlid)] = q_entry
        runs_summary[spec.name] = {
            "checkpoint": str(spec.checkpoint),
            "mode": getattr(cfg, "eagle_aux_injection_mode", "fused_fc"),
            "num_draft_layers": int(getattr(cfg, "num_hidden_layers", len(dscores))),
            "aux_hidden_states_layer_ids": list(getattr(cfg, "aux_hidden_states_layer_ids", [])),
            "draft_attention_by_layer": dscores,
            "draft_attention_ranking": _rank(dscores),
            "closest_target_layers_by_attention": closest,
            "draft_modality_gap_by_layer": gaps,
            "draft_attention_profile_by_layer": dprofiles,
            "draft_vs_intended_map_similarity": dsims,
            "summary_rows": rows_for_run,
        }
        quality_runs[spec.name] = {
            "aux_hidden_states_layer_ids": aux_ids,
            "draft_layers": draft_quality_layers,
        }

    target_gap = _finalize_gap(target_gap_accum)
    target_top = target_ranking[:10]
    result = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "max_samples": args.max_samples,
        "sample_pool_size": sample_num,
        "image_only": args.image_only,
        "shuffle": not args.no_shuffle,
        "seed": args.seed,
        "n_samples": n_image + n_text,
        "n_image": n_image,
        "n_text_only": n_text,
        "image_token_id": image_token_id,
        "uniform_visible_image_baseline": baseline,
        "target": {
            "num_layers": num_target_layers,
            "visual_attention_by_layer": target_scores,
            "visual_attention_ranking": target_ranking,
            "top_visual_attention_layers": target_top,
            "sense_label": _sense_label(target_top[0]["score"] if target_top else None, baseline),
            "modality_gap_by_layer": target_gap,
            "attention_profile_by_layer": {
                str(k): v for k, v in sorted(target_profiles.items())
            },
        },
        "runs": runs_summary,
        "note": (
            "visual_attention = mean over examples/heads/text query tokens of total attention mass "
            "assigned to image-token keys. uniform_visible_image_baseline is the causal visible "
            "image-key fraction for the same query tokens. closest_target_layers_by_attention uses "
            "absolute difference in this scalar attention score, not representation similarity. "
            "map_cosine / map_kl_* compare renormalized image-key attention distributions between "
            "each draft layer and its intended aux target layer. "
            "modality_gap is teacher-forced on eval samples, not timed vLLM acceptance-loop telemetry."
        ),
    }

    quality_payload = {
        "n_samples": n_image + n_text,
        "n_image": n_image,
        "target": {str(k): v for k, v in sorted(target_profiles.items())},
        "runs": quality_runs,
        "note": (
            "Per-layer profiles: mass + renormalized-image-key entropy/concentration/top-k. "
            "map_* metrics: draft↔intended-target similarity on those renormalized maps "
            "(cosine, KL(draft||target), KL(target||draft), L1)."
        ),
    }

    json_path = out_dir / "draft_visual_attention_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    quality_path = out_dir / "draft_visual_attention_quality.json"
    with quality_path.open("w", encoding="utf-8") as f:
        json.dump(quality_payload, f, indent=2)

    csv_path = out_dir / "draft_visual_attention_summary.csv"
    # Union of keys across rows for stable CSV.
    fieldnames = [
        "run",
        "draft_layer_id",
        "draft_visual_attention",
        "uniform_visible_image_baseline",
        "attention_over_uniform",
        "sense_label",
        "closest_target_layer",
        "closest_target_score",
        "closest_abs_gap",
        "intended_target_layer",
        "mass",
        "norm_entropy",
        "concentration",
        "mass_x_concentration",
        "top1_image_share",
        "top5_image_share",
        "effective_image_fraction",
        "target_mass",
        "target_norm_entropy",
        "target_concentration",
        "target_mass_x_concentration",
        "target_top1_image_share",
        "target_top5_image_share",
        "target_effective_image_fraction",
        "map_cosine",
        "map_kl_draft_target",
        "map_kl_target_draft",
        "map_l1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    target_csv = out_dir / "target_visual_attention_ranking.csv"
    with target_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "layer_id", "score"])
        writer.writeheader()
        writer.writerows(target_ranking)

    print(f"\nTarget top visual-attention layers: {target_top[:5]}")
    for row in rows:
        cos = row.get("map_cosine")
        cos_s = f" cos={cos:.3f}" if isinstance(cos, float) else ""
        print(
            f"{row['run']} L{row['draft_layer_id']}: attn={row['draft_visual_attention']:.4f} "
            f"H={row.get('norm_entropy', float('nan')):.3f} "
            f"ratio={row['attention_over_uniform']:.2f} label={row['sense_label']} "
            f"closest target L{row['closest_target_layer']}{cos_s}"
        )
    print(f"Wrote {json_path}")
    print(f"Wrote {quality_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {target_csv}")


if __name__ == "__main__":
    main()
