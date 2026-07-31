#!/usr/bin/env python3
"""Target-only layer importance analysis for SmolVLM / Idefics3.

Ranks transformer layers by several proxies (CE activation grads, agreement,
embedding change, HS information, image attention). Image attention is skipped
automatically on text-only samples.

Example:
  python analyze_target_layers.py \\
    --model-path HuggingFaceTB/SmolVLM-256M-Instruct \\
    --data-path /home/hyang/AngelSlim/dataset/smolvlm_256m_target_gen/data_0-36.jsonl \\
    --output-dir ./outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

# Repo root: .../AngelSlim
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from angelslim.compressor.speculative.train.data.chat_templates import (  # noqa: E402
    ChatTemplateType,
)
from angelslim.compressor.speculative.train.data.dataset_builder.online_dataset_builder import (  # noqa: E402
    OnlineSmolVLMDatasetBuilder,
)


@dataclass
class LayerScores:
    layer_id: int
    ce_grad: float = 0.0
    agreement_kl: float = 0.0  # lower = closer to final logits
    agreement_ce: float = 0.0  # lower = better next-token CE from this layer
    delta_rel: float = 0.0  # ||h_l - h_{l-1}|| / ||h_{l-1}||
    info_eff_rank: float = 0.0
    info_var: float = 0.0
    image_attn: Optional[float] = None  # None if no VL samples contributed


def _default_depth_ids(num_layers: int) -> List[int]:
    return [1, max(1, num_layers // 2 - 1), max(0, num_layers - 4)]


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: [B,S,...]; mask: [B,S] bool/float → mean over masked positions."""
    mask = mask.to(dtype=x.dtype).reshape(mask.shape[0], mask.shape[1], *([1] * (x.ndim - 2)))
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def _effective_rank(x: torch.Tensor) -> float:
    """Participation / effective rank of rows in x [N, H]."""
    if x.numel() == 0 or x.shape[0] < 2:
        return 0.0
    x32 = x.float()
    x32 = x32 - x32.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(x32)
    except RuntimeError:
        return 0.0
    s = s.clamp_min(0.0)
    num = s.sum().square()
    den = s.square().sum().clamp_min(1e-12)
    return float((num / den).item())


def _build_loss_mask_positions(
    loss_mask: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Bool mask [B,S] for score aggregation (prefer assistant loss tokens)."""
    m = loss_mask > 0
    if not m.any():
        m = attention_mask > 0
    return m


def _next_token_ce(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked next-token CE. logits/ids: [B,S], mask on target token positions."""
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous().float()
    if shift_mask.sum() <= 0:
        shift_mask = (input_ids[:, 1:] != 0).float()  # weak fallback
    log_probs = F.log_softmax(shift_logits, dim=-1)
    nll = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (nll * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)


def _probe_logits_from_hs(model, hs: torch.Tensor) -> torch.Tensor:
    """Apply final RMSNorm + lm_head to a layer's hidden states."""
    normed = model.model.text_model.norm(hs)
    return model.lm_head(normed)


def _image_token_mask(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    return input_ids == image_token_id


def _image_attention_score(
    attn: torch.Tensor,
    query_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> Optional[float]:
    """attn: [B,H,S,S]; return mean attention mass from queries → image keys.

    Returns None if there are no image keys or no query positions.
    """
    if not image_mask.any() or not query_mask.any():
        return None
    # Average heads: [B,S,S]
    attn_mean = attn.float().mean(dim=1)
    b = attn_mean.shape[0]
    scores = []
    for i in range(b):
        q = query_mask[i]
        k = image_mask[i]
        if not q.any() or not k.any():
            continue
        # mass on image keys, averaged over query positions
        mass = attn_mean[i, q][:, k].sum(dim=-1).mean()
        scores.append(mass)
    if not scores:
        return None
    return float(torch.stack(scores).mean().item())


def _accumulate_mean(store: Dict[int, List[float]], layer_id: int, value: Optional[float]):
    if value is None:
        return
    store.setdefault(layer_id, []).append(float(value))


ALL_METRICS = (
    "ce_grad",
    "agreement",
    "delta",
    "info",
    "image_attn",
)


def _parse_metrics(spec: str) -> set:
    """Parse comma list or 'all' into metric group names."""
    spec = (spec or "all").strip().lower()
    if spec in ("all", "*"):
        return set(ALL_METRICS)
    parts = {p.strip() for p in spec.split(",") if p.strip()}
    # aliases
    aliases = {
        "agreement_kl": "agreement",
        "agreement_ce": "agreement",
        "delta_rel": "delta",
        "info_eff_rank": "info",
        "info_var": "info",
        "img_attn": "image_attn",
    }
    out = set()
    for p in parts:
        out.add(aliases.get(p, p))
    unknown = out - set(ALL_METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics {sorted(unknown)}; choose from {list(ALL_METRICS)} or all")
    return out


def analyze_batch(
    model,
    batch: Dict[str, Any],
    image_token_id: int,
    device: torch.device,
    enabled: set,
) -> Tuple[Dict[str, Dict[int, float]], bool]:
    """Run selected metrics on one collated batch (expected B=1 for grads).

    Returns (metric_name -> {layer_id -> value}, had_image).
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    pos_mask = _build_loss_mask_positions(loss_mask, attention_mask)

    forward_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        "use_cache": False,
    }
    if "pixel_values" in batch and batch["pixel_values"] is not None:
        forward_kwargs["pixel_values"] = batch["pixel_values"].to(device)
    if "pixel_attention_mask" in batch and batch["pixel_attention_mask"] is not None:
        forward_kwargs["pixel_attention_mask"] = batch["pixel_attention_mask"].to(device)

    image_mask = _image_token_mask(input_ids, image_token_id)
    has_image = bool(image_mask.any().item())
    # Foolproof: never request attentions on text-only; never score empty image masks.
    need_pass_a = bool(enabled & {"agreement", "delta", "info", "image_attn"})
    do_attn = ("image_attn" in enabled) and has_image
    if do_attn:
        forward_kwargs["output_attentions"] = True

    num_layers = model.config.text_config.num_hidden_layers
    metrics: Dict[str, Dict[int, float]] = {
        "ce_grad": {},
        "agreement_kl": {},
        "agreement_ce": {},
        "delta_rel": {},
        "info_eff_rank": {},
        "info_var": {},
        "image_attn": {},
    }

    # ---- Pass A: no-grad structural / agreement / attn metrics ----
    if need_pass_a:
        with torch.no_grad():
            out = model(**forward_kwargs)
            hs_tuple = out.hidden_states  # len = num_layers + 1
            final_probs = None
            if "agreement" in enabled:
                final_logits = out.logits.float()
                final_probs = F.log_softmax(final_logits, dim=-1).exp()

            for layer_id in range(num_layers):
                h = hs_tuple[layer_id + 1]
                h_prev = hs_tuple[layer_id]

                if "delta" in enabled:
                    delta = (h.float() - h_prev.float()).norm(dim=-1)
                    prev_n = h_prev.float().norm(dim=-1).clamp_min(1e-6)
                    rel = delta / prev_n
                    metrics["delta_rel"][layer_id] = float(_masked_mean(rel, pos_mask).item())

                if "info" in enabled:
                    flat = h.float()[pos_mask]
                    metrics["info_eff_rank"][layer_id] = _effective_rank(flat)
                    metrics["info_var"][layer_id] = (
                        float(flat.var(dim=0).mean().item()) if flat.numel() else 0.0
                    )

                if "agreement" in enabled:
                    probe = _probe_logits_from_hs(model, h).float()
                    probe_log_probs = F.log_softmax(probe, dim=-1)
                    kl = F.kl_div(
                        probe_log_probs, final_probs, reduction="none", log_target=False
                    ).sum(-1)
                    metrics["agreement_kl"][layer_id] = float(
                        _masked_mean(kl, pos_mask).item()
                    )
                    metrics["agreement_ce"][layer_id] = float(
                        _next_token_ce(probe, input_ids, loss_mask).item()
                    )

            if do_attn and out.attentions is not None:
                query_mask = pos_mask & (~image_mask)
                if not query_mask.any():
                    query_mask = (attention_mask > 0) & (~image_mask)
                for layer_id, attn in enumerate(out.attentions):
                    score = _image_attention_score(attn, query_mask, image_mask)
                    if score is not None:
                        metrics["image_attn"][layer_id] = score

    # ---- Pass B: CE activation gradients ----
    if "ce_grad" in enabled:
        # Frozen params alone yield requires_grad=False activations. Temporarily
        # enable param grads so the graph reaches each HS; we only read ∂L/∂h_ℓ.
        forward_kwargs.pop("output_attentions", None)
        model.zero_grad(set_to_none=True)
        param_flags = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(True)

        try:
            out_g = model(**forward_kwargs)
            hs_g = out_g.hidden_states
            retained = []
            for layer_id in range(num_layers):
                h = hs_g[layer_id + 1]
                h.retain_grad()
                retained.append(h)

            loss = _next_token_ce(out_g.logits, input_ids, loss_mask)
            loss.backward()

            for layer_id, h in enumerate(retained):
                if h.grad is None:
                    metrics["ce_grad"][layer_id] = 0.0
                    continue
                g = h.grad.float().norm(dim=-1)  # [B,S]
                metrics["ce_grad"][layer_id] = float(_masked_mean(g, pos_mask).item())
        finally:
            for p, flag in zip(model.parameters(), param_flags):
                p.requires_grad_(flag)
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return metrics, has_image


def _mean_maps(
    accum: Dict[str, Dict[int, List[float]]], num_layers: int
) -> Dict[int, LayerScores]:
    out: Dict[int, LayerScores] = {}
    for layer_id in range(num_layers):
        ls = LayerScores(layer_id=layer_id)
        for name in (
            "ce_grad",
            "agreement_kl",
            "agreement_ce",
            "delta_rel",
            "info_eff_rank",
            "info_var",
            "image_attn",
        ):
            vals = accum[name].get(layer_id, [])
            if not vals:
                if name == "image_attn":
                    setattr(ls, name, None)
                continue
            setattr(ls, name, sum(vals) / len(vals))
        out[layer_id] = ls
    return out


METRIC_SPECS = {
    # metric_key on LayerScores -> (enabled_group, higher_is_better, display_name)
    "ce_grad": ("ce_grad", True, "ce_grad"),
    "agreement_kl": ("agreement", False, "agreement_kl"),
    "agreement_ce": ("agreement", False, "agreement_ce"),
    "delta_rel": ("delta", True, "delta_rel"),
    "info_eff_rank": ("info", True, "info_eff_rank"),
    "info_var": ("info", True, "info_var"),
    "image_attn": ("image_attn", True, "image_attn"),
}


def _band_ranges(
    num_layers: int, exclude_first: int = 1, exclude_last: int = 1
) -> List[range]:
    lo = min(exclude_first, num_layers - 1)
    hi = max(lo + 1, num_layers - exclude_last)
    span = hi - lo
    b0 = lo + max(1, span // 3)
    b1 = lo + max(2, (2 * span) // 3)
    return [range(lo, b0), range(b0, b1), range(b1, hi)]


def _metric_value(scores: Dict[int, LayerScores], layer_id: int, metric: str) -> Optional[float]:
    v = getattr(scores[layer_id], metric)
    return None if v is None else float(v)


def _rank_layers(
    scores: Dict[int, LayerScores],
    metric: str,
    higher_is_better: bool,
    num_layers: int,
) -> List[Dict[str, Any]]:
    """Global rank list: [{layer_id, score, global_rank}, ...] best→worst."""
    rows = []
    for lid in range(num_layers):
        v = _metric_value(scores, lid, metric)
        if v is None:
            continue
        rows.append({"layer_id": lid, "score": v})
    rows.sort(key=lambda r: r["score"], reverse=higher_is_better)
    for i, r in enumerate(rows):
        r["global_rank"] = i + 1
    return rows


def _band_ranks(
    scores: Dict[int, LayerScores],
    metric: str,
    higher_is_better: bool,
    num_layers: int,
) -> Dict[str, Any]:
    """Per-band ranking + band top-1 pick of 3."""
    band_names = ["early", "mid", "late"]
    bands_out = {}
    picks = []
    for name, band in zip(band_names, _band_ranges(num_layers)):
        rows = []
        for lid in band:
            v = _metric_value(scores, lid, metric)
            if v is None:
                continue
            rows.append({"layer_id": lid, "score": v})
        rows.sort(key=lambda r: r["score"], reverse=higher_is_better)
        for i, r in enumerate(rows):
            r["band_rank"] = i + 1
        bands_out[name] = {
            "layer_range": [band.start, band.stop - 1],
            "ranking": rows,
            "top1": rows[0]["layer_id"] if rows else None,
        }
        if rows:
            picks.append(rows[0]["layer_id"])
    return {"bands": bands_out, "top3_band": picks}


def _top3_global(global_ranking: List[Dict[str, Any]]) -> List[int]:
    return [r["layer_id"] for r in global_ranking[:3]]


def _random_triple(num_layers: int, seed: int, exclude_last: int = 1) -> List[int]:
    import random

    rng = random.Random(seed)
    pool = list(range(0, max(1, num_layers - exclude_last)))
    if len(pool) < 3:
        return pool
    return sorted(rng.sample(pool, 3))


def _set_vs_final_metrics(
    features: List[torch.Tensor], target: torch.Tensor, pos_mask: torch.Tensor
) -> Dict[str, float]:
    """How close a layer set is to final HS (no learned map).

    - mean_cos / max_cos: cosine of each selected layer to final, then mean/max
    - mean_mse_rel: mean relative MSE of each selected layer to final
    """
    if not features or not pos_mask.any():
        return {"mean_cos": 0.0, "max_cos": 0.0, "mean_mse_rel": 1.0}
    y = target.float()[pos_mask]
    cos_list, mse_list = [], []
    denom = y.pow(2).mean().clamp_min(1e-6)
    for f in features:
        x = f.float()[pos_mask]
        cos_list.append(F.cosine_similarity(x, y, dim=-1).mean())
        mse_list.append((x - y).pow(2).mean() / denom)
    cos_t = torch.stack(cos_list)
    mse_t = torch.stack(mse_list)
    return {
        "mean_cos": float(cos_t.mean().item()),
        "max_cos": float(cos_t.max().item()),
        "mean_mse_rel": float(mse_t.mean().item()),
    }


@torch.no_grad()
def evaluate_layer_sets(
    model,
    loader: DataLoader,
    device: torch.device,
    image_token_id: int,
    candidate_sets: Dict[str, List[int]],
    num_layers: int,
    max_eval_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Score sets by closeness to final HS; full = all layers except last."""
    last_id = num_layers - 1
    full_ids = list(range(last_id))

    def _sanitize(ids: List[int]) -> List[int]:
        out = [i for i in ids if 0 <= i < last_id]
        return out if out else full_ids[:3]

    sanitized = {k: _sanitize(v) for k, v in candidate_sets.items()}
    all_names = list(sanitized.keys()) + ["full_layers"]
    sums = {
        n: {"mean_cos": 0.0, "max_cos": 0.0, "mean_mse_rel": 0.0, "n": 0}
        for n in all_names
    }

    for step, batch in enumerate(loader):
        if max_eval_batches is not None and step >= max_eval_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        pos_mask = _build_loss_mask_positions(loss_mask, attention_mask)
        fwd = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }
        if "pixel_values" in batch and batch["pixel_values"] is not None:
            fwd["pixel_values"] = batch["pixel_values"].to(device)
        if "pixel_attention_mask" in batch and batch["pixel_attention_mask"] is not None:
            fwd["pixel_attention_mask"] = batch["pixel_attention_mask"].to(device)

        out = model(**fwd)
        hs = out.hidden_states
        h_final = hs[-1]

        def feats(ids: List[int]) -> List[torch.Tensor]:
            return [hs[i + 1] for i in ids]

        for name, ids in sanitized.items():
            m = _set_vs_final_metrics(feats(ids), h_final, pos_mask)
            for k in ("mean_cos", "max_cos", "mean_mse_rel"):
                sums[name][k] += m[k]
            sums[name]["n"] += 1

        m_full = _set_vs_final_metrics(feats(full_ids), h_final, pos_mask)
        for k in ("mean_cos", "max_cos", "mean_mse_rel"):
            sums["full_layers"][k] += m_full[k]
        sums["full_layers"]["n"] += 1

        if (step + 1) % 5 == 0:
            print(f"  eval batch {step + 1}")

    results = {}
    full = sums["full_layers"]
    n_full = max(full["n"], 1)
    full_cos = full["mean_cos"] / n_full
    full_mse = full["mean_mse_rel"] / n_full

    for name, s in sums.items():
        n = max(s["n"], 1)
        mean_cos = s["mean_cos"] / n
        max_cos = s["max_cos"] / n
        mean_mse = s["mean_mse_rel"] / n
        results[name] = {
            "layers_requested": candidate_sets.get(name, full_ids),
            "layers_eval": sanitized.get(name, full_ids),
            "mean_cos": mean_cos,
            "max_cos": max_cos,
            "mean_mse_rel": mean_mse,
            "full_mean_cos": full_cos,
            "full_mean_mse_rel": full_mse,
            "gap_cos": full_cos - mean_cos,
            "gap_mse": mean_mse - full_mse,
            # Higher better: closeness to final, relative to full-set mean cos.
            "final_score": mean_cos / max(full_cos, 1e-6),
        }
    return results


def _build_metric_reports(
    scores: Dict[int, LayerScores],
    enabled: set,
    num_layers: int,
    n_image: int,
) -> Dict[str, Any]:
    reports = {}
    for metric, (group, higher, _) in METRIC_SPECS.items():
        if group not in enabled:
            continue
        if metric == "image_attn" and n_image == 0:
            reports[metric] = {
                "status": "skipped",
                "reason": "no image samples",
                "higher_is_better": higher,
            }
            continue
        # Skip if all null/zero unused
        if all(_metric_value(scores, i, metric) is None for i in range(num_layers)):
            continue
        glob = _rank_layers(scores, metric, higher, num_layers)
        band = _band_ranks(scores, metric, higher, num_layers)
        reports[metric] = {
            "higher_is_better": higher,
            "global_ranking": glob,
            "top3_global": _top3_global(glob),
            "band_ranking": band["bands"],
            "top3_band": band["top3_band"],
            "summary": {
                "best_global": glob[0] if glob else None,
                "worst_global": glob[-1] if glob else None,
                "top3_global": _top3_global(glob),
                "top3_band": band["top3_band"],
            },
        }
    return reports


def _print_table(scores: Dict[int, LayerScores], n_image: int, n_text: int):
    print(
        f"\nSamples: {n_image + n_text}  (image={n_image}, text_only={n_text})"
    )
    hdr = (
        f"{'L':>3}  {'ce_grad':>10}  {'agree_kl':>10}  {'agree_ce':>10}  "
        f"{'delta_rel':>10}  {'eff_rank':>9}  {'var':>10}  {'img_attn':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for lid in sorted(scores):
        s = scores[lid]
        img = f"{s.image_attn:.4f}" if s.image_attn is not None else "N/A"
        print(
            f"{lid:3d}  {s.ce_grad:10.4g}  {s.agreement_kl:10.4g}  {s.agreement_ce:10.4g}  "
            f"{s.delta_rel:10.4g}  {s.info_eff_rank:9.3f}  {s.info_var:10.4g}  {img:>10}"
        )


def _print_metric_summaries(reports: Dict[str, Any]):
    print("\n=== Per-metric summary (global top3 | band top3) ===")
    for metric, rep in reports.items():
        if rep.get("status") == "skipped":
            print(f"  {metric}: skipped ({rep.get('reason')})")
            continue
        s = rep["summary"]
        print(
            f"  {metric}: global={s['top3_global']}  band={s['top3_band']}  "
            f"(higher_is_better={rep['higher_is_better']})"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Target-only SmolVLM layer importance analysis")
    p.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "HuggingFaceTB/SmolVLM-256M-Instruct"),
    )
    p.add_argument(
        "--data-path",
        default=os.environ.get(
            "DATA_PATH",
            str(_REPO_ROOT / "dataset/smolvlm_256m_target_gen/data_0-36.jsonl"),
        ),
    )
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
    )
    p.add_argument(
        "--metrics",
        default="all",
        help="Comma list: ce_grad,agreement,delta,info,image_attn  (or 'all')",
    )
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--no-image-attn",
        action="store_true",
        help="Deprecated: omit image_attn from --metrics instead",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval-random",
        action="store_true",
        default=os.environ.get("EVAL_RANDOM", "").lower() in ("1", "true", "yes"),
        help="Also evaluate a random 3-layer set",
    )
    p.add_argument("--num-random", type=int, default=1, help="How many random triples")
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip selected-vs-full reconstruction eval",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    enabled = _parse_metrics(args.metrics)
    if args.no_image_attn:
        enabled.discard("image_attn")

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    attn_impl = "eager" if "image_attn" in enabled else "sdpa"
    print(
        f"Loading processor/model from {args.model_path} → {device}, "
        f"dtype={args.dtype}, attn={attn_impl}, metrics={sorted(enabled)}"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    image_token_id = int(getattr(model.config, "image_token_id", 49190))
    num_layers = int(model.config.text_config.num_hidden_layers)
    depth_ids = _default_depth_ids(num_layers)

    builder = OnlineSmolVLMDatasetBuilder(
        tokenizer=processor,
        max_length=args.max_length,
        shuffle_seed=args.seed,
        chat_template_type=ChatTemplateType.SMOLVLM,
        display=False,
    )
    ds = builder.build_dataset(
        args.data_path,
        num_proc=args.num_proc,
        shuffle=False,
        sample_num=args.max_samples,
        load_from_cache_file=False,
    )
    collator = builder.get_data_collator()
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator)

    accum: Dict[str, Dict[int, List[float]]] = {
        "ce_grad": {},
        "agreement_kl": {},
        "agreement_ce": {},
        "delta_rel": {},
        "info_eff_rank": {},
        "info_var": {},
        "image_attn": {},
    }
    n_image = 0
    n_text = 0

    print(f"Analyzing {len(ds)} samples from {args.data_path}")
    for step, batch in enumerate(loader):
        metrics, has_image = analyze_batch(
            model,
            batch,
            image_token_id=image_token_id,
            device=device,
            enabled=enabled,
        )
        if has_image:
            n_image += 1
        else:
            n_text += 1

        for name, layer_map in metrics.items():
            for lid, val in layer_map.items():
                _accumulate_mean(accum[name], lid, val)

        if (step + 1) % 5 == 0 or (step + 1) == len(ds):
            print(f"  processed {step + 1}/{len(ds)} (image={n_image}, text_only={n_text})")

    scores = _mean_maps(accum, num_layers)
    _print_table(scores, n_image, n_text)

    # --- 1) per-metric global + band ranks ---
    metric_reports = _build_metric_reports(scores, enabled, num_layers, n_image)
    _print_metric_summaries(metric_reports)

    # --- 2/3) candidate sets: band top3 + global top3 (+ random) ---
    candidate_sets: Dict[str, List[int]] = {
        "depth_baseline": depth_ids,
    }
    for metric, rep in metric_reports.items():
        if rep.get("status") == "skipped":
            continue
        candidate_sets[f"{metric}__band"] = list(rep["top3_band"])
        candidate_sets[f"{metric}__global"] = list(rep["top3_global"])

    if args.eval_random:
        for r in range(args.num_random):
            candidate_sets[f"random_{r}"] = _random_triple(
                num_layers, seed=args.seed + r
            )

    evaluation = {}
    comparison_rows = []
    if not args.skip_eval:
        print("\n=== Eval: selected vs full (closeness to final HS) ===")
        evaluation = evaluate_layer_sets(
            model,
            loader,
            device=device,
            image_token_id=image_token_id,
            candidate_sets=candidate_sets,
            num_layers=num_layers,
        )
        # Rank methods by final_score (excl. full_layers reference)
        ranked = sorted(
            ((k, v) for k, v in evaluation.items() if k != "full_layers"),
            key=lambda kv: kv[1]["final_score"],
            reverse=True,
        )
        print(
            f"{'method':<28} {'layers_eval':<16} {'mean_cos':>8} {'mse':>8} "
            f"{'gap_cos':>8} {'score':>7}"
        )
        print("-" * 86)
        for name, ev in ranked:
            layers = ev.get("layers_eval", ev.get("layers"))
            comparison_rows.append(
                {
                    "method": name,
                    "layers": layers,
                    "layers_requested": ev.get("layers_requested", layers),
                    "mean_cos": ev["mean_cos"],
                    "mean_mse_rel": ev["mean_mse_rel"],
                    "gap_cos": ev["gap_cos"],
                    "gap_mse": ev["gap_mse"],
                    "final_score": ev["final_score"],
                }
            )
            print(
                f"{name:<28} {str(layers):<16} "
                f"{ev['mean_cos']:8.4f} {ev['mean_mse_rel']:8.4g} "
                f"{ev['gap_cos']:8.4f} {ev['final_score']:7.4f}"
            )
        if ranked:
            print(
                f"\nBest method: {ranked[0][0]}  "
                f"layers={ranked[0][1].get('layers_eval', ranked[0][1].get('layers'))}"
            )
    tag = "all" if enabled == set(ALL_METRICS) else "+".join(sorted(enabled))
    summary = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "metrics": sorted(enabled),
        "num_layers": num_layers,
        "n_samples": n_image + n_text,
        "n_image": n_image,
        "n_text_only": n_text,
        "depth_only_baseline": depth_ids,
        "metric_reports": metric_reports,
        "candidate_sets": candidate_sets,
        "evaluation": evaluation,
        "final_comparison": comparison_rows,
        "note": (
            "Per metric: global_ranking + band_ranking (early/mid/late). "
            "Eval: mean cosine / MSE of selected layers vs final-layer HS, "
            "compared to full_layers (all but last). "
            "final_score = mean_cos / full_mean_cos. "
            "Last layer stripped from eval features if present in a pick. "
            "Layer ids = AngelSlim aux_hidden_states_layer_ids (HF hs[id+1])."
        ),
        "layers": [asdict(scores[i]) for i in range(num_layers)],
    }

    json_path = os.path.join(args.output_dir, f"layer_importance_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Compact metric summaries
    metrics_path = os.path.join(args.output_dir, f"metric_summaries_{tag}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {m: r.get("summary", r) for m, r in metric_reports.items()},
            f,
            indent=2,
        )

    # Final comparison CSV
    if comparison_rows:
        cmp_path = os.path.join(args.output_dir, f"final_comparison_{tag}.csv")
        with open(cmp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "layers",
                    "layers_requested",
                    "mean_cos",
                    "mean_mse_rel",
                    "gap_cos",
                    "gap_mse",
                    "final_score",
                ],
            )
            w.writeheader()
            for row in comparison_rows:
                out = dict(row)
                out["layers"] = json.dumps(out["layers"])
                out["layers_requested"] = json.dumps(out.get("layers_requested", []))
                w.writerow(out)
        print(f"Wrote {cmp_path}")

    csv_path = os.path.join(args.output_dir, f"layer_importance_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "layer_id",
                "ce_grad",
                "agreement_kl",
                "agreement_ce",
                "delta_rel",
                "info_eff_rank",
                "info_var",
                "image_attn",
            ],
        )
        w.writeheader()
        for i in range(num_layers):
            w.writerow(asdict(scores[i]))

    print(f"\nWrote {json_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
