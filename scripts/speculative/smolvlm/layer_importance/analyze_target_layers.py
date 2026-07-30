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
) -> Dict[str, LayerScores]:
    out: Dict[str, LayerScores] = {}
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


def _banded_pick(
    scores: Dict[int, LayerScores],
    num_layers: int,
    primary: str = "ce_grad",
    higher_is_better: bool = True,
    exclude_first: int = 1,
    exclude_last: int = 1,
) -> List[int]:
    """Pick one layer from early / mid / late bands by primary score."""
    lo = min(exclude_first, num_layers - 1)
    hi = max(lo + 1, num_layers - exclude_last)
    span = hi - lo
    b0 = lo + max(1, span // 3)
    b1 = lo + max(2, (2 * span) // 3)
    bands = [
        range(lo, b0),
        range(b0, b1),
        range(b1, hi),
    ]
    picks = []
    for band in bands:
        cand = [i for i in band if i in scores]
        if not cand:
            continue

        def key(i, _p=primary):
            v = getattr(scores[i], _p)
            if v is None:
                return float("-inf") if higher_is_better else float("inf")
            return v

        best = max(cand, key=key) if higher_is_better else min(cand, key=key)
        picks.append(best)
    return picks


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

    # eager required only when collecting attentions
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

    banded: Dict[str, Any] = {}
    if "ce_grad" in enabled:
        banded["ce_grad"] = _banded_pick(
            scores, num_layers, primary="ce_grad", higher_is_better=True
        )
    if "agreement" in enabled:
        banded["agreement_kl"] = _banded_pick(
            scores, num_layers, primary="agreement_kl", higher_is_better=False
        )
    if "delta" in enabled:
        banded["delta_rel"] = _banded_pick(
            scores, num_layers, primary="delta_rel", higher_is_better=True
        )
    if "info" in enabled:
        banded["info_eff_rank"] = _banded_pick(
            scores, num_layers, primary="info_eff_rank", higher_is_better=True
        )
    if "image_attn" in enabled:
        banded["image_attn"] = (
            _banded_pick(scores, num_layers, primary="image_attn", higher_is_better=True)
            if n_image > 0
            else None
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
        "banded_selection": banded,
        "suggested_aux_hidden_states_layer_ids": banded.get("ce_grad"),
        "note": (
            "Layer ids match AngelSlim aux_hidden_states_layer_ids "
            "(HF hidden_states index = id + 1). "
            "image_attn is N/A when n_image=0."
        ),
        "layers": [asdict(scores[i]) for i in range(num_layers)],
    }

    json_path = os.path.join(args.output_dir, f"layer_importance_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # Also write stable names for the all-metrics run
    if enabled == set(ALL_METRICS):
        with open(os.path.join(args.output_dir, "layer_importance.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

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
    if enabled == set(ALL_METRICS):
        with open(
            os.path.join(args.output_dir, "layer_importance.csv"), "w", newline="", encoding="utf-8"
        ) as f:
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

    print("\nDepth-only baseline:     ", depth_ids)
    for k, v in banded.items():
        print(f"Banded pick ({k}): {v}")
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
