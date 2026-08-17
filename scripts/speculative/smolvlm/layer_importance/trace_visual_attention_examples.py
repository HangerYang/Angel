#!/usr/bin/env python3
"""Trace per-token visual attention for a few SmolVLM image examples.

Stores full head-averaged text-query -> image-key attention vectors for every
assistant/loss token and every layer, for target and configured drafts.

Output:
  visual_attention_traces.pt    torch payload with full tensors
  visual_attention_traces.json  metadata + top-k summaries
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
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


def _parse_runs(spec: Optional[str]) -> List[RunSpec]:
    if not spec:
        return _default_runs()
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).name
        out.append(RunSpec(name.strip(), Path(path).expanduser().resolve()))
    return out


def _load_state_dict(ckpt: Path) -> Dict[str, torch.Tensor]:
    safes = sorted(ckpt.glob("*.safetensors"))
    if safes:
        from safetensors.torch import load_file

        state: Dict[str, torch.Tensor] = {}
        for path in safes:
            state.update(load_file(str(path)))
        return state
    bin_path = ckpt / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No weights under {ckpt}")


def _load_draft(ckpt: Path, device: torch.device, dtype: torch.dtype):
    cfg = DraftModelConfig.from_file(ckpt / "config.json")
    model = create_draft_model(cfg)
    missing, unexpected = model.load_state_dict(_load_state_dict(ckpt), strict=False)
    missing = [k for k in missing if not (k.startswith("t2d") or k.startswith("d2t"))]
    if missing:
        print(f"WARNING {ckpt}: missing={len(missing)} e.g. {missing[:5]}")
    if unexpected:
        print(f"WARNING {ckpt}: unexpected={len(unexpected)} e.g. {unexpected[:5]}")
    model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, cfg


def _query_positions(loss_mask: torch.Tensor, attention_mask: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
    q = (loss_mask > 0) & (attention_mask > 0) & (~image_mask)
    if not q.any():
        q = (attention_mask > 0) & (~image_mask)
    return q


def _decode_tokens(processor, input_ids: torch.Tensor, positions: List[int]) -> List[str]:
    toks = []
    for pos in positions:
        tid = int(input_ids[pos].item())
        try:
            tok = processor.tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            tok = str(tid)
        toks.append(tok)
    return toks


def _tokens_to_text(processor, input_ids: torch.Tensor) -> str:
    try:
        return processor.tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
    except Exception:
        return ""


def _visual_vectors(attentions, query_pos: List[int], image_pos: List[int]) -> torch.Tensor:
    """Return [layers, query_tokens, image_tokens], head-averaged."""
    rows = []
    for attn in attentions:
        # [B,H,S,S] -> [S,S], then select query rows and image-key cols.
        a = attn[0].float().mean(dim=0)
        rows.append(a[query_pos][:, image_pos].detach().cpu())
    return torch.stack(rows, dim=0)


def _topk_summary(vectors: torch.Tensor, query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
    """Summarize [layers,T,V] with top visual-index per token/layer."""
    out = []
    layers, t_count, v_count = vectors.shape
    k = min(top_k, v_count)
    for t in range(t_count):
        token_entry = {"query_index": t, "token": query_tokens[t], "layers": []}
        for layer in range(layers):
            vals, idx = torch.topk(vectors[layer, t], k=k)
            token_entry["layers"].append(
                {
                    "layer_id": layer,
                    "visual_mass": float(vectors[layer, t].sum().item()),
                    "top_visual_indices": [int(i) for i in idx.tolist()],
                    "top_visual_scores": [float(v) for v in vals.tolist()],
                }
            )
        out.append(token_entry)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    p.add_argument(
        "--data-path",
        default=str(_REPO_ROOT / "dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl"),
    )
    p.add_argument("--runs", default=None)
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "visual_attention_traces_5"),
    )
    p.add_argument("--num-examples", type=int, default=5)
    p.add_argument("--sample-pool-size", type=int, default=200)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Loading target {args.model_path} on {device}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    target = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
        attn_implementation="eager",
    )
    target.to(device)
    target.eval()
    for param in target.parameters():
        param.requires_grad_(False)

    drafts = []
    for spec in _parse_runs(args.runs):
        print(f"Loading draft {spec.name}: {spec.checkpoint}")
        draft, cfg = _load_draft(spec.checkpoint, device, dtype)
        drafts.append((spec, draft, cfg))

    image_token_id = int(getattr(target.config, "image_token_id", 49190))
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
        shuffle=True,
        sample_num=args.sample_pool_size,
        load_from_cache_file=False,
    )
    ds = ds.filter(
        lambda batch: [bool(p) and p != "[]" for p in batch["image_paths"]],
        batched=True,
        num_proc=args.num_proc,
        load_from_cache_file=False,
        desc="Filtering image-only trace samples",
    )
    if len(ds) < args.num_examples:
        raise RuntimeError(f"Only found {len(ds)} image examples in pool of {args.sample_pool_size}")
    ds = ds.select(range(args.num_examples))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=builder.get_data_collator())

    pt_payload: Dict[str, Any] = {"examples": []}
    json_meta: Dict[str, Any] = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "num_examples": args.num_examples,
        "sample_pool_size": args.sample_pool_size,
        "image_token_id": image_token_id,
        "top_k": args.top_k,
        "runs": [spec.name for spec, _, _ in drafts],
        "note": (
            "Full tensors are in visual_attention_traces.pt. For each model, "
            "visual_attention has shape [num_layers, num_query_tokens, num_image_tokens], "
            "head-averaged, where query tokens are assistant/loss tokens. visual_index is "
            "the order within image_token_positions. For Eagle/progressive 2H layers this is "
            "post-softmax attention from the single concatenated attention module, not a "
            "left/right stream decomposition."
        ),
        "examples": [],
    }

    for ex_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        image_mask = input_ids == image_token_id
        query_mask = _query_positions(loss_mask, attention_mask, image_mask)
        query_pos = torch.nonzero(query_mask[0], as_tuple=False).flatten().tolist()
        image_pos = torch.nonzero(image_mask[0], as_tuple=False).flatten().tolist()
        if not image_pos:
            continue
        query_tokens = _decode_tokens(processor, input_ids[0].detach().cpu(), query_pos)
        image_paths = json.loads(batch.get("image_paths", ["[]"])[0]) if "image_paths" in batch else []

        fwd: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_attentions": True,
            "output_hidden_states": True,
            "use_cache": False,
        }
        if "pixel_values" in batch and batch["pixel_values"] is not None:
            fwd["pixel_values"] = batch["pixel_values"].to(device)
        if "pixel_attention_mask" in batch and batch["pixel_attention_mask"] is not None:
            fwd["pixel_attention_mask"] = batch["pixel_attention_mask"].to(device)

        with torch.no_grad():
            target_out = target(**fwd)
        target_vectors = _visual_vectors(target_out.attentions, query_pos, image_pos)

        pt_example: Dict[str, Any] = {
            "example_index": ex_idx,
            "input_ids": input_ids[0].detach().cpu(),
            "attention_mask": attention_mask[0].detach().cpu(),
            "loss_mask": loss_mask[0].detach().cpu(),
            "query_positions": torch.tensor(query_pos, dtype=torch.long),
            "image_token_positions": torch.tensor(image_pos, dtype=torch.long),
            "query_tokens": query_tokens,
            "image_paths": image_paths,
            "target": {"visual_attention": target_vectors},
            "drafts": {},
        }
        meta_example: Dict[str, Any] = {
            "example_index": ex_idx,
            "sequence_length": int(input_ids.shape[1]),
            "num_query_tokens": len(query_pos),
            "num_image_tokens": len(image_pos),
            "query_positions": query_pos,
            "query_tokens": query_tokens,
            "image_token_positions": image_pos,
            "image_paths": image_paths,
            "decoded_text": _tokens_to_text(processor, input_ids[0].detach().cpu()),
            "target_topk": _topk_summary(target_vectors, query_tokens, args.top_k),
            "drafts": {},
        }

        aux_cache: Dict[Tuple[int, ...], torch.Tensor] = {}
        for spec, draft, cfg in drafts:
            aux_ids = tuple(int(x) for x in getattr(cfg, "aux_hidden_states_layer_ids", [1, 14, 26]))
            if aux_ids not in aux_cache:
                aux_cache[aux_ids] = torch.cat([target_out.hidden_states[i + 1] for i in aux_ids], dim=-1)
            with torch.no_grad():
                d_out = draft(
                    hidden_states=aux_cache[aux_ids].to(dtype=next(draft.parameters()).dtype),
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=True,
                    output_hidden_states=True,
                )
            draft_vectors = _visual_vectors(d_out["attentions"], query_pos, image_pos)
            pt_example["drafts"][spec.name] = {
                "visual_attention": draft_vectors,
                "mode": getattr(cfg, "eagle_aux_injection_mode", "fused_fc"),
                "aux_hidden_states_layer_ids": list(aux_ids),
            }
            meta_example["drafts"][spec.name] = {
                "mode": getattr(cfg, "eagle_aux_injection_mode", "fused_fc"),
                "num_layers": int(draft_vectors.shape[0]),
                "aux_hidden_states_layer_ids": list(aux_ids),
                "topk": _topk_summary(draft_vectors, query_tokens, args.top_k),
            }

        pt_payload["examples"].append(pt_example)
        json_meta["examples"].append(meta_example)
        print(
            f"traced example {ex_idx + 1}/{args.num_examples}: "
            f"queries={len(query_pos)} image_tokens={len(image_pos)}"
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pt_path = out_dir / "visual_attention_traces.pt"
    json_path = out_dir / "visual_attention_traces.json"
    torch.save(pt_payload, pt_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_meta, f, indent=2)
    print(f"Wrote {pt_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
