# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal LoRA wrappers for real_hawk / layer-skip draft training (no peft)."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep in sync with llama_eagle3.REAL_HAWK_MODES
REAL_HAWK_MODES = frozenset({"real_hawk", "layer_skip_lora"})


class LoRALinear(nn.Module):
    """Frozen base ``nn.Linear`` + trainable low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {r}")
        if base.bias is not None:
            raise ValueError("LoRALinear expects bias-free Linear (SmolVLM draft)")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(self.r, in_f, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.r, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        # Base stays frozen; only A/B train.
        self.base.weight.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base.weight, self.base.bias)
        # When disabled (verify path on shared modules), return base only.
        if not getattr(self, "enabled", True):
            return base_out
        lora_x = self.lora_dropout(x)
        # x @ A^T @ B^T * scale
        lora_out = (lora_x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def merge_weight(self) -> torch.Tensor:
        """Return W + (B @ A) * scale (for export)."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return self.base.weight + delta.to(dtype=self.base.weight.dtype)


_DEFAULT_LORA_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _match_lora_target(name: str, target_modules: Sequence[str]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return leaf in target_modules or name in target_modules


def inject_lora(
    module: nn.Module,
    *,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    target_modules: Sequence[str] | None = None,
) -> List[str]:
    """Replace matching ``nn.Linear`` children with ``LoRALinear``.

    Returns list of replaced module names (dotted, relative to ``module``).
    """
    targets = tuple(target_modules) if target_modules else _DEFAULT_LORA_SUFFIXES
    replaced: List[str] = []

    def _recurse(root: nn.Module, prefix: str) -> None:
        for child_name, child in list(root.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and _match_lora_target(full, targets):
                setattr(
                    root,
                    child_name,
                    LoRALinear(child, r=r, alpha=alpha, dropout=dropout),
                )
                replaced.append(full)
            elif isinstance(child, LoRALinear):
                continue
            else:
                _recurse(child, full)

    _recurse(module, "")
    return replaced


def freeze_all_parameters(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_parameters(params: Iterable[nn.Parameter]) -> None:
    for p in params:
        p.requires_grad_(True)


def apply_real_hawk_training_setup(
    draft: nn.Module,
    *,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    target_modules: Sequence[str] | None = None,
) -> dict:
    """Freeze draft body, inject LoRA into draft layers, unfreeze LoRA + fuse + head.

    Expects a hawk-shaped draft: ``layers``, ``fuse_w1``, ``fuse_w2``, ``lm_head``,
    ``norm``, frozen ``embed_tokens``. Used by ``real_hawk`` (and alias
    ``layer_skip_lora``).
    """
    freeze_all_parameters(draft)
    # Inject only under draft.layers.* (not lm_head / fuse).
    replaced = inject_lora(
        getattr(draft, "layers", draft),
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )
    # Re-qualify names for logging.
    replaced = [f"layers.{n}" for n in replaced]
    # Trainable: LoRA A/B, hawk fuse, final norm + lm_head.
    trainable_names: List[str] = []
    for name, p in draft.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_fuse = name.startswith("fuse_w1") or name.startswith("fuse_w2")
        is_head = name.startswith("lm_head") or name.startswith("norm")
        if is_lora or is_fuse or is_head:
            p.requires_grad_(True)
            trainable_names.append(name)
        else:
            p.requires_grad_(False)

    n_train = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in draft.parameters())
    return {
        "lora_modules": replaced,
        "trainable_names": trainable_names,
        "num_trainable": n_train,
        "num_total": n_total,
    }


# Back-compat alias
apply_layer_skip_lora_training_setup = apply_real_hawk_training_setup


def merge_lora_into_state_dict(draft: nn.Module) -> dict:
    """Export hawk-compatible state_dict with LoRA merged into Linear weights."""
    # Temporarily swap LoRALinear → merged Linear for state_dict collection.
    swaps: list[tuple[nn.Module, str, nn.Module, nn.Module]] = []

    def _collect(root: nn.Module, prefix: str) -> None:
        for child_name, child in list(root.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                merged = nn.Linear(
                    child.base.in_features,
                    child.base.out_features,
                    bias=False,
                    device=child.base.weight.device,
                    dtype=child.base.weight.dtype,
                )
                with torch.no_grad():
                    merged.weight.copy_(child.merge_weight())
                swaps.append((root, child_name, child, merged))
                setattr(root, child_name, merged)
            else:
                _collect(child, full)

    _collect(draft, "")
    try:
        return {k: v.detach().cpu().clone() for k, v in draft.state_dict().items()}
    finally:
        for root, name, original, _merged in swaps:
            setattr(root, name, original)
