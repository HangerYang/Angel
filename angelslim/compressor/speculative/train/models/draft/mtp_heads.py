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

"""Medusa-style multi-token-prediction (MTP) heads, trained on top of a frozen,
already-trained EAGLE3 draft trunk.

Standalone experiment: given the trunk's final hidden state ``h_final`` for a
single draft forward pass, each extra head predicts a token further ahead
(t+2, t+3, ...) with a single ``Linear(H,H)`` + residual, reusing the trunk's
own ``norm``/``lm_head`` (so the only new trainable params are the small heads
themselves). See lora_utils.apply_real_hawk_training_setup for the freeze
pattern this mirrors.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .lora_utils import freeze_all_parameters

__all__ = ["MTPHeads", "apply_mtp_training_setup"]


class MTPHeads(nn.Module):
    """``K-1`` independent extra heads predicting t+2, t+3, ... from ``h_final``.

    Heads are independent of each other (head[1] does not consume head[0]'s
    output) and all read the same ``h_final`` -- one extra small batched matmul
    per head, not a sequential re-encode.
    """

    def __init__(self, hidden_size: int, num_extra_heads: int):
        super().__init__()
        if num_extra_heads < 1:
            raise ValueError(f"num_extra_heads must be >= 1, got {num_extra_heads}")
        self.hidden_size = hidden_size
        self.num_extra_heads = num_extra_heads
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_extra_heads)]
        )

    def forward(
        self,
        h_final: torch.Tensor,
        norm: nn.Module,
        lm_head: nn.Module,
    ) -> List[torch.Tensor]:
        """Returns ``num_extra_heads`` logits tensors, each ``[B, S, draft_vocab_size]``.

        ``logits[k]`` predicts the token at offset ``t + 2 + k`` (k=0 -> t+2).
        ``norm``/``lm_head`` are the frozen trunk's own modules (shared, not
        copied) -- callers are expected to have frozen them already.
        """
        logits = []
        for head in self.heads:
            fused = norm(head(h_final) + h_final)
            logits.append(lm_head(fused).float())
        return logits


def apply_mtp_training_setup(draft: nn.Module, mtp_heads: MTPHeads) -> dict:
    """Freeze the entire draft trunk, leave ``mtp_heads`` trainable.

    Mirrors ``lora_utils.apply_real_hawk_training_setup``'s freeze/unfreeze
    contract and return shape, for a "freeze trunk, train new heads only"
    experiment instead of a LoRA one.
    """
    freeze_all_parameters(draft)
    trainable_names: List[str] = []
    for name, p in mtp_heads.named_parameters():
        p.requires_grad_(True)
        trainable_names.append(f"mtp_heads.{name}")

    n_train = sum(p.numel() for p in mtp_heads.parameters())
    n_total = n_train + sum(p.numel() for p in draft.parameters())
    return {
        "trainable_names": trainable_names,
        "num_trainable": n_train,
        "num_total": n_total,
    }
