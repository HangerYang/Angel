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

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import Trainer

from angelslim.utils.lazy_imports import deepspeed

from ...utils import padding

logger = logging.getLogger(__name__)


class Eagle3Trainer(Trainer, ABC):
    """
    EAGLE3 Trainer for speculative decoding training.

    Implements training logic for EAGLE3 model using a draft model to predict
    tokens based on hidden states from a target model.
    """

    def __init__(self, draft_model: nn.Module, length: int, **kwargs):
        """
        Initialize the OnlineEagle3Trainer.

        Args:
            draft_model: Draft model for token prediction
            length: Number of speculative decoding steps
            **kwargs: Additional arguments passed to parent Trainer
        """
        # GT target-HS warmup steps (alias: progressive_target_hs_warmup_steps).
        warmup = kwargs.pop("target_hs_warmup_steps", None)
        if warmup is None:
            warmup = kwargs.pop("progressive_target_hs_warmup_steps", 0)
        else:
            kwargs.pop("progressive_target_hs_warmup_steps", None)
        self.target_hs_warmup_steps = int(warmup or 0)
        # Back-compat alias used by older call sites / logs.
        self.progressive_target_hs_warmup_steps = self.target_hs_warmup_steps
        draft_model_config = kwargs.pop("draft_model_config", None)
        self.skew_kl_loss_weight = float(
            getattr(draft_model_config, "skew_kl_loss_weight", 0.0) or 0.0
        )
        self.skew_kl_alpha = float(
            getattr(draft_model_config, "skew_kl_alpha", 0.1) or 0.1
        )
        self.skew_kl_direction = str(
            getattr(draft_model_config, "skew_kl_direction", "reverse") or "reverse"
        )
        self.skew_kl_loss_mode = str(
            getattr(draft_model_config, "skew_kl_loss_mode", "additive")
            or "additive"
        )
        self.skew_kl_stage_weights = list(
            getattr(draft_model_config, "skew_kl_stage_weights", []) or []
        )
        super().__init__(model=draft_model, **kwargs)
        self.length = length
        self._train_start_time = None
        self._train_pending_log: dict = {}
        self._train_pending_log_count: int = 0
        self._eval_pending_log: dict = {}
        self._eval_pending_log_count: int = 0
        self._logged_draft_hs_feedback = False
        self._logged_target_hs_warmup = False
        self._logged_aux_losses = False
        # Set by create_optimizer() for non-DeepSpeed bf16 DDP (ZeRO-equivalent).
        self._fp32_optimizer = None

    @staticmethod
    def _stage_weight(weights: List[float], idx: int, default: float = 1.0) -> float:
        if not weights:
            return default
        if idx < len(weights):
            return float(weights[idx])
        return float(weights[-1])

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=values.dtype, device=values.device)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def create_optimizer(self, model=None):
        """Use FP32 master Adam under plain DDP to match DeepSpeed ZeRO moments.

        DeepSpeed ZeRO keeps Adam state in FP32 even when params are bf16.
        HF AdamW + bf16 DDP keeps moments in bf16 and plateaus. Mirror ZeRO
        with FP32MasterWeightOptimizer when DeepSpeed is off.
        """
        if self.is_deepspeed_enabled:
            return super().create_optimizer()

        from torch.optim import AdamW

        from .fp32_master_optimizer import FP32MasterWeightOptimizer, FP32StateAdamW

        if self.is_fsdp_enabled:
            args = self.args
            param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad]}]
            optimizer = FP32StateAdamW(
                param_groups,
                lr=args.learning_rate,
                betas=(
                    getattr(args, "adam_beta1", 0.9),
                    getattr(args, "adam_beta2", 0.999),
                ),
                eps=getattr(args, "adam_epsilon", 1e-8),
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
            )
            self.optimizer = optimizer
            logger.info(
                "Eagle3: using FP32StateAdamW (FSDP) so Adam moments match ZeRO FP32."
            )
            return self.optimizer

        bf16_params = [p for p in self.model.parameters() if p.requires_grad]
        if not bf16_params:
            return super().create_optimizer()

        args = self.args
        inner_optimizer = AdamW(
            bf16_params,
            lr=args.learning_rate,
            betas=(
                getattr(args, "adam_beta1", 0.9),
                getattr(args, "adam_beta2", 0.999),
            ),
            eps=getattr(args, "adam_epsilon", 1e-8),
            weight_decay=args.weight_decay,
        )
        fp32_opt = FP32MasterWeightOptimizer(
            bf16_params=bf16_params,
            inner_optimizer=inner_optimizer,
            max_grad_norm=args.max_grad_norm,
        )
        self._fp32_optimizer = fp32_opt
        self.optimizer = fp32_opt
        logger.info(
            "Eagle3: using FP32MasterWeightOptimizer (DDP) so Adam moments match "
            "DeepSpeed ZeRO FP32 optimizer state."
        )
        return self.optimizer

    def _clip_grad_norm(self, *args, **kwargs):
        """Skip HF bf16 grad clip when FP32-master optimizers clip internally."""
        if self._fp32_optimizer is not None:
            return torch.tensor(0.0)

        from .fp32_master_optimizer import FP32StateAdamW

        optimizer = self.optimizer
        if hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        if isinstance(optimizer, FP32StateAdamW):
            return torch.tensor(0.0)
        return super()._clip_grad_norm(*args, **kwargs)

    def train(self, *args, **kwargs):
        """Override train method to record training start time for estimating remaining time."""
        self._train_start_time = time.time()
        return super().train(*args, **kwargs)

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        """
        Merge acc/ploss accumulators with the base Trainer's loss log.
        """
        if "loss" in logs and self._train_pending_log:
            train_count = max(self._train_pending_log_count, 1)
            acc_ploss = {k: v / train_count for k, v in self._train_pending_log.items()}
            merged = {}

            # step
            max_steps = 0
            if self.state is not None:
                global_step = self.state.global_step
                max_steps = self.state.max_steps
                merged["step"] = global_step

            # epoch
            if "epoch" in logs:
                merged["epoch"] = logs["epoch"]
            if "loss" in logs:
                merged["loss"] = logs["loss"]
            if "grad_norm" in logs:
                merged["grad_norm"] = logs["grad_norm"]

            if "learning_rate" in logs:
                merged["lr"] = logs["learning_rate"]

            # train acc/ploss
            merged.update(acc_ploss)

            # eval acc/ploss — merged when a training log fires
            if self._eval_pending_log:
                eval_count = max(self._eval_pending_log_count, 1)
                merged.update({k: v / eval_count for k, v in self._eval_pending_log.items()})
                self._eval_pending_log.clear()
                self._eval_pending_log_count = 0

            # remaining_time
            if (
                self.state is not None
                and self._train_start_time is not None
                and global_step > 0
                and max_steps > 0
            ):
                elapsed = time.time() - self._train_start_time
                time_per_step = elapsed / global_step
                remaining_seconds = int(time_per_step * (max_steps - global_step))
                hours, remainder = divmod(remaining_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                merged["remaining_time"] = f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"

            self._train_pending_log.clear()
            self._train_pending_log_count = 0
            super().log(merged, start_time)
        else:
            super().log(logs, start_time)

    @property
    def draft_model(self) -> nn.Module:
        """Underlying draft module (unwrap DDP / Accelerate / DeepSpeed)."""
        model = self.model
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            try:
                return accelerator.unwrap_model(model)
            except Exception:
                pass
        return model.module if hasattr(model, "module") else model

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None,
        return_outputs: bool = False,
    ) -> Tuple[List[torch.Tensor], List, List[float]]:
        """
        Compute the training loss for the model.

        Args:
            model: The model for which to compute the loss
            inputs: Input data dictionary with input_ids, attention_mask,
                loss_mask, position_ids
            num_items_in_batch: Number of items in batch (unused)
            return_outputs: Whether to return model outputs (unused)

        Returns:
            Tuple of (prediction_losses, value_losses, accuracies) for each step
        """
        data_for_draft_model = self.prepare_data_for_draft_model(inputs)

        attention_mask = data_for_draft_model["attention_mask"]  # Batch x Seq
        position_ids = data_for_draft_model["position_ids"]
        input_ids = data_for_draft_model["input_ids"]  # Batch x Seq
        target_logits = data_for_draft_model["target_logits"]  # Batch x Seq x Vocab
        loss_mask = data_for_draft_model["loss_mask"]  # Batch x Seq x 1
        hidden_states = data_for_draft_model["hidden_states"]  # Batch x Seq x Hidden

        hidden_states = self.down_project_hidden_states(hidden_states)
        attention_mask, position_ids = self.prepare_attention_mask_and_position_ids(
            hidden_states, attention_mask, position_ids
        )
        loss = self.draft_model_training_time_test(
            input_ids,
            hidden_states,
            attention_mask,
            position_ids,
            target_logits,
            loss_mask,
            log_prefix="train",
        )

        return loss

    @abstractmethod
    def prepare_data_for_draft_model(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for draft model training.
        """
        pass

    def down_project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Down project hidden states for draft model training.
        """
        # Step 4: Prepare hidden states with gradient tracking
        if not hidden_states.requires_grad:
            hidden_states.requires_grad = True
        hidden_states = self.draft_model.combine_hidden_states(hidden_states)
        return hidden_states

    def prepare_attention_mask_and_position_ids(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare attention mask for draft model training.
        """
        # Step 5: Prepare attention mask and position IDs
        batch_size, seq_length, _ = hidden_states.shape
        device = hidden_states.device

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            if position_ids.ndim == 3:
                # MRoPE format: (3, batch, seq_len), keep as-is
                position_ids = position_ids.long()
            else:
                position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)

        attention_mask = self.draft_model.prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )

        return attention_mask, position_ids

    def draft_model_training_time_test(
        self,
        input_ids,
        hidden_states,
        attention_mask,
        position_ids,
        target_logits,
        loss_mask,
        log_prefix="",
    ):
        _, seq_length, _ = hidden_states.shape

        # Step 6: Initialize containers for losses, accuracies and cache
        plosses, acces = [], []
        skew_kl_losses = []
        draft = self.draft_model  # unwrapped — mode flags / _aux_inject live here
        use_draft_feedback = bool(
            getattr(draft, "progressive_staged", False) or getattr(draft, "hawk", False)
        )
        feedback_applied = 0
        target_shift_applied = 0
        # Snapshot target aux tape (h_target per draft layer) before feedback
        # replaces inject with draft outs. Compare draft h_i vs this tape with
        # Smooth-L1 on speculative tokens 1..3 (idx 0..2).
        n_draft_layers = len(getattr(draft, "layers", []) or [])
        target_aux_tape = None
        if use_draft_feedback and getattr(draft, "_aux_inject", None) is not None:
            target_aux_tape = tuple(t.detach() for t in draft._aux_inject)
        sl1_layer_sum = [0.0] * max(n_draft_layers, 0)
        sl1_layer_count = [0] * max(n_draft_layers, 0)
        # How many speculative tokens (1-indexed) to include in the metric.
        sl1_token_budget = min(3, self.length)
        # Warmup: always teacher-force with shifted GT target aux HS.
        # progressive eagle → progressive GT; hawk / real_hawk → hawk GT.
        target_hs_warmup = (
            log_prefix == "train"
            and (
                getattr(draft, "progressive_staged", False)
                or getattr(draft, "hawk", False)
            )
            and self.target_hs_warmup_steps > 0
            and getattr(self.state, "global_step", 0) < self.target_hs_warmup_steps
        )
        if target_hs_warmup and not getattr(self, "_logged_target_hs_warmup", False):
            warmup_kind = (
                "progressive eagle GT"
                if getattr(draft, "progressive_staged", False)
                else "hawk GT"
            )
            logger.info(
                "Eagle3 target-HS warmup (%s): steps 0..%d use shifted ground-truth "
                "aux HS on every speculative substep; after that, draft-HS feedback.",
                warmup_kind,
                self.target_hs_warmup_steps - 1,
            )
            self._logged_target_hs_warmup = True
        if hasattr(draft, "init_cache_hidden"):
            cache_hidden = draft.init_cache_hidden()
        else:
            cache_hidden = [[], []]

        # Step 7: Iterative speculative decoding training loop
        for idx in range(self.length):
            # Step 7.1: Get input embeddings with gradient tracking
            inputs_embeds = draft.embed_input_ids(input_ids)
            if not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True

            # Step 7.2: Encode through draft model layers
            if getattr(draft, "gradient_checkpointing", False) and draft.training:
                hidden_states, cache_hidden = torch.utils.checkpoint.checkpoint(
                    draft.encode_layers,
                    inputs_embeds,
                    hidden_states,
                    cache_hidden,
                    attention_mask,
                    position_ids,
                    True,
                    use_reentrant=False,
                )
            else:
                hidden_states, cache_hidden = draft.encode_layers(
                    inputs_embeds=inputs_embeds,
                    hidden_states=hidden_states,
                    cache_hidden=cache_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                )

            # Step 7.2b: Smooth-L1(draft h_i, target aux_i) on tokens 1..3.
            if (
                target_aux_tape is not None
                and idx < sl1_token_budget
                and getattr(draft, "_last_layer_outs", None)
            ):
                with torch.no_grad():
                    outs = draft._last_layer_outs
                    for li, (h_draft, h_tgt) in enumerate(
                        zip(outs, target_aux_tape)
                    ):
                        if h_draft.shape != h_tgt.shape:
                            continue
                        sl1_layer_sum[li] += torch.nn.functional.smooth_l1_loss(
                            h_draft.detach().float(),
                            h_tgt.float(),
                        ).item()
                        sl1_layer_count[li] += 1

            # Step 7.3: Compute logits from hidden states
            logits = draft.compute_logits(hidden_states)

            # Step 7.4: Compute target distribution and position mask
            with torch.no_grad():
                target_max_token = target_logits.argmax(-1)
                target_mask = draft.t2d[target_max_token][..., None].int()
                position_mask = target_mask * loss_mask

                target_head = target_logits[..., draft.t2d].float()
                target_p = nn.Softmax(dim=2)(target_head).detach()

            # Step 7.5: Compute loss
            out_logp = nn.LogSoftmax(dim=2)(logits)
            loss = -torch.sum(position_mask * target_p * out_logp, dim=2).mean()

            if log_prefix == "train" and self.skew_kl_loss_weight > 0.0:
                stage_weight = self._stage_weight(
                    self.skew_kl_stage_weights, idx, default=1.0
                )
                if stage_weight != 0.0:
                    alpha = min(max(self.skew_kl_alpha, 0.0), 1.0)
                    out_logp_f = out_logp.float()
                    out_p = out_logp_f.exp()
                    eps = torch.finfo(torch.float32).tiny
                    direction = self.skew_kl_direction.lower()
                    terms = []
                    if direction in ("forward", "bidirectional", "both"):
                        mix_q = alpha * target_p + (1.0 - alpha) * out_p
                        terms.append(
                            target_p * (target_p.clamp_min(eps).log() - mix_q.clamp_min(eps).log())
                        )
                    if direction in ("reverse", "bidirectional", "both"):
                        mix_p = alpha * out_p + (1.0 - alpha) * target_p
                        terms.append(out_p * (out_logp_f - mix_p.clamp_min(eps).log()))
                    if not terms:
                        raise ValueError(
                            "skew_kl_direction must be forward, reverse, or bidirectional"
                        )
                    skew_kl = sum(torch.sum(term, dim=-1, keepdim=True) for term in terms)
                    skew_kl_step = self._masked_mean(skew_kl, position_mask)
                    if self.skew_kl_loss_mode.lower() == "replace":
                        beta = max(0.0, min(1.0, self.skew_kl_loss_weight * stage_weight))
                        loss = (1.0 - beta) * loss + beta * skew_kl_step
                        skew_kl_losses.append(skew_kl_step)
                    elif self.skew_kl_loss_mode.lower() == "additive":
                        skew_kl_losses.append(stage_weight * skew_kl_step)
                    else:
                        raise ValueError(
                            "skew_kl_loss_mode must be additive or replace"
                        )

            # Step 7.6: Compute accuracy
            with torch.no_grad():
                correct = (logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)
                accuracy = correct.sum().item() / (loss_mask.sum().item() + 1e-6)

            # Step 7.7: Store loss and accuracy
            plosses.append(loss)
            acces.append(accuracy)

            # Step 7.8: Update inputs for next iteration (skip on last step)
            if idx < self.length - 1:
                input_ids = padding(input_ids, left=False)
                target_logits = padding(target_logits, left=False)
                loss_mask = padding(loss_mask, left=False)
                # Keep target aux tape aligned with the shifted sequence so
                # tokens 2/3 still compare h_i vs the matching target HS.
                if target_aux_tape is not None and idx + 1 < sl1_token_budget:
                    target_aux_tape = tuple(
                        padding(t, left=False) for t in target_aux_tape
                    )
                # Warmup: teacher-force with shifted GT target HS for all
                # speculative substeps (progressive eagle GT or hawk GT).
                # Outside warmup, progressive/hawk use same-depth draft outs.
                if target_hs_warmup:
                    if not hasattr(draft, "shift_aux_inject"):
                        raise RuntimeError(
                            "target-HS warmup requires shift_aux_inject"
                        )
                    draft.shift_aux_inject(
                        left=False, allow_target_hs_warmup=True
                    )
                    if hasattr(draft, "next_hidden_from_encode"):
                        hidden_states = draft.next_hidden_from_encode(hidden_states)
                    target_shift_applied += 1
                elif use_draft_feedback:
                    if not hasattr(draft, "take_progressive_draft_feedback"):
                        raise RuntimeError(
                            "progressive/hawk draft missing take_progressive_draft_feedback"
                        )
                    # Prove encode stored per-layer outs before we consume them.
                    n_outs = len(getattr(draft, "_last_layer_outs", None) or [])
                    if n_outs != len(draft.layers):
                        raise RuntimeError(
                            "progressive/hawk encode_layers did not store per-layer outs "
                            f"(got {n_outs}, expected {len(draft.layers)}). "
                            "Draft-HS feedback cannot run — check model unwrap / mode flags."
                        )
                    prev_inject0 = (
                        draft._aux_inject[0] if draft._aux_inject is not None else None
                    )
                    seed = draft.take_progressive_draft_feedback()
                    if seed is None:
                        raise RuntimeError(
                            "take_progressive_draft_feedback returned None after encode"
                        )
                    # Evidence: inject tape must now be the draft layer outs, not
                    # the target aux from combine_hidden_states.
                    if draft._aux_inject is None or draft._aux_inject[0] is prev_inject0:
                        raise RuntimeError(
                            "draft-HS feedback did not replace _aux_inject with layer outs"
                        )
                    if draft._aux_inject[0] is not draft._last_layer_outs[0]:
                        raise RuntimeError(
                            "draft-HS feedback inject[0] is not encode layer-out h0"
                        )
                    hidden_states = seed
                    feedback_applied += 1
                    if not getattr(self, "_logged_draft_hs_feedback", False):
                        mode = (
                            "hawk"
                            if getattr(draft, "hawk", False)
                            else "progressive_staged"
                        )
                        logger.info(
                            "Eagle3 %s: target HS only on first draft token; "
                            "steps 1+ use draft outs only (L0←%s, injects←h0..h%d). "
                            "train/draft_hs_feedback=1 required. "
                            "Logging Smooth-L1(h_i, h_target_i) on tokens 1..%d.",
                            mode,
                            (
                                "post-norm(h_last)"
                                if getattr(draft, "norm_output", False)
                                else "h0"
                            ),
                            len(draft.layers) - 1,
                            sl1_token_budget,
                        )
                        self._logged_draft_hs_feedback = True
                # Stock fused_fc: EAGLE 3.1 feeds post-norm HS into the next step.
                else:
                    if hasattr(draft, "next_hidden_from_encode"):
                        hidden_states = draft.next_hidden_from_encode(hidden_states)
                    if hasattr(draft, "shift_aux_inject"):
                        draft.shift_aux_inject(left=False)

        if (
            use_draft_feedback
            and not target_hs_warmup
            and feedback_applied != max(self.length - 1, 0)
        ):
            raise RuntimeError(
                f"draft-HS feedback expected {max(self.length - 1, 0)} applies, "
                f"got {feedback_applied}"
            )
        if target_hs_warmup and target_shift_applied != max(self.length - 1, 0):
            raise RuntimeError(
                f"target-HS warmup expected {max(self.length - 1, 0)} applies, "
                f"got {target_shift_applied}"
            )

        # Step 8: Compute weighted loss
        ploss_weight = [0.8**i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        skew_kl_loss = (
            sum(skew_kl_losses) / len(skew_kl_losses)
            if skew_kl_losses
            else ploss.new_zeros(())
        )
        skew_kl_add_weight = (
            self.skew_kl_loss_weight
            if self.skew_kl_loss_mode.lower() == "additive"
            else 0.0
        )
        ploss = ploss + skew_kl_add_weight * skew_kl_loss

        if (
            log_prefix == "train"
            and not self._logged_aux_losses
            and self.skew_kl_loss_weight > 0.0
        ):
            logger.info(
                "Eagle3 aux losses: skew_kl(w=%s,a=%s,dir=%s,mode=%s)",
                self.skew_kl_loss_weight,
                self.skew_kl_alpha,
                self.skew_kl_direction,
                self.skew_kl_loss_mode,
            )
            self._logged_aux_losses = True

        log = {f"{log_prefix}/acc_{i}": acces[i] for i in range(len(acces))}
        log.update({f"{log_prefix}/ploss_{i}": plosses[i].item() for i in range(len(plosses))})
        if log_prefix == "train" and skew_kl_losses:
            log[f"{log_prefix}/skew_kl_loss"] = skew_kl_loss.item()
        # 1.0 => draft-HS feedback ran every speculative step after 0; 0 => off/stock.
        log[f"{log_prefix}/draft_hs_feedback"] = (
            float(feedback_applied) / float(max(self.length - 1, 1))
            if use_draft_feedback
            else 0.0
        )
        log[f"{log_prefix}/target_hs_warmup"] = (
            float(target_shift_applied) / float(max(self.length - 1, 1))
            if target_hs_warmup
            else 0.0
        )
        # Per-layer Smooth-L1(draft h_i, target aux_i), averaged over tokens 1..3.
        for li, (s, c) in enumerate(zip(sl1_layer_sum, sl1_layer_count)):
            if c > 0:
                log[f"{log_prefix}/aux_vs_draft_sl1_h{li}"] = s / float(c)
        # Route into the appropriate accumulator.
        if log_prefix == "eval":
            for k, v in log.items():
                self._eval_pending_log[k] = self._eval_pending_log.get(k, 0.0) + v
            self._eval_pending_log_count += 1
        else:
            for k, v in log.items():
                self._train_pending_log[k] = self._train_pending_log.get(k, 0.0) + v
            self._train_pending_log_count += 1
        # Step 9: Return loss
        return ploss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to handle DeepSpeed ZeRO-3 model saving.

        Args:
            output_dir: Directory to save the model. If None, uses self.args.output_dir
            _internal_call: Internal flag used by Trainer
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # Check if using DeepSpeed ZeRO-3
        is_deepspeed_zero3 = (
            self.is_deepspeed_enabled
            and hasattr(self.accelerator.state, "deepspeed_plugin")
            and self.accelerator.state.deepspeed_plugin.zero_stage == 3
        )

        if is_deepspeed_zero3:
            # Handle ZeRO-3 model saving
            self._save_zero3_model(output_dir, _internal_call)
        else:
            # Fall back to parent class save_model
            super().save_model(output_dir, _internal_call)

    def _save_zero3_model(self, output_dir: str, _internal_call: bool = False):
        """
        Save model with DeepSpeed ZeRO-3 specific logic.

        Args:
            output_dir: Directory to save the model
            _internal_call: Internal flag used by Trainer
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save with DeepSpeed's state_dict gathering
        # All processes must participate in parameter gathering to avoid deadlock
        with deepspeed.zero.GatheredParameters(self.model.parameters()):
            state_dict = self.model.state_dict()

        # Only main process saves the model
        if self.args.should_save and self.accelerator.is_main_process:
            self.model.save_pretrained(
                output_dir,
                is_main_process=True,
                state_dict=state_dict,
                save_function=torch.save,
            )

            # Save training arguments
            from transformers.trainer import TRAINING_ARGS_NAME

            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Wait for all processes
        self.accelerator.wait_for_everyone()

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        """
        data_for_draft_model = self.prepare_data_for_draft_model(inputs)

        attention_mask = data_for_draft_model["attention_mask"]
        # inputs_embeds = data_for_draft_model["inputs_embeds"]
        position_ids = data_for_draft_model.get("position_ids", None)
        input_ids = data_for_draft_model["input_ids"]
        target_logits = data_for_draft_model["target_logits"]
        loss_mask = data_for_draft_model["loss_mask"]
        hidden_states = data_for_draft_model["hidden_states"]

        with torch.no_grad():
            hidden_states = self.down_project_hidden_states(hidden_states)
            attention_mask, position_ids = self.prepare_attention_mask_and_position_ids(
                hidden_states, attention_mask, position_ids
            )
            loss = self.draft_model_training_time_test(
                input_ids,
                hidden_states,
                attention_mask,
                position_ids,
                target_logits,
                loss_mask,
                log_prefix="eval",
            )
        return (loss, None, None)
