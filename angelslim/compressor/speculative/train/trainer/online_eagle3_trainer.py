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

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

try:
    from ....vistoken.splice import compress_image_rows
    from ....vistoken.target_attn_prune import TargetQKCapture, prune_sample_image_rows
except ModuleNotFoundError:
    compress_image_rows = None
    TargetQKCapture = None
    prune_sample_image_rows = None
from ...utils import padding
from .eagle3_trainer import Eagle3Trainer
from .trainer_factory import Eagle3TrainerFactory


@Eagle3TrainerFactory.register("online", "LLM")
class OnlineEagle3Trainer(Eagle3Trainer):
    """
    Online EAGLE3 Trainer for speculative decoding training.

    Implements training logic for EAGLE3 model using a draft model to predict
    tokens based on hidden states from a target model.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        length: int,
        draft_model_config: Dict[str, Any],
        **kwargs,
    ):
        """
        Initialize the OnlineEagle3Trainer.

        Args:
            draft_model: Draft model for token prediction
            target_model: Target model for generating hidden states
            length: Number of speculative decoding steps
            draft_model_config: Configuration dictionary for draft model
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(
            draft_model=draft_model,
            length=length,
            draft_model_config=draft_model_config,
            **kwargs,
        )
        self.target_model = target_model
        self._aux_hidden_states_layer_ids = getattr(
            draft_model_config, "aux_hidden_states_layer_ids", None
        )

    def prepare_data_for_draft_model(self, inputs):
        # Step 1: Extract input tensors
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        loss_mask = inputs["loss_mask"]
        position_ids = inputs.get("position_ids", None)

        # Step 2: Get hidden states and logits from target model
        hidden_states, target_logits = self.target_model.get_hidden_states_and_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aux_hidden_states_layer_ids=self._aux_hidden_states_layer_ids,
        )

        # Step 3: Apply right padding and move tensors to correct device
        target_logits = padding(target_logits, left=False).to(input_ids.device)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(input_ids.device)

        return {
            "hidden_states": hidden_states,
            "target_logits": target_logits,
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }


@Eagle3TrainerFactory.register("online", "VLM")
class OnlineVLMEagle3Trainer(Eagle3Trainer):
    """
    Online EAGLE3 Trainer for speculative decoding training.
    Implements training logic for EAGLE3 model using a draft model to predict
    tokens based on hidden states from a target model.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        length: int,
        draft_model_config: Dict[str, Any],
        **kwargs,
    ):
        """
        Initialize the OnlineEagle3Trainer.
        Args:
            draft_model: Draft model for token prediction
            target_model: Target model for generating hidden states
            length: Number of speculative decoding steps
            draft_model_config: Configuration dictionary for draft model
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(
            draft_model=draft_model,
            length=length,
            draft_model_config=draft_model_config,
            **kwargs,
        )
        self.target_model = target_model
        self._aux_hidden_states_layer_ids = getattr(
            draft_model_config, "aux_hidden_states_layer_ids", None
        )
        # Optional target-attention row pruning in front of vistoken's
        # compressor -- see angelslim/compressor/vistoken/target_attn_prune.py.
        # {"group_size": x, "keep_m": M, "mode": "target_attn"|"random"|"none"}
        self._vistoken_prune_cfg = getattr(draft_model_config, "vistoken_prune", None)
        self._target_head_dim = None

    def _get_target_head_dim(self) -> int:
        if self._target_head_dim is None:
            cfg = getattr(self.target_model.model, "config", None)
            cfg = getattr(cfg, "text_config", cfg)
            head_dim = getattr(cfg, "head_dim", None)
            if head_dim is None:
                head_dim = cfg.hidden_size // cfg.num_attention_heads
            self._target_head_dim = int(head_dim)
        return self._target_head_dim

    def prepare_data_for_draft_model(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        loss_mask = inputs["loss_mask"]

        gist_embeddings = inputs.get("gist_embeddings")
        kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "attention_mask", "loss_mask", "gist_embeddings"]
        }
        if self.branch_distill_loss_weight > 0.0:
            # Branch distillation re-scores a substituted copy of this sequence,
            # so keep it (and the image kwargs) before the left shift below.
            self._branch_ctx = {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask,
                "kwargs": kwargs,
            }
        # Row compression, before the left shift: every tensor here is still
        # aligned 1:1 on absolute positions, so one splice keeps them so.
        compressor = getattr(self.draft_model, "vistoken", None)
        prune_cfg = self._vistoken_prune_cfg if compressor is not None else None
        prune_mode = (prune_cfg or {}).get("mode", "none")
        layer_ids = self._aux_hidden_states_layer_ids

        if prune_mode != "none" and prune_mode != "random":
            if TargetQKCapture is None:
                raise RuntimeError(
                    "vistoken target-attention pruning is not available on clean main; "
                    "use the visual compression feature branch"
                )
            with TargetQKCapture(self.target_model.get_text_decoder_layers(), layer_ids) as qk_cap:
                hidden_states, target_logits, _, position_ids = (
                    self.target_model.get_hidden_states_and_logits(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        aux_hidden_states_layer_ids=self._aux_hidden_states_layer_ids,
                        **kwargs,
                    )
                )
        else:
            qk_cap = None
            # Get hidden states and logits from target model
            hidden_states, target_logits, _, position_ids = (
                self.target_model.get_hidden_states_and_logits(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    aux_hidden_states_layer_ids=self._aux_hidden_states_layer_ids,
                    **kwargs,
                )
            )

        if compressor is not None:
            if compress_image_rows is None:
                raise RuntimeError(
                    "vistoken compression is not available on clean main; use the visual compression feature branch"
                )
            compressor_rows_per_sample = None
            if prune_mode != "none":
                bsz = input_ids.shape[0]
                valid = (
                    attention_mask.bool()
                    if attention_mask is not None
                    else torch.ones_like(input_ids, dtype=torch.bool)
                )
                compressor_rows_per_sample = [
                    prune_sample_image_rows(
                        qk_cap,
                        layer_ids,
                        self._get_target_head_dim(),
                        input_ids[b],
                        valid[b],
                        loss_mask[b],
                        self.draft_model.vistoken_image_token_id,
                        compressor.cfg.tile_tokens,
                        prune_cfg["group_size"],
                        prune_cfg["keep_m"],
                        prune_mode,
                    )
                    for b in range(bsz)
                ]
            spliced = compress_image_rows(
                compressor,
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                target_logits=target_logits,
                position_ids=position_ids,
                image_token_id=self.draft_model.vistoken_image_token_id,
                compressor_rows_per_sample=compressor_rows_per_sample,
            )
            hidden_states = spliced["hidden_states"]
            input_ids = spliced["input_ids"]
            target_logits = spliced["target_logits"]
            loss_mask = spliced["loss_mask"]
            attention_mask = spliced["attention_mask"]
            position_ids = spliced["position_ids"]
            if gist_embeddings is not None:
                raise NotImplementedError(
                    "gist conditioning and row compression both rewrite the "
                    "draft sequence; they have not been reconciled"
                )

        # Apply right padding and move tensors to correct device
        target_logits = padding(target_logits, left=False).to(input_ids.device)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(input_ids.device)

        result = {
            "hidden_states": hidden_states,
            "target_logits": target_logits,
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        if gist_embeddings is not None:
            result["gist_embeddings"] = padding(gist_embeddings, left=False).to(
                input_ids.device
            )
        return result

    def branch_teacher_logits(self, input_ids):
        """Re-score a branch-substituted sequence with the target model.

        Same images and mask as the batch it came from; only the token ids at
        the branch positions differ.
        """
        ctx = getattr(self, "_branch_ctx", None)
        if ctx is None:
            raise RuntimeError(
                "branch distillation ran before prepare_data_for_draft_model "
                "stashed the unshifted sequence"
            )
        with torch.no_grad():
            _, target_logits, _, _ = self.target_model.get_hidden_states_and_logits(
                input_ids=input_ids,
                attention_mask=ctx["attention_mask"],
                aux_hidden_states_layer_ids=self._aux_hidden_states_layer_ids,
                **ctx["kwargs"],
            )
        return target_logits.detach().to(input_ids.device)


@Eagle3TrainerFactory.register("online", "Audio")
class OnlineAudioEagle3Trainer(Eagle3Trainer):
    """
    Online EAGLE3 Trainer for speculative decoding training.
    Implements training logic for EAGLE3 model using a draft model to predict
    tokens based on hidden states from a target model.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        length: int,
        draft_model_config: Dict[str, Any],
        **kwargs,
    ):
        """
        Initialize the OnlineEagle3Trainer.
        Args:
            draft_model: Draft model for token prediction
            target_model: Target model for generating hidden states
            length: Number of speculative decoding steps
            draft_model_config: Configuration dictionary for draft model
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(
            draft_model=draft_model,
            length=length,
            draft_model_config=draft_model_config,
            **kwargs,
        )
        self.target_model = target_model
        self._aux_hidden_states_layer_ids = getattr(
            draft_model_config, "aux_hidden_states_layer_ids", None
        )

    def prepare_data_for_draft_model(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        loss_mask = inputs["loss_mask"]

        kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "attention_mask", "loss_mask"]
        }
        # Get hidden states and logits from target model
        hidden_states, target_logits, _, position_ids = (
            self.target_model.get_hidden_states_and_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                aux_hidden_states_layer_ids=self._aux_hidden_states_layer_ids,
                **kwargs,
            )
        )

        # Apply right padding and move tensors to correct device
        target_logits = padding(target_logits, left=False).to(input_ids.device)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(input_ids.device)

        result_dict = {}
        result_dict.update(kwargs)
        result_dict.update(
            {
                "hidden_states": hidden_states,
                "target_logits": target_logits,
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "attention_mask": attention_mask,
            }
        )
        return result_dict


@Eagle3TrainerFactory.register("online", "TTS")
class OnlineTTSEagle3Trainer(Eagle3Trainer):
    """
    Online EAGLE3 Trainer for speculative decoding training.

    Implements training logic for EAGLE3 model using a draft model to predict
    tokens based on hidden states from a target model.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        length: int,
        draft_model_config: Dict[str, Any],
        **kwargs,
    ):
        """
        Initialize the OnlineEagle3Trainer.
        Args:
            draft_model: Draft model for token prediction
            target_model: Target model for generating hidden states
            length: Number of speculative decoding steps
            draft_model_config: Configuration dictionary for draft model
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(
            draft_model=draft_model,
            length=length,
            draft_model_config=draft_model_config,
            **kwargs,
        )
        self.target_model = target_model

    def prepare_data_for_draft_model(self, inputs):
        if self.target_model.backend.model_name == "cosyvoice3":
            data_for_draft_model = self._prepare_data_for_draft_model_cosyvoice3(inputs)
        else:
            raise NotImplementedError("This model is not implemented")
        return data_for_draft_model

    def _prepare_data_for_draft_model_cosyvoice3(self, inputs):
        (
            hidden_states,
            target_logits,
            inputs_embeds,
            loss_mask,
            attention_mask,
        ) = self.target_model.get_hidden_states_and_logits(input_ids=inputs)

        device = inputs_embeds.device
        dtype = self.draft_model.fc.weight.dtype

        target_logits = padding(target_logits, left=False).to(device)
        inputs_embeds = padding(inputs_embeds, left=False)
        loss_mask = loss_mask[..., None].to(device)

        return {
            "hidden_states": hidden_states.to(dtype),
            "target_logits": target_logits,
            "inputs_embeds": inputs_embeds.to(dtype),
            "loss_mask": loss_mask,
            "position_ids": None,
            "attention_mask": attention_mask.squeeze(0),
        }

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
        position_ids = data_for_draft_model["position_ids"]  # Batch x Seq
        target_logits = data_for_draft_model["target_logits"]  # Batch x Seq x Vocab
        loss_mask = data_for_draft_model["loss_mask"]  # Batch x Seq x 1
        hidden_states = data_for_draft_model["hidden_states"]  # Batch x Seq x Hidden
        inputs_embeds = data_for_draft_model["inputs_embeds"]

        hidden_states = self.down_project_hidden_states(hidden_states)
        attention_mask, position_ids = self.prepare_attention_mask_and_position_ids(
            hidden_states, attention_mask, position_ids
        )
        loss = self.draft_model_training_time_test(
            hidden_states,
            attention_mask,
            position_ids,
            target_logits,
            loss_mask,
            inputs_embeds,
            log_prefix="train",
        )

        return loss

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
        data_for_draft_model = self.prepare_data_for_draft_model(**inputs)

        attention_mask = data_for_draft_model["attention_mask"]  # Batch x Seq
        position_ids = data_for_draft_model["position_ids"]  # Batch x Seq
        target_logits = data_for_draft_model["target_logits"]  # Batch x Seq x Vocab
        loss_mask = data_for_draft_model["loss_mask"]  # Batch x Seq x 1
        hidden_states = data_for_draft_model["hidden_states"]  # Batch x Seq x Hidden
        inputs_embeds = data_for_draft_model["inputs_embeds"]

        with torch.no_grad():
            hidden_states = self.down_project_hidden_states(hidden_states)
            attention_mask, position_ids = self.prepare_attention_mask_and_position_ids(
                hidden_states, attention_mask, position_ids
            )
            loss = self.draft_model_training_time_test(
                hidden_states,
                attention_mask,
                position_ids,
                target_logits,
                loss_mask,
                inputs_embeds,
                log_prefix="eval",
            )
        return (loss, None, None)

    def draft_model_training_time_test(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        target_logits,
        loss_mask,
        inputs_embeds,
        log_prefix="",
    ):
        # Step 6: Initialize containers for losses, accuracies and cache
        plosses, acces = [], []
        if hasattr(self.draft_model, "init_cache_hidden"):
            cache_hidden = self.draft_model.init_cache_hidden()
        else:
            cache_hidden = [[], []]

        # Step 7: Iterative speculative decoding training loop
        for idx in range(self.length):
            # Step 7.1: Get input embeddings with gradient tracking
            if not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True

            # Step 7.2: Encode through draft model layers
            hidden_states, cache_hidden = self.draft_model.encode_layers(
                inputs_embeds=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

            # Step 7.3: Compute logits from hidden states
            logits = self.draft_model.compute_logits(hidden_states)

            # Step 7.4: Compute target distribution and position mask
            with torch.no_grad():
                target_max_token = target_logits.argmax(-1)
                target_mask = self.draft_model.t2d[target_max_token][..., None].int()
                position_mask = target_mask * loss_mask

                target_head = target_logits[..., self.draft_model.t2d].float()
                target_p = nn.Softmax(dim=2)(target_head).detach()

            # Step 7.5: Compute loss
            out_logp = nn.LogSoftmax(dim=2)(logits)
            loss = -torch.sum(position_mask * target_p * out_logp, dim=2).mean()

            # Step 7.6: Compute accuracy
            with torch.no_grad():
                correct = (logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)
                accuracy = correct.sum().item() / (loss_mask.sum().item() + 1e-6)

            # Step 7.7: Store loss and accuracy
            plosses.append(loss)
            acces.append(accuracy)

            # Step 7.8: Update inputs for next iteration (skip on last step)
            if idx < self.length - 1:
                inputs_embeds = padding(inputs_embeds, left=False)
                target_logits = padding(target_logits, left=False)
                loss_mask = padding(loss_mask, left=False)

        # Step 8: Compute weighted loss
        ploss_weight = [0.8**i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])

        log = {f"{log_prefix}/acc_{i}": round(float(acces[i]), 3) for i in range(len(acces))}
        log.update(
            {
                f"{log_prefix}/ploss_{i}": round(float(plosses[i].item()), 3)
                for i in range(len(plosses))
            }
        )
        self.log(log)

        # Step 9: Return loss
        return ploss
