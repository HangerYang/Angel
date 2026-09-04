"""KV-cache enabled wrapper for the Idefics3 / SmolVLM architecture.

`HuggingFaceTB/SmolVLM-256M-Instruct` (and the other SmolVLM checkpoints that use a
SmolLM2 backbone) are stored with `architectures=["Idefics3ForConditionalGeneration"]`.
The text decoder is a plain Llama model, so we can reuse ViSpec's pre-allocated
`KVCache`-based Llama decoder (`modeling_llama_kv.LlamaModel`) exactly like the LLaVA
path does, and only override the multimodal `forward` so that:

  * the text decoder receives ViSpec's `KVCache` `past_key_values` (instead of an HF
    `Cache`), and is *not* passed the `cache_position` kwarg it does not understand;
  * the "if inputs_embeds is passed on the first call, input_ids must not be None"
    guard from the stock model is dropped (ViSpec prefills with fused `inputs_embeds`
    and `input_ids=None`);
  * the last-layer hidden states are always returned for the draft model.

To keep the rest of ViSpec untouched we also expose a lightweight ``language_model``
view that mimics a ``*ForCausalLM`` (``.model`` = the KV Llama decoder, ``.lm_head``,
``.config`` = the text config).  This lets every existing ``base_model.language_model``
code path (KV-cache init, tree-mask injection, lm_head routing) work as-is.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Idefics3Config, Idefics3ForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.idefics3.modeling_idefics3 import (
    Idefics3BaseModelOutputWithPast,
    Idefics3Model,
)

from .modeling_llama_kv import LlamaModel as KVLlamaModel


class _CausalLMView:
    """A minimal stand-in for a ``*ForCausalLM`` so that ViSpec's existing
    ``base_model.language_model`` code paths keep working for Idefics3/SmolVLM.

    It is deliberately *not* an ``nn.Module`` so it does not register any duplicate
    parameters; it merely references the already-built KV text decoder and lm_head.
    """

    def __init__(self, decoder: nn.Module, lm_head: nn.Module, config):
        self.model = decoder
        self.lm_head = lm_head
        self.config = config

    @property
    def dtype(self):
        return self.lm_head.weight.dtype

    @property
    def device(self):
        return self.lm_head.weight.device


class CustomIdefics3Model(Idefics3Model):
    """Idefics3 model whose text decoder is ViSpec's KV-cache Llama decoder."""

    def __init__(self, config: Idefics3Config):
        super().__init__(config)
        # Replace the stock HF text decoder with the pre-allocated KVCache Llama decoder.
        self.text_model = KVLlamaModel(config.text_config)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Encode images into the text embedding space (vision tower + connector).

        Mirrors the vision block of the stock ``Idefics3Model.forward``.
        """
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.dtype)
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            pixel_attention_mask = pixel_attention_mask.view(
                batch_size * num_images, *pixel_attention_mask.shape[2:]
            )
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        image_hidden_states = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        ).last_hidden_state
        image_hidden_states = self.connector(image_hidden_states)
        return image_hidden_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Idefics3BaseModelOutputWithPast]:
        output_hidden_states = True if output_hidden_states is None else output_hidden_states
        return_dict = True if return_dict is None else return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.text_model.embed_tokens(input_ids)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError(
                "You cannot specify both pixel_values and image_hidden_states at the same time"
            )
        elif pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(
                dtype=self.dtype, device=inputs_embeds.device
            )

        # Only merge image features when they were freshly encoded (i.e. this is the
        # prefill step and input_ids is available to locate the <image> placeholders).
        # During ViSpec speculative decoding the fusion has already been done and the
        # base model is called with fused inputs_embeds + input_ids=None.
        if image_hidden_states is not None and input_ids is not None:
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not return_dict:
            return tuple(
                v
                for v in [
                    outputs.last_hidden_state,
                    outputs.past_key_values,
                    outputs.hidden_states,
                    outputs.attentions,
                    image_hidden_states,
                ]
                if v is not None
            )

        return Idefics3BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


class CustomIdefics3ForConditionalGeneration(Idefics3ForConditionalGeneration):
    """Idefics3/SmolVLM generation model backed by ViSpec's KVCache Llama decoder."""

    def __init__(self, config: Idefics3Config):
        super().__init__(config)
        self.model = CustomIdefics3Model(config)
        self.post_init()
        # Expose a ``*ForCausalLM``-like view so ViSpec's existing ``language_model``
        # branches (KV init / tree mask / lm_head) work without further changes.
        self.language_model = _CausalLMView(
            self.model.text_model, self.lm_head, config.text_config
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_hidden_states = True if output_hidden_states is None else output_hidden_states

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
