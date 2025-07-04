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

import re

import torch
import torch.nn as nn


def filter_func(name):
    pattern = re.compile(
        r".*(mlp_t5|pooler|style_embedder|x_embedder|t_embedder|extra_embedder).*"
    )
    return pattern.match(name) is not None


class DiTHook:
    def __init__(self, model, use_transformer_engine=False):
        """
        Args:
            model(nn.Moudle, required): the model to be quant
        """
        self.model = model
        self.input_activation = {}
        self.output_activation = {}
        self.input_activation_cnt = {}
        self.output_activation_cnt = {}
        self.use_transformer_engine = use_transformer_engine
        self._apply_hook()

    def _apply_hook(self):
        self._forward_hook_list = []
        for name, sub_layer in self.model.named_modules():
            if filter_func(name):
                continue
            instance_list = nn.Linear
            if isinstance(sub_layer, instance_list):
                forward_pre_hook_handle = sub_layer.register_forward_hook(
                    self._forward_pre_hook
                )
                self._forward_hook_list.append(forward_pre_hook_handle)

    def _forward_pre_hook(self, layer, input, output):
        layer_name = ""
        for name, module in self.model.named_modules():
            if filter_func(name):
                continue
            if module == layer:
                layer_name = name
                break
        x = (
            output[0].detach().cpu()
            if isinstance(output, tuple)
            else output.detach().cpu()
        )
        self.output_activation[layer_name] = (
            self.output_activation.get(layer_name, torch.zeros(x.shape).to(x.dtype)) + x
        )
        self.output_activation_cnt[layer_name] = (
            self.output_activation_cnt.get(layer_name, 0) + 1
        )
        y = (
            input[0].detach().cpu()
            if isinstance(input, tuple)
            else input.detach().cpu()
        )
        self.input_activation[layer_name] = (
            self.input_activation.get(layer_name, torch.zeros(y.shape).to(y.dtype)) + y
        )
        self.input_activation_cnt[layer_name] = (
            self.input_activation_cnt.get(layer_name, 0) + 1
        )

    def remove_hook(self):
        for hook in self._forward_hook_list:
            hook.remove()
        self._forward_hook_list = []

    def clean_acitvation_list(self):
        self.input_activation = {}
        self.output_activation = {}
