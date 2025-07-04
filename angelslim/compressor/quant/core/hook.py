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

from ..observers import ParentObserver, PTQObserver
from .quant_func import get_fp_maxval, get_fp_search_maxval

__all__ = ["PTQHook", "DiTHook"]


class PTQHook:
    def __init__(self, model):
        self.quant_model = model
        self._forward_hook_list = []
        # {name: layer}
        self.quant_layers_dict = {}
        # {layer: observer}
        self.observer_dict = {}
        self.kv_names = []

    def apply_hook(self):
        self.quant_layers_dict = self.quant_model.get_observer_layers()
        self.kv_names = self.quant_model.get_kvcache_observer_layers_names(
            self.quant_layers_dict.keys()
        )
        act_observer = self.quant_model.quant_algo_dict["act_observer"]
        weight_observer = self.quant_model.quant_algo_dict["weight_observer"]
        kv_cache_observer = self.quant_model.quant_algo_dict["kv_cache_observer"]

        quant_parent_dict = self.quant_model.get_parent_dict(self.quant_layers_dict)
        parent_observers = {
            v: ParentObserver() for v in set(quant_parent_dict.values())
        }

        # apply observers
        for name, sub_layer in self.quant_layers_dict.items():
            extra_kwargs = (
                {"parent_observer": parent_observers[quant_parent_dict[name]]}
                if name in quant_parent_dict
                else {}
            )
            observer = PTQObserver(
                sub_layer,
                act_observer,
                weight_observer,
                kv_cache_observer if name in self.kv_names else None,
                self.quant_model.quant_algo_dict,
                **extra_kwargs
            )
            forward_hook_handle = sub_layer.register_forward_hook(self._forward_hook)
            self.observer_dict[sub_layer] = observer
            self._forward_hook_list.append(forward_hook_handle)

    def apply_smooth_hook(self, smooth_mapping_layers, smooth_observer):
        for smooth_layer, _ in smooth_mapping_layers.values():
            observer = PTQObserver(
                smooth_layer,
                act_observer=None,
                weight_observer=None,
                kv_cache_observer=None,
                quant_algo_dict=self.quant_model.quant_algo_dict,
                smooth_act_observer=smooth_observer,
            )
            forward_hook_handle = smooth_layer.register_forward_hook(self._forward_hook)
            self.observer_dict[smooth_layer] = observer
            self._forward_hook_list.append(forward_hook_handle)

    def _forward_hook(self, layer, input, output):
        x = input[0].clone() if isinstance(input, tuple) else input.clone()
        y = output[0].clone() if isinstance(output, tuple) else output.clone()
        if hasattr(self.quant_model, "apply_layer_norm_list"):
            if layer in self.quant_model.apply_layer_norm_list:
                x = self.quant_model.apply_layer_norm(layer, x)
        self.observer_dict[layer](x, y)
        return output

    def remove_hook(self):
        for hook in self._forward_hook_list:
            hook.remove()
        self._forward_hook_list = []

    def post_process(self):
        maxval = get_fp_maxval(bits=8)
        if self.quant_model.quant_algo_dict["w_quant_algo"] == "fp8":
            for k, v in self.quant_model.weight_scales_dict.items():
                self.quant_model.weight_scales_dict[k] = v / maxval.type(v.dtype)
        if self.quant_model.quant_algo_dict["a_quant_algo"] == "fp8":
            for name, sub_layer in self.quant_layers_dict.items():
                if sub_layer in self.observer_dict:
                    if name in self.quant_model.act_scales_dict.keys():
                        act_dtype = self.quant_model.act_scales_dict[name].dtype
                        if "Search" in str(self.observer_dict[sub_layer]):
                            tmp_maxval = get_fp_search_maxval(
                                self.observer_dict[sub_layer].sampled_input
                            )
                            self.quant_model.act_scales_dict[name] = (
                                self.quant_model.act_scales_dict[name]
                                / tmp_maxval.type(act_dtype)
                            )
                        else:
                            self.quant_model.act_scales_dict[name] = (
                                self.quant_model.act_scales_dict[name]
                                / maxval.type(act_dtype)
                            )
        if self.quant_model.quant_algo_dict["c_quant_algo"] == "fp8":
            for k, v in self.quant_model.kv_cache_scales_dict.items():
                self.quant_model.kv_cache_scales_dict[k] = v / maxval.type(v.dtype)


def _filter_func(name):
    pattern = re.compile(
        r".*(mlp_t5|pooler|style_embedder|x_embedder|t_embedder|extra_embedder).*"
    )
    return pattern.match(name) is not None


class DiTHook:
    def __init__(self, model):
        """
        Args:
            model(nn.Moudle, required): the model to be quant
        """
        self.model = model
        self.input_activation = []
        self.output_activation = []

        self._apply_hook()

    def _apply_hook(self):
        self._forward_hook_list = []
        for name, sub_layer in self.model.named_modules():
            if _filter_func(name):
                continue
            if isinstance(sub_layer, (torch.nn.Conv2d, torch.nn.Linear)):
                if "blocks" in name:
                    # handle
                    forward_pre_hook_handle = sub_layer.register_forward_hook(
                        self._forward_pre_hook
                    )
                    self._forward_hook_list.append(forward_pre_hook_handle)

    def _forward_pre_hook(self, layer, input, output):
        layer_name = ""
        for name, module in self.model.named_modules():
            if _filter_func(name):
                continue
            if module == layer:
                layer_name = name
                break
        x = (
            output[0].detach().cpu()
            if isinstance(output, tuple)
            else output.detach().cpu()
        )
        self.output_activation.append((layer_name, x))
        y = (
            input[0].detach().cpu()
            if isinstance(input, tuple)
            else input.detach().cpu()
        )
        self.input_activation.append((layer_name, y))

    def remove_hook(self):
        for hook in self._forward_hook_list:
            hook.remove()
        self._forward_hook_list = []

    def clean_acitvation_list(self):
        self.input_activation = []
        self.output_activation = []
