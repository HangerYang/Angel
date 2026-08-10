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

from .draft_model_factory import DraftModelConfig, create_draft_model
from .llama_eagle3 import (
    CosyVoice3Eagle3LlamaForCausalLM,
    Eagle3LlamaForCausalLM,
    HAWK_FUSE_MODES,
    REAL_HAWK_MODES,
)
from .lora_utils import (
    apply_layer_skip_lora_training_setup,
    apply_real_hawk_training_setup,
    merge_lora_into_state_dict,
)
from .qwen_dflare import QwenDFlareDraftModel
from .qwen_dflash import QwenDFlashDraftModel

__all__ = [
    "create_draft_model",
    "DraftModelConfig",
    "Eagle3LlamaForCausalLM",
    "CosyVoice3Eagle3LlamaForCausalLM",
    "QwenDFlashDraftModel",
    "QwenDFlareDraftModel",
    "HAWK_FUSE_MODES",
    "REAL_HAWK_MODES",
    "apply_real_hawk_training_setup",
    "apply_layer_skip_lora_training_setup",
    "merge_lora_into_state_dict",
]
