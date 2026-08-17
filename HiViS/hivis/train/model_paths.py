"""Resolve local model directories and Hugging Face repository IDs for training."""

import json
from pathlib import Path

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, cached_file


_LM_HEAD_KEYS = ("language_model.lm_head.weight", "lm_head.weight")
_EMBED_KEYS = (
    "language_model.model.embed_tokens.weight",
    "model.text_model.embed_tokens.weight",
    "model.embed_tokens.weight",
)


def is_qwen_model(config):
    identifiers = [getattr(config, "model_type", ""), *(getattr(config, "architectures", None) or [])]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        identifiers.extend([getattr(text_config, "model_type", ""), *(getattr(text_config, "architectures", None) or [])])
    return any("qwen" in str(identifier).lower() for identifier in identifiers)


def resolve_base_model_path(model_path):
    local_path = Path(model_path).expanduser()
    if local_path.is_dir():
        return str(local_path)

    # Try sharded model (has index JSON).
    try:
        index_path = Path(cached_file(model_path, SAFE_WEIGHTS_INDEX_NAME))
        with index_path.open() as file:
            weight_map = json.load(file)["weight_map"]
        for candidate_keys in (_LM_HEAD_KEYS, _EMBED_KEYS):
            weight_key = next((key for key in candidate_keys if key in weight_map), None)
            if weight_key is None:
                raise KeyError(f"None of the expected weight names {candidate_keys} exists in {model_path}/{SAFE_WEIGHTS_INDEX_NAME}.")
            cached_file(model_path, weight_map[weight_key])
        return str(index_path.parent)
    except Exception:
        pass

    # Fall back to single-file model (model.safetensors with no index).
    single = cached_file(model_path, "model.safetensors")
    return str(Path(single).parent)
