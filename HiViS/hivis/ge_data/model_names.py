"""Map supported base models to stable data-directory names."""

from transformers import AutoConfig


DEFAULT_MODEL_PATHS = {"llava": "llava-hf/llava-v1.6-vicuna-7b-hf", "qwen": "Qwen/Qwen2.5-VL-7B-Instruct"}


def _name_from_reference(model_reference):
    name = str(model_reference).lower().replace("_", "-")
    if "qwen2.5-vl" in name or "qwen2-5-vl" in name or "qwen25vl" in name:
        return "qwen25vl_7b"
    size = "13b" if "13b" in name else "7b" if "7b" in name else None
    if size and ("llava-v1.6" in name or "llava1.6" in name or "llava16" in name):
        return f"llava16_{size}"
    if size and ("llava-1.5" in name or "llava-v1.5" in name or "llava1.5" in name or "llava15" in name):
        return f"llava15_{size}"
    return None


def model_directory_name(model_reference, config=None):
    reference_name = _name_from_reference(model_reference)
    if config is None and "/" not in str(model_reference) and reference_name is not None:
        return reference_name
    if config is None:
        try:
            config = AutoConfig.from_pretrained(model_reference)
        except (OSError, ValueError):
            if reference_name is not None:
                return reference_name
            raise

    model_type = str(getattr(config, "model_type", "")).lower()
    text_config = getattr(config, "text_config", None) or config
    hidden_size = getattr(text_config, "hidden_size", getattr(config, "hidden_size", None))
    if "qwen2_5_vl" in model_type or "qwen2.5_vl" in model_type:
        if hidden_size != 3584:
            raise ValueError(f"Unsupported Qwen2.5-VL hidden size: {hidden_size}")
        return "qwen25vl_7b"
    if model_type == "idefics3":
        size_tag = {576: "256m", 2048: "2b"}.get(hidden_size, f"{hidden_size}d")
        return f"smolvlm_{size_tag}"
    size = {4096: "7b", 5120: "13b"}.get(hidden_size)
    if model_type == "llava_next" and size is not None:
        return f"llava16_{size}"
    if model_type == "llava" and size is not None:
        return f"llava15_{size}"
    if reference_name is not None:
        return reference_name
    raise ValueError(f"Unsupported model type and hidden size: model_type={model_type}, hidden_size={hidden_size}")
