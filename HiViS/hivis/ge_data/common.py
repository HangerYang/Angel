"""Shared hidden-state-generation helpers for HiViS's per-model ge_data_*.py scripts.

Kept model-agnostic on purpose: model loading, vision-token handling, and
chat markers differ enough between Qwen2.5-VL, SmolVLM, and LLaVA that
folding them in here would just move the duplication around rather than
remove it. What's actually identical across models lives here instead, so
each ge_data_<model>.py only carries its own model-specific bits.
"""

import base64
import io
import json
import re

import torch
from datasets import Dataset, Features, Value
from PIL import Image

# Text-only and multimodal records have different "content" shapes.
RECORD_FEATURES = Features(
    {
        "id": Value("string"),
        "conversations": [
            {
                "role": Value("string"),
                "content": [
                    {
                        "type": Value("string"),
                        "text": Value("string"),
                        "image": Value("string"),
                    }
                ],
            }
        ],
    }
)


def load_record_dataset(data_file):
    """Load a role/content JSONL file into a Dataset, row by row.

    `datasets.load_dataset("json", ..., features=RECORD_FEATURES)` builds its
    arrow table in ~10MB chunks and infers each chunk's arrow type from the
    records actually in it, then casts every chunk to RECORD_FEATURES. A
    one-pass input file built by concatenating a long pure-text block with a
    long multimodal block (as this data-generation step's own combined input
    does -- see ge_data_qwen.py/ge_data_smolvlm.py) puts thousands of
    consecutive rows that never contain an "image" key back to back; any
    chunk entirely inside that run gets the narrower struct<type,text>, and
    pyarrow cannot cast that into RECORD_FEATURES's struct<type,text,image>
    once it's nested inside a list column ("Couldn't cast array of type
    struct<type,text> to {...,image,...}") -- this reproduces regardless of
    chunksize, since no chunk size is both small enough to avoid pyarrow's
    ~2GB offset overflow and large enough to guarantee straddling both runs.
    `Dataset.from_generator` sidesteps this by encoding each row against
    RECORD_FEATURES individually as it's read, filling a row's missing
    "image" key with null per-row instead of casting a whole pre-built chunk.
    """

    def _rows():
        with open(data_file) as handle:
            for line in handle:
                yield json.loads(line)

    return Dataset.from_generator(_rows, features=RECORD_FEATURES)

# Verbatim from eval_data/sharegpt/prepare_data.py's is_code_heavy(), so a
# text record lands in the same code/non_code bucket HiViS's own stock
# ShareGPT pipeline would put it in.
CODE_KEYWORDS = ("def ", "class ", "import ", "return", "#include", "public ", "private ", "func ", "let ", "var ")


def is_code_heavy(text, code_ratio_threshold=0.3, symbol_ratio_threshold=0.1):
    if not text:
        return False
    code_characters = sum(len(block) for block in re.findall(r"```.*?```", text, flags=re.DOTALL))
    symbol_count = len(re.findall(r"[@{}();<>/=+\-_*]", text))
    keyword_count = sum(keyword in text for keyword in CODE_KEYWORDS)
    return (
        code_characters / len(text) >= code_ratio_threshold
        or symbol_count / len(text) >= symbol_ratio_threshold
        or keyword_count > 2
    )


def record_has_image(record):
    return any(
        entry.get("type") == "image"
        for turn in record["conversations"]
        for entry in turn.get("content", [])
    )


def assistant_text(record):
    """Join assistant-turn text, matching prepare_data.py's ' '.join(gpt_values)."""
    parts = []
    for turn in record["conversations"]:
        if turn.get("role") != "assistant":
            continue
        parts.append(
            " ".join(e.get("text", "") for e in turn.get("content", []) if e.get("type") == "text")
        )
    return " ".join(parts)


def load_image_field(img_ref):
    """Load a PIL image from either a base64 data URI or a file path."""
    if img_ref.startswith("data:image/"):
        _, _, b64_data = img_ref.partition(",")
        with Image.open(io.BytesIO(base64.b64decode(b64_data))) as opened:
            return opened.convert("RGB").copy()
    with Image.open(img_ref) as opened:
        return opened.convert("RGB").copy()


def record_to_messages_and_image(record):
    """Convert role/content format to HF messages list and load the first image."""
    messages = []
    image = None
    for turn in record["conversations"]:
        role = turn["role"]  # "user" or "assistant"
        new_content = []
        for entry in turn.get("content", []):
            entry_type = entry.get("type")
            if entry_type == "image":
                img_path = entry.get("image", "")
                if img_path and image is None:
                    try:
                        image = load_image_field(img_path)
                    except Exception as exc:
                        print(f"Could not load image {img_path[:80]}: {exc}")
                new_content.append({"type": "image"})
            elif entry_type == "text":
                new_content.append({"type": "text", "text": entry.get("text", "")})
        messages.append({"role": role, "content": new_content})
    return messages, image


def build_loss_mask(input_ids, assistant_tokens, end_tokens):
    """Mark each assistant turn's token span as 1, everything else 0."""
    loss_mask = torch.zeros_like(input_ids)
    assistant_length = len(assistant_tokens)
    end_length = len(end_tokens)
    for batch_index, tokens in enumerate(input_ids.cpu()):
        assistant_start = None
        token_index = 0
        while token_index < tokens.numel():
            if (
                assistant_start is None
                and token_index <= tokens.numel() - assistant_length
                and tokens[token_index : token_index + assistant_length].tolist() == assistant_tokens
            ):
                assistant_start = token_index + assistant_length
                token_index += assistant_length
                continue
            if (
                assistant_start is not None
                and token_index <= tokens.numel() - end_length
                and tokens[token_index : token_index + end_length].tolist() == end_tokens
            ):
                loss_mask[batch_index, assistant_start:token_index] = 1
                assistant_start = None
                token_index += end_length
                continue
            token_index += 1
        if assistant_start is not None:
            end = -assistant_length if assistant_length else None
            loss_mask[batch_index, assistant_start:end] = 1
    return loss_mask


def save_sample(output_dir, sample):
    sample_index = len(list(output_dir.glob("data_*.ckpt")))
    torch.save(sample, output_dir / f"data_{sample_index}.ckpt")
