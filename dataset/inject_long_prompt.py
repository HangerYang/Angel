#!/usr/bin/env python3
"""Inject ViSpec's vision-only long-prompt instruction into a mixed jsonl.

For every multimodal record (any content item with type=="image" anywhere in
its conversation), prepends a Vicuna-style system turn and appends " Please
answer with at least 1000 words." to the first user turn -- matching upstream
ViSpec's ge_data_all_llava_pretrain_gen.py prompt
(https://github.com/KangJialiang/ViSpec), verified verbatim from their source
this session. Text-only records pass through unchanged.

This only prepares the *input* prompts; it does not call any model. Feed the
output into scripts/speculative/smolvlm/generate_data_for_target_model.py to
get actual regenerated responses from a served target model.

Example:
  python dataset/inject_long_prompt.py \\
    --input dataset/preprocessed/mixed_sharegpt_llava665k_70k70k.jsonl \\
    --output dataset/preprocessed/mixed_sharegpt_llava665k_70k70k_long_prompt_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VISPEC_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
VISPEC_LENGTH_INSTRUCTION = " Please answer with at least 1000 words."


def is_multimodal(record: dict) -> bool:
    return any(
        isinstance(item, dict) and item.get("type") == "image"
        for turn in record.get("conversations", [])
        for item in turn.get("content", [])
    )


def inject_long_prompt(record: dict) -> dict:
    if not is_multimodal(record):
        return record

    conversations = record["conversations"]
    out_conversations = [
        {"role": "system", "content": [{"type": "text", "text": VISPEC_SYSTEM_PROMPT}]}
    ]
    first_user_seen = False
    for turn in conversations:
        content = list(turn["content"])
        if turn["role"] == "user" and not first_user_seen:
            content = content + [{"type": "text", "text": VISPEC_LENGTH_INSTRUCTION}]
            first_user_seen = True
        out_conversations.append({"role": turn["role"], "content": content})

    return {**record, "conversations": out_conversations}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = n_mm = 0
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            n += 1
            out_record = inject_long_prompt(record)
            if out_record is not record:
                n_mm += 1
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
    print(f"rows={n} multimodal_prompted={n_mm} output={args.output}")


if __name__ == "__main__":
    main()
