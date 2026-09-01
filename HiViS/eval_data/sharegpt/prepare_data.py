import argparse
import json
import re
from pathlib import Path


RE_CHAR_REPEAT = re.compile(r"([^\s])\1{9,}", re.UNICODE)
RE_CHUNK_REPEAT = re.compile(r"([^\s]{1,4})\1{5,}", re.UNICODE)
RE_TOKEN_REPEAT = re.compile(r"(?<!\w)([^\s]{1,4})([\s,;:_\\/\-|]{1,3})\1(?:\2\1){4,}(?!\w)", re.UNICODE)
CODE_KEYWORDS = ("def ", "class ", "import ", "return", "#include", "public ", "private ", "func ", "let ", "var ")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean ShareGPT data and mark samples excluded from second-stage training.")
    parser.add_argument("--input", type=Path, required=True, help="Input ShareGPT JSON or JSONL file.")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("sharegpt.jsonl"))
    return parser.parse_args()


def iter_records(path):
    with path.open("r", encoding="utf-8") as file:
        first_character = file.read(1)
        file.seek(0)
        if first_character == "[":
            yield from json.load(file)
        else:
            for line in file:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def has_bad_repetition(text):
    return bool(text and (RE_CHAR_REPEAT.search(text) or RE_CHUNK_REPEAT.search(text) or RE_TOKEN_REPEAT.search(text)))


def is_code_heavy(text, code_ratio_threshold=0.3, symbol_ratio_threshold=0.1):
    if not text:
        return False
    code_characters = sum(len(block) for block in re.findall(r"```.*?```", text, flags=re.DOTALL))
    symbol_count = len(re.findall(r"[@{}();<>/=+\-_*]", text))
    keyword_count = sum(keyword in text for keyword in CODE_KEYWORDS)
    return code_characters / len(text) >= code_ratio_threshold or symbol_count / len(text) >= symbol_ratio_threshold or keyword_count > 2


def prepare_record(record):
    if not isinstance(record, dict) or not isinstance(record.get("conversations"), list) or not record["conversations"]:
        return None
    gpt_values = [turn.get("value", "") for turn in record["conversations"] if isinstance(turn, dict) and turn.get("from") == "gpt" and isinstance(turn.get("value"), str)]
    if any(has_bad_repetition(value) for value in gpt_values):
        return None
    has_markdown = any(isinstance(turn, dict) and "markdown" in turn for turn in record["conversations"])
    has_image = any(isinstance(turn, dict) and "<image>" in str(turn.get("value", "")).lower() for turn in record["conversations"])
    if has_markdown or has_image:
        return None
    record = dict(record)
    record["is_code"] = is_code_heavy(" ".join(gpt_values))
    return record


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = kept = code = 0
    with args.output.open("w", encoding="utf-8") as output_file:
        for record in iter_records(args.input):
            total += 1
            record = prepare_record(record)
            if record is None:
                continue
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            code += int(record["is_code"])
    print(f"Processed {total} records, kept {kept}, code {code}, non-code {kept - code}. Output: {args.output}")


if __name__ == "__main__":
    main()
