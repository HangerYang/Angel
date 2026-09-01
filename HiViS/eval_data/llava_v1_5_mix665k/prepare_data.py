import argparse
from contextlib import contextmanager
import io
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_INPUT = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json"
DEFAULT_OUTPUT = Path(__file__).with_name("llava_v1_5_mix665k_long_context.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean and filter the LLaVA-v1.5 mix665k training data."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Hugging Face file URL or a local JSON/JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path of the filtered JSONL file.",
    )
    return parser.parse_args()


def normalize_input_path(input_path):
    """Turn a Hugging Face file-view URL into its downloadable URL."""
    parsed = urlparse(input_path)
    if parsed.netloc == "huggingface.co" and "/blob/" in parsed.path:
        return input_path.replace("/blob/", "/resolve/", 1)
    return input_path


@contextmanager
def open_input(input_path):
    parsed = urlparse(input_path)
    if parsed.scheme in {"http", "https"}:
        request = Request(input_path, headers={"User-Agent": "HiViS-data-preparation"})
        response = urlopen(request)
        stream = io.TextIOWrapper(response, encoding="utf-8")
        try:
            yield stream
        finally:
            stream.close()
    else:
        with Path(input_path).open("r", encoding="utf-8") as stream:
            yield stream


def iter_json_array(stream, chunk_size=64 * 1024):
    decoder = json.JSONDecoder()
    buffer = ""

    while True:
        buffer = buffer.lstrip()
        if not buffer:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError("Unexpected end of file while reading a JSON array.")
            buffer = chunk
            continue

        if buffer[0] == "]":
            return
        if buffer[0] == ",":
            buffer = buffer[1:]
            continue

        try:
            item, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise
            buffer += chunk
            continue

        yield item
        buffer = buffer[end:]


def iter_records(stream):
    first_character = stream.read(1)
    while first_character and first_character.isspace():
        first_character = stream.read(1)

    if not first_character:
        return
    if first_character == "[":
        yield from iter_json_array(stream)
        return

    first_line = first_character + stream.readline()
    if first_line.strip():
        yield json.loads(first_line)
    for line in stream:
        if line.strip():
            yield json.loads(line)


def clean_item(item):
    if not isinstance(item, dict):
        return None

    item = dict(item)
    if "id" in item:
        item["id"] = str(item["id"])
    item.pop("model", None)

    conversations = []
    for turn in item.get("conversations", []):
        if isinstance(turn, dict) and "from" in turn and "value" in turn:
            conversations.append({"from": turn["from"], "value": turn["value"]})
    item["conversations"] = conversations
    return item


def should_keep(item):
    if not item.get("image"):
        return False

    gpt_word_counts = [
        len(turn["value"].strip().split())
        for turn in item["conversations"]
        if turn["from"] == "gpt"
    ]
    return bool(gpt_word_counts) and all(count >= 5 for count in gpt_word_counts)


def main():
    args = parse_args()
    input_path = normalize_input_path(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_count = 0
    kept_count = 0
    with open_input(input_path) as input_file:
        with args.output.open("w", encoding="utf-8") as output_file:
            for raw_item in iter_records(input_file):
                total_count += 1
                item = clean_item(raw_item)
                if item is None or not should_keep(item):
                    continue
                output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept_count += 1

    print(f"Processed {total_count} records and kept {kept_count}. Output: {args.output}")


if __name__ == "__main__":
    main()
