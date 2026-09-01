"""Calculate HiViS and autoregressive baseline decoding speeds."""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "--model_path", dest="model_path", default="llava-hf/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--baseline-json", "--baseline_json", dest="baseline_json", required=True)
    parser.add_argument("--hivis-json", "--hivis_json", dest="hivis_json", required=True)
    parser.add_argument("--output-file", default="speedup.txt")
    return parser.parse_args()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def mean_hivis_speed(records):
    speeds = []
    for record in records:
        choice = record["choices"][0]
        tokens = sum(choice["new_tokens"])
        wall_time = sum(choice["wall_time"])
        speeds.append(tokens / wall_time)
    return float(np.asarray(speeds).mean()) if speeds else float("nan")


def mean_baseline_speed(records, tokenizer):
    speeds = []
    for record in records:
        choice = record["choices"][0]
        tokens = sum(len(tokenizer(turn).input_ids) - 1 for turn in choice["turns"])
        wall_time = sum(choice["wall_time"])
        speeds.append(tokens / wall_time)
    return float(np.asarray(speeds).mean()) if speeds else float("nan")


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    hivis_speed = mean_hivis_speed(read_jsonl(args.hivis_json))
    baseline_speed = mean_baseline_speed(read_jsonl(args.baseline_json), tokenizer)
    speedup_ratio = hivis_speed / baseline_speed

    print("hivis speed", hivis_speed)
    print("baseline speed", baseline_speed)
    print("speedup ratio:", speedup_ratio)

    hivis_path = Path(args.hivis_json)
    result_name = f"{hivis_path.parent.name}/{hivis_path.name}"
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(f"{result_name}\thivis_speed={hivis_speed:.6f}\tbaseline_speed={baseline_speed:.6f}\tratio={speedup_ratio:.6f}\n")


if __name__ == "__main__":
    main()
