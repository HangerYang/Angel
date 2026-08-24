"""Split HiViS data generation across GPUs and launch one process per GPU."""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .model_names import DEFAULT_MODEL_PATHS, model_directory_name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llava", "qwen"), default="llava")
    parser.add_argument("--data-type", choices=("text", "multimodal"), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=68000, help="Exclusive end index.")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.end < args.start:
        parser.error("require 0 <= --start <= --end")
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATHS[args.model]
    if args.outdir is None:
        dataset_name = "sharegpt" if args.data_type == "text" else "llava_v1_5_mix665k"
        args.outdir = Path("eval_data/generated") / model_directory_name(args.model_path) / dataset_name
    return args


def split_range(start, end, number_of_parts):
    length = end - start
    base_size, remainder = divmod(length, number_of_parts)
    ranges = []
    current = start
    for part_index in range(number_of_parts):
        part_size = base_size + (part_index < remainder)
        ranges.append((current, current + part_size))
        current += part_size
    return ranges


def build_command(args, worker_index, gpu, start, end):
    module = f"hivis.ge_data.ge_data_{args.model}"
    command = [
        sys.executable,
        "-m",
        module,
        "--data-type",
        args.data_type,
        "--start",
        str(start),
        "--end",
        str(end),
        "--index",
        str(worker_index),
        "--gpu-index",
        str(gpu),
        "--outdir",
        str(args.outdir),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
    ]
    for option, value in (
        ("--model-path", args.model_path),
        ("--data-file", args.data_file),
        ("--image-root", args.image_root),
        ("--max-length", args.max_length),
    ):
        if value is not None:
            command.extend((option, str(value)))
    if args.fail_fast:
        command.append("--fail-fast")
    return command


def run(command):
    return subprocess.run(command, check=False).returncode


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    ranges = split_range(args.start, args.end, len(args.gpus))
    commands = [
        build_command(args, index, gpu, start, end)
        for index, (gpu, (start, end)) in enumerate(zip(args.gpus, ranges))
        if start < end
    ]
    for command in commands:
        print(" ".join(command), flush=True)
    if args.dry_run or not commands:
        return

    failures = []
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = {executor.submit(run, command): command for command in commands}
        for future in as_completed(futures):
            return_code = future.result()
            if return_code:
                failures.append((return_code, futures[future]))
    if failures:
        summary = "\n".join(
            f"exit {return_code}: {' '.join(command)}"
            for return_code, command in failures
        )
        raise SystemExit(f"{len(failures)} data-generation worker(s) failed:\n{summary}")


if __name__ == "__main__":
    main()
