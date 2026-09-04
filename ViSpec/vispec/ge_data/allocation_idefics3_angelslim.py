"""Shard `ge_data_all_idefics3_angelslim` across GPUs.

Same structure as ViSpec's other allocation scripts -- split the record range
into one contiguous slice per GPU and run a worker on each -- but it carries
the jsonl path, the image root and BOTH output dirs through to the workers.
"""

import argparse
import sys

import torch

parser = argparse.ArgumentParser(description="ViSpec data gen over an AngelSlim jsonl")
parser.add_argument("--outdir", type=str, default=None,
                    help="Output root for TEXT-ONLY records (ViSpec stage 2.1).")
parser.add_argument("--multimodal-outdir", type=str, default=None,
                    help="Output root for records WITH an image (ViSpec stage 2.2).")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=132942)
parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--gpus_per_model", type=int, default=1)
parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                    help="Physical GPU ids to use. Defaults to all visible GPUs.")
parser.add_argument("--datapath", type=str, required=True)
parser.add_argument("--image-root", type=str,
                    default="/home/hyang/Angel/dataset/raw/images")
args = parser.parse_args()

import os
from concurrent.futures import ThreadPoolExecutor

if args.outdir is None and args.multimodal_outdir is None:
    parser.error("give at least one of --outdir / --multimodal-outdir")

if args.gpu_ids:
    gpus = [
        args.gpu_ids[i : i + args.gpus_per_model]
        for i in range(0, len(args.gpu_ids), args.gpus_per_model)
    ]
else:
    gpus = [
        list(range(i, i + args.gpus_per_model))
        for i in range(0, torch.cuda.device_count(), args.gpus_per_model)
    ]
num_p = len(gpus)


def split_range(start, end, n, over=False):
    length = end - start + 1
    base_interval = length // n
    additional = length % n
    intervals = []
    previous = start
    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append((previous, previous + current_interval - 1))
        previous += current_interval
    return intervals


def run_command(cmd):
    os.system(cmd)


# The worker appends its own shard index, so every shard writes beside the
# others under one root rather than into a range-named tree.
suffix = f"idefics3_angelslim_{args.start}_{args.end}_mubf16"
text_outdir = os.path.join(args.outdir, suffix) if args.outdir else None
mm_outdir = (
    os.path.join(args.multimodal_outdir, suffix) if args.multimodal_outdir else None
)
for d in (text_outdir, mm_outdir):
    if d:
        os.makedirs(d, exist_ok=True)

data_a = split_range(args.start, args.end, num_p, over=True)
commands = []
for i in range(num_p):
    start, end = data_a[i]
    parts = [
        sys.executable,
        "-m vispec.ge_data.ge_data_all_idefics3_angelslim",
        f"--start={start}",
        f"--end={end}",
        f"--index={i}",
        "--gpu_index " + " ".join(map(str, gpus[i])),
        f"--model {args.model}",
        f"--temperature {args.temperature}",
        f"--max_new_tokens {args.max_new_tokens}",
        f"--datapath {args.datapath}",
        f"--image-root {args.image_root}",
    ]
    if text_outdir:
        parts.append(f"--outdir {text_outdir}")
    if mm_outdir:
        parts.append(f"--multimodal-outdir {mm_outdir}")
    commands.append(" ".join(parts))

with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
