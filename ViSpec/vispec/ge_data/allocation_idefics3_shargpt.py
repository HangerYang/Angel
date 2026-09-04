import argparse
import copy
import sys

import torch

parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--outdir", type=str, default="0")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=67999)
parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--gpus_per_model", type=int, default=1)
parser.add_argument(
    "--gpu_ids",
    type=int,
    nargs="+",
    default=None,
    help="Physical GPU ids to use (e.g. 2 3 4 5 6 7). Defaults to all visible GPUs.",
)
args = parser.parse_args()

import os
from concurrent.futures import ThreadPoolExecutor

s = args.start
e = args.end

if args.gpu_ids:
    gpus = [
        args.gpu_ids[i : i + args.gpus_per_model]
        for i in range(0, len(args.gpu_ids), args.gpus_per_model)
    ]
else:
    num_p = torch.cuda.device_count()
    gpus = [
        [j for j in range(i, i + args.gpus_per_model)]
        for i in range(0, num_p, args.gpus_per_model)
    ]
num_p = len(gpus)


outdir = "{}/idefics3_shargpt_{}_{}_mubf16".format(args.outdir, s, e)


def split_range(start, end, n, over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n  # Get the remainder of the division
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


if not os.path.exists(outdir):
    os.makedirs(outdir)


data_a = split_range(s, e, num_p, over=True)
commands = []
for i in range(num_p):
    index = i
    start = data_a[i][0]
    end = data_a[i][1]
    gpu_index = gpus[i]
    gpu_index_str = " ".join(map(str, gpu_index))
    command = "{} -m vispec.ge_data.ge_data_all_idefics3_shargpt --start={} --end={} --index={} --gpu_index {} --outdir {} --model {}".format(
        sys.executable, start, end, index, gpu_index_str, outdir, args.model
    )
    commands.append(command)

with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
