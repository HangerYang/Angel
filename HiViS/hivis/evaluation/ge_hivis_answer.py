"""Run EAGLE, HiViS, or ViSpec decoding on a supported benchmark."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import shortuuid
import torch
from tqdm import tqdm

from ..model.model_hivis import EaModel
from .benchmark_data import SAMPLE_COUNT, load_benchmark, prepare_inputs


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def truncate_list(values, stop_token_id):
    if stop_token_id not in values:
        return values
    return values[: values.index(stop_token_id) + 1]


def infer_model_id(base_model_path):
    """Build a readable output id from a local snapshot path or repo id."""
    for part in Path(base_model_path).parts:
        if part.startswith("models--"):
            return part.split("--")[-1]
    return Path(base_model_path.rstrip("/")).name


def _dataset_slice(dataset, begin, end):
    """Apply the optional debug range to Dataset, DatasetDict, or a list."""
    start = 0 if begin is None else begin
    stop = len(dataset) if end is None else min(end, len(dataset))
    if start == 0 and stop == len(dataset):
        return dataset
    if hasattr(dataset, "select"):
        return dataset.select(range(start, stop))
    return dataset[start:stop]


def run_eval(
    base_model_path,
    ea_model_path,
    model_id,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    args,
):
    del num_choices, max_gpu_memory
    assert num_gpus_total % num_gpus_per_model == 0
    worker_count = num_gpus_total // num_gpus_per_model
    dataset = _dataset_slice(
        load_benchmark(args.dataset, args.candidate_count),
        question_begin,
        question_end,
    )

    call_args = (
        base_model_path,
        ea_model_path,
        model_id,
        dataset,
        answer_file,
        max_new_token,
        temperature,
        args,
    )
    if worker_count == 1:
        get_model_answers(*call_args)
        return

    # Kept for compatibility with the original multi-GPU CLI. Each worker gets
    # its own slice and temporary answer file to avoid concurrent JSONL writes.
    import ray

    remote_eval = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    jobs = []
    part_files = []
    for worker_index in range(worker_count):
        indices = range(worker_index, len(dataset), worker_count)
        part = dataset.select(indices) if hasattr(dataset, "select") else [dataset[i] for i in indices]
        part_file = f"{answer_file}.part-{worker_index}"
        part_files.append(part_file)
        jobs.append(
            remote_eval(
                base_model_path,
                ea_model_path,
                model_id,
                part,
                part_file,
                max_new_token,
                temperature,
                args,
            )
        )
    ray.get(jobs)
    with open(answer_file, "w", encoding="utf-8") as output:
        for part_file in part_files:
            with open(part_file, encoding="utf-8") as part:
                output.writelines(part)
            os.remove(part_file)


@torch.inference_mode()
def get_model_answers(
    base_model_path,
    ea_model_path,
    model_id,
    dataset,
    answer_file,
    max_new_token,
    temperature,
    args,
):
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        draft_method=args.draft_method,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = model.get_tokenizer()
    model.eval()
    print("Check model training state:", model.training)
    print("CUDA VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    for warmup_index in range(min(3, len(dataset))):
        torch.manual_seed(0)
        inputs = prepare_inputs(
            model,
            dataset,
            warmup_index,
            args.dataset,
            truncation=True,
        )
        model.eagenerate(
            inputs,
            temperature=temperature,
            max_new_tokens=max_new_token,
            is_llama3=args.model_type == "llama-3-instruct",
            log=True,
        )
    print("Warmup done")

    answer_dir = os.path.dirname(os.path.expanduser(answer_file))
    if answer_dir:
        os.makedirs(answer_dir, exist_ok=True)

    all_accept_lengths = []
    completed = 0
    for question_id in tqdm(range(len(dataset))):
        if completed >= args.target_samples:
            break
        try:
            inputs = prepare_inputs(
                model, dataset, question_id, args.dataset, truncation=False
            )
            input_ids = torch.as_tensor(inputs.input_ids).cuda()
            if input_ids.shape[1] > args.max_input_tokens:
                print(
                    f"Skipping candidate {question_id}: input length "
                    f"{input_ids.shape[1]} exceeds {args.max_input_tokens}",
                    flush=True,
                )
                continue

            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, new_token, idx, accept_lengths = model.eagenerate(
                inputs,
                temperature=temperature,
                max_new_tokens=max_new_token,
                is_llama3=args.model_type == "llama-3-instruct",
                log=True,
            )
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
        except (RuntimeError, ValueError) as error:
            if not args.skip_errors:
                raise
            print(
                f"Skipping candidate {question_id}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            torch.cuda.empty_cache()
            continue

        output_ids = output_ids[0][len(input_ids[0]) :]
        output_ids[output_ids > tokenizer.vocab_size] = 0
        all_accept_lengths.extend(accept_lengths)

        decode_ids = truncate_list(output_ids.tolist(), tokenizer.eos_token_id)
        if args.model_type == "llama-3-instruct":
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            decode_ids = truncate_list(decode_ids, eot_id)
        text = tokenizer.decode(
            decode_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        answer = {
            "question_id": question_id,
            "answer_id": shortuuid.uuid(),
            "model_id": model_id,
            "choices": [
                {
                    "index": question_id,
                    "turns": [text],
                    "idxs": [int(idx)],
                    "new_tokens": [int(new_token)],
                    "wall_time": [elapsed],
                    "accept_length": accept_lengths,
                }
            ],
            "tstamp": time.time(),
        }
        with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as output:
            output.write(json.dumps(answer) + "\n")
        completed += 1

    if completed < args.target_samples:
        raise RuntimeError(
            f"Only generated {completed} successful answers from {len(dataset)} "
            f"candidates; target was {args.target_samples}"
        )

    mean_accept_length = sum(all_accept_lengths) / len(all_accept_lengths)
    print(mean_accept_length)
    with open("results.txt", "a", encoding="utf-8") as output:
        output.write(f"{answer_file}: {mean_accept_length}\n")


def reorg_answer_file(answer_file):
    """Sort answers by question id and remove duplicate records."""
    answers = {}
    with open(answer_file, encoding="utf-8") as input_file:
        for line in input_file:
            answers[json.loads(line)["question_id"]] = line
    with open(answer_file, "w", encoding="utf-8") as output_file:
        for question_id in sorted(answers):
            output_file.write(answers[question_id])


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draft-method",
        choices=["eagle", "hivis", "vispec"],
        default="hivis",
        help="Draft model implementation used for speculative decoding.",
    )
    parser.add_argument("--ea-model-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--load-in-8bit", action="store_false")
    parser.add_argument(
        "--model-type",
        default="vicuna",
        choices=["llama-2-chat", "vicuna", "mixtral", "llama-3-instruct"],
    )
    parser.add_argument("--model-id")
    parser.add_argument("--dataset", default="ChartQA")
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file")
    parser.add_argument("--max-new-token", type=int, default=500)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--target-samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--candidate-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--max-input-tokens", type=int, default=4000)
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue to later candidates when an individual sample fails.",
    )
    parser.add_argument("--tree-choices", default="mc_sim_7b_63")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--testing", default="try")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.model_id = args.model_id or infer_model_id(args.base_model_path)
    setup_seed(args.seed)

    args.model_id = f"{args.model_id}-temperature-{args.temperature}-{args.testing}-depth-{args.depth}"
    if args.draft_method != "hivis":
        args.model_id = f"{args.model_id}-{args.draft_method}"
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    answer_file = args.answer_file or f"{args.dataset}/{args.model_id}.jsonl"
    print(f"Output to {answer_file}")
    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args,
    )
    reorg_answer_file(answer_file)


if __name__ == "__main__":
    main()
