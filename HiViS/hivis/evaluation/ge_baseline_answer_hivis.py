"""Run autoregressive (non-speculative) baseline decoding for HiViS models."""

import argparse
import json
import os
import time

import shortuuid
import torch
from tqdm import tqdm

from ..model.model_hivis import EaModel
from .benchmark_data import SAMPLE_COUNT, load_benchmark, prepare_inputs
from .ge_hivis_answer import (
    _dataset_slice,
    infer_model_id,
    reorg_answer_file,
    setup_seed,
    truncate_list,
)


@torch.inference_mode()
def generate_baseline_answers(model, dataset, answer_file, args):
    tokenizer = model.get_tokenizer()
    model.eval()
    print("Check model training state:", model.training)
    print("CUDA VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    for _ in range(args.warmup_steps):
        torch.manual_seed(0)
        inputs = prepare_inputs(model, dataset, 0, args.dataset, truncation=True)
        model.naivegenerate(
            inputs,
            temperature=args.temperature,
            max_new_tokens=args.max_new_token,
            is_llama3=args.model_type == "llama-3-instruct",
            log=True,
        )
    print("Warmup done")

    answer_path = os.path.expanduser(answer_file)
    answer_dir = os.path.dirname(answer_path)
    if answer_dir:
        os.makedirs(answer_dir, exist_ok=True)

    for question_id in tqdm(range(len(dataset))):
        inputs = prepare_inputs(
            model, dataset, question_id, args.dataset, truncation=False
        )
        input_length = inputs.input_ids.shape[1]

        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token, idx, decode_setup_time = model.naivegenerate(
            inputs,
            temperature=args.temperature,
            max_new_tokens=args.max_new_token,
            is_llama3=args.model_type == "llama-3-instruct",
            log=True,
        )
        torch.cuda.synchronize()
        wall_time = time.time() - start_time

        decode_ids = output_ids[0][input_length:].tolist()
        decode_ids = truncate_list(decode_ids, tokenizer.eos_token_id)
        if args.model_type == "llama-3-instruct":
            decode_ids = truncate_list(
                decode_ids, tokenizer.convert_tokens_to_ids("<|eot_id|>")
            )
        text = tokenizer.decode(
            decode_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        choice = {
            "index": question_id,
            "turns": [text],
            "idxs": [int(idx)],
            "new_tokens": [int(new_token)],
            "wall_time": [wall_time],
            "decode_time": [float(decode_setup_time)],
        }
        answer = {
            "question_id": question_id,
            "answer_id": shortuuid.uuid(),
            "model_id": args.model_id,
            "choices": [choice],
            "tstamp": time.time(),
        }
        with open(answer_path, "a", encoding="utf-8") as output:
            output.write(json.dumps(answer) + "\n")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-model-path", required=True)
    parser.add_argument("--base-model-path", required=True)
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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=3)
    return parser


def main():
    args = build_parser().parse_args()
    args.model_id = args.model_id or infer_model_id(args.base_model_path)
    args.model_id = f"{args.model_id}-temperature-{args.temperature}-baseline"
    setup_seed(args.seed)

    dataset = _dataset_slice(
        load_benchmark(args.dataset, SAMPLE_COUNT),
        args.question_begin,
        args.question_end,
    )
    if len(dataset) == 0:
        raise ValueError("The selected benchmark range is empty")

    answer_file = args.answer_file or f"{args.dataset}/{args.model_id}.jsonl"
    print(f"Output to {answer_file}")
    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    generate_baseline_answers(model, dataset, answer_file, args)
    reorg_answer_file(answer_file)


if __name__ == "__main__":
    main()
