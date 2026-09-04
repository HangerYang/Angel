import argparse

parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument("--outdir", type=str, default="outdir0")
parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolVLM-256M-Instruct")
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer

modelname = args.model


def _content(text):
    # SmolVLM / Idefics3 chat template expects a list of typed content blocks.
    return [{"type": "text", "text": text}]


def build_dataset_rank(tokenizer, split="train", select=None):
    ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")
    ds = ds["train"]
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names

    def preprocess_function(examples):
        new_examples = {"conversation": [], "input_ids": [], "loss_mask": []}
        for i in range(len(examples["id"])):
            roles = {"human": "user", "gpt": "assistant"}
            convroles = ["user", "assistant"]
            source = examples["conversations"][i]
            if len(source) == 0 or roles.get(source[0]["from"]) != "user":
                source = source[1:]

            messages = []
            ok = True
            for j, sentence in enumerate(source):
                role = roles.get(sentence["from"])
                if role != convroles[j % 2]:
                    ok = False
                    break
                messages.append({"role": role, "content": _content(sentence["value"])})
            if not ok or len(messages) < 2:
                continue

            conversation = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
                add_special_tokens=False,
            ).input_ids[0]

            # Separator-agnostic loss mask: only assistant-generated tokens are
            # supervised. For each assistant turn we locate its token span by
            # re-tokenizing the prompt prefix (up to the generation prompt) and the
            # full prefix (including the assistant answer).
            loss_mask = torch.zeros_like(input_ids)
            for k in range(1, len(messages), 2):
                prompt_prefix = tokenizer.apply_chat_template(
                    messages[:k], tokenize=False, add_generation_prompt=True
                )
                full_prefix = tokenizer.apply_chat_template(
                    messages[: k + 1], tokenize=False, add_generation_prompt=False
                )
                start = len(
                    tokenizer(prompt_prefix, add_special_tokens=False).input_ids
                )
                end = len(tokenizer(full_prefix, add_special_tokens=False).input_ids)
                start = min(start, input_ids.shape[0])
                end = min(end, input_ids.shape[0])
                loss_mask[start:end] = 1

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        remove_columns=original_columns1,
        load_from_cache_file=False,
    )
    ds1.set_format(type="torch")
    return ds1


tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)
dataset = build_dataset_rank(tokenizer)
print(dataset)
model = AutoModelForImageTextToText.from_pretrained(
    modelname, device_map="auto", torch_dtype=torch.bfloat16
)
model.eval()


@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    outs_big = model(input_ids.cuda(), output_hidden_states=True)
    inputs_embeds_big = outs_big.hidden_states[0]
    hidden_state_big = outs_big.hidden_states[-1]
    td = {
        "inputs_embeds": inputs_embeds_big.cpu()[0],
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0],
    }
    return td


outdir = f"{args.outdir}/{args.index}"
if not os.path.exists(outdir):
    os.makedirs(outdir)


def writedata(name, data_point, idx):
    if not os.path.exists(name):
        os.makedirs(name)
    torch.save(data_point, f"{name}/data_{idx}.ckpt")


for i, data in enumerate(tqdm(dataset)):
    outdata = ge(data)
    writedata(outdir, outdata, i)
