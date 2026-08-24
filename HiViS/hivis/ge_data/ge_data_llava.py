"""Generate text or multimodal hidden-state data for LLaVA HiViS training."""

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_PATH = "llava-hf/llava-v1.6-vicuna-7b-hf"
DEFAULT_TEXT_DATA = "eval_data/sharegpt/sharegpt.jsonl"
DEFAULT_MULTIMODAL_DATA = "eval_data/llava_v1_5_mix665k/llava_v1_5_mix665k_long_context.jsonl"
DEFAULT_IMAGE_ROOT = "eval_data/llava_v1_5_mix665k/images"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LLaVA hidden-state data for HiViS training."
    )
    parser.add_argument(
        "--data-type",
        choices=("text", "multimodal"),
        required=True,
        help="Generate pure-text ShareGPT data or multimodal LLaVA data.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--end",
        type=int,
        default=68000,
        help="Exclusive end index after deterministic dataset shuffling.",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--gpu-index",
        "--gpu_index",
        dest="gpu_index",
        type=int,
        nargs="+",
        default=[0],
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--data-file",
        default=None,
        help="JSON/JSONL input. Defaults depend on --data-type.",
    )
    parser.add_argument("--image-root", type=Path, default=Path(DEFAULT_IMAGE_ROOT))
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Defaults to 2048 for text and 4096 for multimodal data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first bad sample instead of logging and skipping it.",
    )
    args = parser.parse_args()
    if args.start < 0 or args.end < args.start:
        parser.error("require 0 <= --start <= --end")
    if args.max_length is None:
        args.max_length = 2048 if args.data_type == "text" else 4096
    if args.data_file is None:
        args.data_file = (
            DEFAULT_TEXT_DATA
            if args.data_type == "text"
            else DEFAULT_MULTIMODAL_DATA
        )
    return args


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))

# CUDA visibility must be configured before importing torch/transformers.
import torch
from datasets import load_dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)

from .model_names import model_directory_name


def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model_classes = {
        "llava": LlavaForConditionalGeneration,
        "llava_next": LlavaNextForConditionalGeneration,
    }
    try:
        model_class = model_classes[config.model_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model type {config.model_type!r}; expected LLaVA or LLaVA-NeXT."
        ) from exc
    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, config, AutoProcessor.from_pretrained(model_path)


def conversation_to_messages(conversation, include_image):
    messages = []
    image_added = False
    for message in conversation:
        role = "user" if message.get("from") == "human" else "assistant"
        value = message.get("value", "")
        content = []
        if role == "user" and include_image and "<image>" in value:
            parts = value.split("<image>")
            for part_index, part in enumerate(parts):
                text = part.strip()
                if text:
                    content.append({"type": "text", "text": text})
                if part_index < len(parts) - 1 and not image_added:
                    content.append({"type": "image"})
                    image_added = True
        else:
            content.append({"type": "text", "text": value})
        messages.append({"role": role, "content": content})

    # Some records have an image path but omit the literal <image> marker.
    if include_image and not image_added:
        for message in messages:
            if message["role"] == "user":
                message["content"].insert(0, {"type": "image"})
                image_added = True
                break
    return messages


class ConversationDataset(Dataset):
    def __init__(self, dataset, data_type, image_root):
        self.dataset = dataset
        self.multimodal = data_type == "multimodal"
        self.image_root = image_root

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        record = self.dataset[index]
        image = None
        if self.multimodal:
            image_path = record.get("image")
            if not image_path:
                raise ValueError(f"Multimodal sample {index} has no image path")
            full_path = self.image_root / image_path
            try:
                with Image.open(full_path) as opened_image:
                    image = opened_image.convert("RGB")
            except Exception as error:
                # Let generate_sample raise inside the per-sample error handler.
                image = None
                print(f"Could not load image {full_path}: {error}")
        messages = conversation_to_messages(
            record["conversations"], include_image=self.multimodal
        )
        return image, messages, bool(record.get("is_code", False))


def collate_fn(batch):
    images, conversations, is_code = zip(*batch)
    prompts = [
        processor.apply_chat_template(messages, add_generation_prompt=True)
        for messages in conversations
    ]
    return prompts, list(images), is_code


def remove_image_tokens(input_ids, target, image_token_id):
    ids_list = []
    target_list = []
    for batch_index in range(input_ids.shape[0]):
        keep = input_ids[batch_index] != image_token_id
        ids_list.append(input_ids[batch_index][keep])
        target_list.append(target[batch_index][keep])
    return (
        pad_sequence(ids_list, batch_first=True, padding_value=0),
        pad_sequence(target_list, batch_first=True, padding_value=0.0),
    )


def build_loss_mask(input_ids, assistant_tokens, user_tokens):
    loss_mask = torch.zeros_like(input_ids)
    assistant_length = len(assistant_tokens)
    user_length = len(user_tokens)
    for batch_index, tokens in enumerate(input_ids.cpu()):
        assistant_start = None
        token_index = 0
        while token_index < tokens.numel():
            if (
                assistant_start is None
                and token_index <= tokens.numel() - assistant_length
                and tokens[token_index : token_index + assistant_length].tolist()
                == assistant_tokens
            ):
                assistant_start = token_index + assistant_length
                token_index += assistant_length
                continue
            if (
                assistant_start is not None
                and token_index <= tokens.numel() - user_length
                and tokens[token_index : token_index + user_length].tolist()
                == user_tokens
            ):
                loss_mask[batch_index, assistant_start:token_index] = 1
                assistant_start = None
                token_index += user_length
                continue
            token_index += 1
        if assistant_start is not None:
            # Exclude the template's trailing special token(s), matching the source scripts.
            end = -assistant_length if assistant_length else None
            loss_mask[batch_index, assistant_start:end] = 1
    return loss_mask


@torch.no_grad()
def generate_sample(batch):
    prompts, images, _ = batch
    processor_args = {
        "text": prompts,
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": args.max_length,
        "padding_side": "left",
    }
    if args.data_type == "multimodal":
        processor_args["images"] = images
    inputs = processor(**processor_args).to(model.device)
    outputs = model(**inputs, output_hidden_states=True)

    input_ids = inputs.input_ids
    target = outputs.hidden_states[-1]
    if args.data_type == "multimodal":
        input_ids, target = remove_image_tokens(
            input_ids,
            target,
            image_token_id,
        )

    loss_mask = build_loss_mask(input_ids, assistant_tokens, user_tokens)
    return {
        "loss_mask": loss_mask.cpu()[0],
        "input_ids": input_ids.cpu()[0],
        "target": target.cpu()[0],
    }


def save_sample(output_dir, sample):
    sample_index = len(list(output_dir.glob("data_*.ckpt")))
    torch.save(sample, output_dir / f"data_{sample_index}.ckpt")


model, model_config, processor = load_model(args.model_path)
if args.outdir is None:
    dataset_name = "sharegpt" if args.data_type == "text" else "llava_v1_5_mix665k"
    args.outdir = Path("eval_data/generated") / model_directory_name(args.model_path, model_config) / dataset_name
tokenizer = processor.tokenizer
assistant_marker, user_marker = (
    ("[/INST]:", "[INST]:")
    if "mistral" in args.model_path.lower()
    else ("ASSISTANT:", "USER:")
)
assistant_tokens = tokenizer.encode(assistant_marker, add_special_tokens=False)
user_tokens = tokenizer.encode(user_marker, add_special_tokens=False)
image_token_id = getattr(model_config, "image_token_index", None)
if args.data_type == "multimodal" and image_token_id is None:
    raise ValueError("The model config has no image_token_index")

dataset = load_dataset("json", data_files=args.data_file, split="train")
dataset = dataset.shuffle(seed=args.seed)
if args.data_type == "multimodal":
    dataset = dataset.filter(
        lambda row: row.get("image") is not None
        and row.get("conversations") is not None
    )
else:
    dataset = dataset.filter(lambda row: row.get("conversations") is not None)
end = min(args.end, len(dataset))
dataset = dataset.select(range(args.start, end))
data_loader = DataLoader(
    ConversationDataset(dataset, args.data_type, args.image_root),
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
)

if args.data_type == "text":
    code_output_dir = args.outdir / "code" / str(args.index)
    non_code_output_dir = args.outdir / "non_code" / str(args.index)
    code_output_dir.mkdir(parents=True, exist_ok=True)
    non_code_output_dir.mkdir(parents=True, exist_ok=True)
else:
    output_dir = args.outdir / str(args.index)
    output_dir.mkdir(parents=True, exist_ok=True)
for batch_index, batch in enumerate(tqdm(data_loader)):
    torch.cuda.empty_cache()
    try:
        if args.data_type == "text":
            output_dir = code_output_dir if batch[2][0] else non_code_output_dir
        save_sample(output_dir, generate_sample(batch))
    except Exception as error:
        absolute_index = args.start + batch_index
        if args.fail_fast:
            raise
        tqdm.write(f"Skipping dataset index {absolute_index}: {error}")
