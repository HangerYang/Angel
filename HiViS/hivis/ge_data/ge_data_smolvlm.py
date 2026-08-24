"""Generate hidden-state data from a SmolVLM (Idefics3) model for HiViS training.

Handles the role/content dataset format where image entries carry absolute paths,
auto-detecting whether each record is text-only or multimodal.
"""

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_PATH = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_DATA = "dataset/mixed_text_vl_36/mixed_text_vl_36.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SmolVLM hidden-state data for HiViS training."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100000)
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
    parser.add_argument("--data-file", default=DEFAULT_DATA)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))

import torch
from datasets import load_dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Idefics3ForConditionalGeneration

from .model_names import model_directory_name


def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type != "idefics3":
        raise ValueError(
            f"Unsupported model type {config.model_type!r}; expected idefics3 (SmolVLM)."
        )
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, config, AutoProcessor.from_pretrained(model_path)


def record_to_messages_and_image(record):
    """Convert role/content format to HF messages list and load the first image."""
    conversations = record["conversations"]
    messages = []
    image = None
    for turn in conversations:
        role = turn["role"]  # "user" or "assistant"
        content_entries = turn["content"]
        new_content = []
        for entry in content_entries:
            entry_type = entry.get("type")
            if entry_type == "image":
                img_path = entry.get("image", "")
                if img_path and image is None:
                    try:
                        with Image.open(img_path) as opened:
                            image = opened.convert("RGB").copy()
                    except Exception as exc:
                        print(f"Could not load image {img_path}: {exc}")
                new_content.append({"type": "image"})
            elif entry_type == "text":
                new_content.append({"type": "text", "text": entry.get("text", "")})
        messages.append({"role": role, "content": new_content})
    return messages, image


class ConversationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        record = self.dataset[index]
        messages, image = record_to_messages_and_image(record)
        return messages, image


def collate_fn(batch):
    conversations, images = zip(*batch)
    prompt = processor.apply_chat_template(
        conversations[0], add_generation_prompt=True
    )
    return prompt, list(images)


def remove_image_tokens(input_ids, target, image_token_id):
    ids_list = []
    target_list = []
    for i in range(input_ids.shape[0]):
        keep = input_ids[i] != image_token_id
        ids_list.append(input_ids[i][keep])
        target_list.append(target[i][keep])
    return (
        pad_sequence(ids_list, batch_first=True, padding_value=0),
        pad_sequence(target_list, batch_first=True, padding_value=0.0),
    )


def build_loss_mask(input_ids, assistant_tokens, end_tokens):
    loss_mask = torch.zeros_like(input_ids)
    asst_len = len(assistant_tokens)
    end_len = len(end_tokens)
    for bi, tokens in enumerate(input_ids.cpu()):
        asst_start = None
        i = 0
        while i < tokens.numel():
            if (
                asst_start is None
                and i <= tokens.numel() - asst_len
                and tokens[i : i + asst_len].tolist() == assistant_tokens
            ):
                asst_start = i + asst_len
                i += asst_len
                continue
            if (
                asst_start is not None
                and i <= tokens.numel() - end_len
                and tokens[i : i + end_len].tolist() == end_tokens
            ):
                loss_mask[bi, asst_start:i] = 1
                asst_start = None
                i += end_len
                continue
            i += 1
        if asst_start is not None:
            end = -asst_len if asst_len else None
            loss_mask[bi, asst_start:end] = 1
    return loss_mask


@torch.no_grad()
def generate_sample(batch):
    prompt, images = batch
    has_image = any(img is not None for img in images)
    processor_kwargs = {
        "text": [prompt],
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": args.max_length,
        "padding_side": "left",
    }
    if has_image:
        processor_kwargs["images"] = [img for img in images if img is not None]
    inputs = processor(**processor_kwargs).to(model.device)
    outputs = model(**inputs, output_hidden_states=True)
    input_ids = inputs.input_ids
    target = outputs.hidden_states[-1]
    if has_image:
        input_ids, target = remove_image_tokens(input_ids, target, image_token_id)
    loss_mask = build_loss_mask(input_ids, assistant_tokens, end_tokens)
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
    args.outdir = (
        Path("eval_data/generated")
        / model_directory_name(args.model_path, model_config)
        / "smolvlm_mixed"
    )

tokenizer = processor.tokenizer
assistant_tokens = tokenizer.encode("Assistant:", add_special_tokens=False)
end_tokens = tokenizer.encode("<end_of_utterance>", add_special_tokens=False)
image_token_id = model_config.image_token_id

output_dir = args.outdir / str(args.index)
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("json", data_files=str(args.data_file), split="train")
dataset = dataset.shuffle(seed=args.seed)
dataset = dataset.filter(lambda row: row.get("conversations") is not None)
end = min(args.end, len(dataset))
dataset = dataset.select(range(args.start, end))

data_loader = DataLoader(
    ConversationDataset(dataset),
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
)

for batch_index, batch in enumerate(tqdm(data_loader)):
    torch.cuda.empty_cache()
    try:
        save_sample(output_dir, generate_sample(batch))
    except Exception as error:
        absolute_index = args.start + batch_index
        if args.fail_fast:
            raise
        tqdm.write(f"Skipping dataset index {absolute_index}: {error}")
