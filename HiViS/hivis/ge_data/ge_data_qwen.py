"""Generate hidden-state data from a Qwen2.5-VL model for HiViS training.

Model-specific parts only -- see common.py for the schema/loss-mask/is_code
logic shared with ge_data_smolvlm.py. One pass over the input file loads the
model once and routes each record based on its own content: has an image ->
--multimodal-outdir; text-only -> --outdir/code or --outdir/non_code (HiViS's
own is_code_heavy() heuristic, applied to the assistant turns).
"""

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_DATA = "dataset/preprocessed/mixed_sharegpt_llava665k_70k70k_b64.jsonl"
VISION_START_TOKEN_ID = 151652
IMAGE_TOKEN_ID = 151655


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Qwen2.5-VL hidden-state data for HiViS training."
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
    parser.add_argument(
        "--outdir", type=Path, default=None, help="Output directory for text-only records."
    )
    parser.add_argument(
        "--multimodal-outdir",
        type=Path,
        default=None,
        help="Output directory for records with at least one image.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-file", default=DEFAULT_DATA)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.end < args.start:
        parser.error("require 0 <= --start <= --end")
    return args


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))

# CUDA visibility must be configured before importing torch/transformers.
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .common import (
    assistant_text,
    build_loss_mask,
    is_code_heavy,
    load_record_dataset,
    record_has_image,
    record_to_messages_and_image,
    save_sample,
)
from .model_names import model_directory_name


def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type != "qwen2_5_vl":
        raise ValueError(
            f"Unsupported model type {config.model_type!r}; expected qwen2_5_vl."
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return model, config, AutoProcessor.from_pretrained(model_path)


class ConversationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        record = self.dataset[index]
        messages, image = record_to_messages_and_image(record)
        has_image = record_has_image(record)
        is_code = False if has_image else is_code_heavy(assistant_text(record))
        return messages, image, has_image, is_code


def collate_fn(batch):
    conversations, images, has_image, is_code = zip(*batch)
    prompt = processor.apply_chat_template(
        conversations[0], add_generation_prompt=True
    )
    return prompt, list(images), has_image[0], is_code[0]


def build_visual_keep_mask(input_ids):
    """Preserve the visual-span removal rule from the source Qwen script."""
    keep = torch.ones_like(input_ids, dtype=torch.bool)
    positions = (input_ids == VISION_START_TOKEN_ID).nonzero(as_tuple=False).squeeze(-1)
    if positions.numel() == 0:
        return keep
    if positions.numel() != 1:
        raise ValueError(
            f"Expected one vision span, found {positions.numel()} vision-start tokens"
        )
    start = positions.item()
    image_token_count = int((input_ids == IMAGE_TOKEN_ID).sum().item())
    end = min(start + image_token_count + 1, input_ids.numel() - 1)
    keep[start : end + 2] = False
    return keep


def remove_visual_span(input_ids, target, position_ids):
    ids_list = []
    target_list = []
    position_list = []
    for batch_index in range(input_ids.shape[0]):
        keep = build_visual_keep_mask(input_ids[batch_index])
        ids_list.append(input_ids[batch_index][keep])
        target_list.append(target[batch_index][keep])
        position_list.append(position_ids[batch_index][keep])
    return (
        pad_sequence(ids_list, batch_first=True, padding_value=0),
        pad_sequence(target_list, batch_first=True, padding_value=0.0),
        pad_sequence(position_list, batch_first=True, padding_value=0),
    )


@torch.no_grad()
def generate_sample(batch):
    prompt, images, has_image, _is_code = batch
    processor_kwargs = {
        "text": [prompt],
        "return_tensors": "pt",
        "padding": True,
        "padding_side": "left",
        "truncation": True,
        "max_length": args.max_length,
    }
    if has_image:
        processor_kwargs["images"] = [img for img in images if img is not None]

    inputs = processor(**processor_kwargs).to(model.device)
    outputs = model(**inputs, output_hidden_states=True)
    input_ids = inputs.input_ids
    target = outputs.hidden_states[-1]
    sequence_length = input_ids.shape[1]
    position_ids = torch.arange(
        sequence_length, dtype=torch.long, device=input_ids.device
    ).unsqueeze(0)

    if has_image:
        input_ids, target, position_ids = remove_visual_span(
            input_ids, target, position_ids
        )

    loss_mask = build_loss_mask(input_ids, assistant_tokens, end_tokens)
    return {
        "loss_mask": loss_mask.cpu()[0],
        "input_ids": input_ids.cpu()[0],
        "position_ids": position_ids.cpu()[0],
        "target": target.cpu()[0],
    }


model, model_config, processor = load_model(args.model_path)
if args.outdir is None:
    args.outdir = Path("eval_data/generated") / model_directory_name(args.model_path, model_config) / "sharegpt"
if args.multimodal_outdir is None:
    args.multimodal_outdir = (
        Path("eval_data/generated") / model_directory_name(args.model_path, model_config) / "llava_v1_5_mix665k"
    )

tokenizer = processor.tokenizer
assistant_tokens = tokenizer.encode(
    "<|im_start|>assistant\n", add_special_tokens=False
)
end_tokens = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

code_output_dir = args.outdir / "code" / str(args.index)
non_code_output_dir = args.outdir / "non_code" / str(args.index)
multimodal_output_dir = args.multimodal_outdir / str(args.index)
code_output_dir.mkdir(parents=True, exist_ok=True)
non_code_output_dir.mkdir(parents=True, exist_ok=True)
multimodal_output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_record_dataset(args.data_file)
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
        _, _, has_image, is_code = batch
        if has_image:
            output_dir = multimodal_output_dir
        else:
            output_dir = code_output_dir if is_code else non_code_output_dir
        save_sample(output_dir, generate_sample(batch))
    except Exception as error:
        absolute_index = args.start + batch_index
        if args.fail_fast:
            raise
        tqdm.write(f"Skipping dataset index {absolute_index}: {error}")
