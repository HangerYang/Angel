"""Generate hidden-state data from a SmolVLM (Idefics3) model for HiViS training.

Model-specific parts only -- see common.py for the schema/loss-mask/is_code
logic shared with ge_data_qwen.py. One pass over the input file loads the
model once and routes each record based on its own content: has an image ->
--multimodal-outdir; text-only -> --outdir/code or --outdir/non_code (HiViS's
own is_code_heavy() heuristic, applied to the assistant turns).
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
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Idefics3ForConditionalGeneration

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


@torch.no_grad()
def generate_sample(batch):
    prompt, images, has_image, _is_code = batch
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


model, model_config, processor = load_model(args.model_path)
if args.outdir is None:
    args.outdir = Path("eval_data/generated") / model_directory_name(args.model_path, model_config) / "sharegpt"
if args.multimodal_outdir is None:
    args.multimodal_outdir = (
        Path("eval_data/generated") / model_directory_name(args.model_path, model_config) / "smolvlm_mixed"
    )

tokenizer = processor.tokenizer
assistant_tokens = tokenizer.encode("Assistant:", add_special_tokens=False)
end_tokens = tokenizer.encode("<end_of_utterance>", add_special_tokens=False)
image_token_id = model_config.image_token_id

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
