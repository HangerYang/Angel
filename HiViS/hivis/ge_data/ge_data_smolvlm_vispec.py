"""Generate hidden-state data from a SmolVLM (Idefics3) model for ViSpec-style training.

Identical to ge_data_smolvlm.py (teacher-forced forward pass, same loss-mask
mechanism) except multimodal samples are prefixed with ViSpec's Vicuna-style
system message and get "Please answer with at least 1000 words." appended to
the first user turn, matching upstream ViSpec's ge_data_all_llava_pretrain_gen.py
prompt (https://github.com/KangJialiang/ViSpec). Text-only samples are left
untouched, matching upstream ViSpec's ge_data_all_llava_shargpt.py.

Note this keeps the existing (already target-generated) assistant answer and
teacher-forces over it rather than calling model.generate() the way upstream
ViSpec does -- the instruction text is prompt context only, it does not force
the captured target to actually be 1000 words.
"""

import argparse
import base64
import io
import os
from pathlib import Path


DEFAULT_MODEL_PATH = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_DATA = "dataset/mixed_text_vl_36/mixed_text_vl_36.jsonl"

VISPEC_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
VISPEC_LENGTH_INSTRUCTION = " Please answer with at least 1000 words."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SmolVLM hidden-state data for ViSpec-style training."
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
    parser.add_argument(
        "--no-length-instruction",
        action="store_true",
        help="Disable the ViSpec system prompt / 1000-word instruction (falls back to plain teacher-forcing).",
    )
    return parser.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))

import torch
from datasets import Features, Value, load_dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Content items are either {"type": "text", "text": ...} or {"type": "image",
# "image": ...}. Without an explicit schema, datasets' JSON loader infers the
# "content" struct type from the first chunk it reads; since text-only and
# multimodal records live in different regions of the file, later chunks fail
# to cast into that narrower inferred schema. Fixing the schema up front
# avoids the mismatch.
RECORD_FEATURES = Features(
    {
        "id": Value("string"),
        "conversations": [
            {
                "role": Value("string"),
                "content": [
                    {
                        "type": Value("string"),
                        "text": Value("string"),
                        "image": Value("string"),
                    }
                ],
            }
        ],
    }
)
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


def load_image_field(img_ref):
    """Load a PIL image from either a base64 data URI or a file path."""
    if img_ref.startswith("data:image/"):
        _, _, b64_data = img_ref.partition(",")
        with Image.open(io.BytesIO(base64.b64decode(b64_data))) as opened:
            return opened.convert("RGB").copy()
    with Image.open(img_ref) as opened:
        return opened.convert("RGB").copy()


def record_to_messages_and_image(record, *, apply_vispec_prompt):
    """Convert role/content format to HF messages list and load the first image.

    When apply_vispec_prompt is set and the record is multimodal, prepends a
    system turn and appends the "at least 1000 words" instruction to the
    first user turn -- matching upstream ViSpec's multimodal prompt, applied
    only to multimodal samples (text-only ShareGPT samples are untouched).
    """
    conversations = record["conversations"]
    is_multimodal = any(
        entry.get("type") == "image"
        for turn in conversations
        for entry in turn.get("content", [])
    )

    messages = []
    if apply_vispec_prompt and is_multimodal:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": VISPEC_SYSTEM_PROMPT}]}
        )

    image = None
    first_user_seen = False
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
                        image = load_image_field(img_path)
                    except Exception as exc:
                        print(f"Could not load image {img_path[:80]}: {exc}")
                new_content.append({"type": "image"})
            elif entry_type == "text":
                new_content.append({"type": "text", "text": entry.get("text", "")})
        if (
            apply_vispec_prompt
            and is_multimodal
            and role == "user"
            and not first_user_seen
        ):
            new_content.append({"type": "text", "text": VISPEC_LENGTH_INSTRUCTION})
            first_user_seen = True
        messages.append({"role": role, "content": new_content})
    return messages, image


class ConversationDataset(Dataset):
    def __init__(self, dataset, apply_vispec_prompt):
        self.dataset = dataset
        self.apply_vispec_prompt = apply_vispec_prompt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        record = self.dataset[index]
        messages, image = record_to_messages_and_image(
            record, apply_vispec_prompt=self.apply_vispec_prompt
        )
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
        / "smolvlm_vispec_mixed"
    )

tokenizer = processor.tokenizer
assistant_tokens = tokenizer.encode("Assistant:", add_special_tokens=False)
end_tokens = tokenizer.encode("<end_of_utterance>", add_special_tokens=False)
image_token_id = model_config.image_token_id

output_dir = args.outdir / str(args.index)
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "json", data_files=str(args.data_file), split="train", features=RECORD_FEATURES
)
dataset = dataset.shuffle(seed=args.seed)
dataset = dataset.filter(lambda row: row.get("conversations") is not None)
end = min(args.end, len(dataset))
dataset = dataset.select(range(args.start, end))

data_loader = DataLoader(
    ConversationDataset(dataset, apply_vispec_prompt=not args.no_length_instruction),
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
