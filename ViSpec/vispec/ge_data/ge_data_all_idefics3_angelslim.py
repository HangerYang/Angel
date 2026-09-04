"""Generate ViSpec training data from an AngelSlim-format record jsonl.

ViSpec's own generators read two fixed sources -- Aeala/ShareGPT_Vicuna_unfiltered
for text and LLaVA-Pretrain for images -- and neither is what this project
trains on. This reads one AngelSlim `{"id", "conversations": [...]}` jsonl
instead and routes each record by its own content, so ViSpec and the
EAGLE3/HiViS baselines consume the *same* source data and stay comparable.

Two output dirs, because ViSpec's two training stages read different ones and
the data they want is genuinely different:

  * text-only records -> --outdir. ViSpec's teacher-forced path: one plain
    forward over the stored conversation, loss masked to the assistant turns.
    (`ge_data_all_idefics3_shargpt.py`)
  * records with an image -> --multimodal-outdir. ViSpec's method proper: the
    Vicuna system prompt plus "Please answer with at least 1000 words.", then
    `generate()`, loss masked to the model's own rollout.
    (`ge_data_all_idefics3_pretrain_gen.py`)

The visual span is deliberately kept in the saved tensors -- `image_mask` marks
it and the ImgAdaptor is trained to compress it. This is why HiViS's already
generated hidden states cannot be reused here: `ge_data_smolvlm.py` deletes the
image rows before saving, so the rows ViSpec needs are simply absent.
"""

import argparse

parser = argparse.ArgumentParser(description="ViSpec data gen over an AngelSlim jsonl")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument("--outdir", type=str, default=None,
                    help="Output root for TEXT-ONLY records (ViSpec stage 2.1).")
parser.add_argument("--multimodal-outdir", type=str, default=None,
                    help="Output root for records WITH an image (ViSpec stage 2.2).")
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max-length", type=int, default=2048,
                    help="Truncation for the text path's teacher-forced input_ids.")
parser.add_argument(
    "--datapath",
    type=str,
    required=True,
    help="AngelSlim record jsonl, e.g. .../smolvlm256m/train_path.jsonl",
)
parser.add_argument(
    "--image-root",
    type=str,
    default="/home/hyang/Angel/dataset/raw/images",
    help="Root for relative image paths (coco/..., textvqa/...). Unused when "
         "the records carry inline base64, as train_path_b64.jsonl does.",
)
args = parser.parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

import base64
import io as _io
import json

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

VISPEC_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
VISPEC_LENGTH_INSTRUCTION = "Please answer with at least 1000 words."


def read_records(path, start, end):
    """Yield (line_index, record) for line indices in [start, end).

    Streamed rather than json-loaded whole: train_path.jsonl is ~0.65 GB and
    every worker reads its own slice of the same file.
    """
    with open(path, encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index < start:
                continue
            if line_index >= end:
                return
            line = line.strip()
            if line:
                yield line_index, json.loads(line)


def split_content(content):
    """An AngelSlim content list -> (its text, its relative image paths)."""
    texts, images = [], []
    for block in content:
        if block.get("type") == "text":
            texts.append(block["text"])
        elif block.get("type") == "image":
            images.append(block["image"])
    return "\n".join(texts).strip(), images


def load_image_ref(ref, root):
    """Load one image reference, whichever form this jsonl uses.

    The same folder ships both: `train_path.jsonl` carries relative paths
    (`coco/...`, resolved under --image-root) and `train_path_b64.jsonl`
    carries inline `data:image/...;base64,` URIs. Accepting both means either
    file works here, and matches what HiViS's `load_image_field` accepts, so
    the two pipelines can be pointed at the same data.
    """
    if ref.startswith("data:image/"):
        _, _, payload = ref.partition(",")
        with Image.open(_io.BytesIO(base64.b64decode(payload))) as opened:
            return opened.convert("RGB").copy()
    for candidate in (ref, os.path.join("images", ref)):
        path = os.path.join(root, candidate)
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
    # Absolute paths, or a root that already includes the prefix.
    return Image.open(ref if os.path.isabs(ref) else os.path.join(root, ref)).convert("RGB")


def build_text_messages(record):
    """Text-only record -> SmolVLM chat messages, or None if unusable.

    Same shape check as ViSpec's ShareGPT path: turns must alternate starting
    at user, and a lone user turn supervises nothing.
    """
    roles = ["user", "assistant"]
    messages = []
    for turn_index, turn in enumerate(record["conversations"]):
        if turn.get("role") != roles[turn_index % 2]:
            return None
        text, _ = split_content(turn["content"])
        messages.append({"role": turn["role"], "content": [{"type": "text", "text": text}]})
    if len(messages) < 2:
        return None
    return messages


def build_multimodal_prompt(record):
    """Image record -> (prompt string, absolute image paths), ViSpec-style.

    Mirrors `ge_data_all_idefics3_pretrain_gen.py`: only the user turns are
    kept (the stored assistant answer is discarded -- ViSpec trains on the
    target's own rollout), each carrying the length instruction.
    """
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": VISPEC_SYSTEM_PROMPT}]}
    ]
    image_refs = []
    for turn in record["conversations"]:
        if turn.get("role") != "user":
            continue
        text, images = split_content(turn["content"])
        content = [{"type": "text", "text": text}]
        for ref in images:
            content.append({"type": "image"})
            image_refs.append(ref)
        content.append({"type": "text", "text": VISPEC_LENGTH_INSTRUCTION})
        conversation.append({"role": "user", "content": content})
    if not image_refs:
        return None, []
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt, image_refs


def build_loss_mask(messages, input_ids):
    """Supervise only assistant-generated tokens.

    Separator-agnostic, same method as ViSpec's ShareGPT path: re-tokenize the
    prompt prefix and the full prefix around each assistant turn and mark the
    span between them, rather than pattern-matching a template separator.
    """
    loss_mask = torch.zeros_like(input_ids)
    for turn_index in range(1, len(messages), 2):
        prompt_prefix = processor.apply_chat_template(
            messages[:turn_index], tokenize=False, add_generation_prompt=True
        )
        full_prefix = processor.apply_chat_template(
            messages[: turn_index + 1], tokenize=False, add_generation_prompt=False
        )
        start = len(tokenizer(prompt_prefix, add_special_tokens=False).input_ids)
        end = len(tokenizer(full_prefix, add_special_tokens=False).input_ids)
        start = min(start, input_ids.shape[0])
        end = min(end, input_ids.shape[0])
        loss_mask[start:end] = 1
    return loss_mask


@torch.no_grad()
def generate_text_sample(record):
    """Teacher-forced forward over a text-only record."""
    messages = build_text_messages(record)
    if messages is None:
        return None
    conversation = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = tokenizer(
        conversation,
        return_tensors="pt",
        max_length=args.max_length,
        truncation=True,
        add_special_tokens=False,
    ).input_ids[0]
    loss_mask = build_loss_mask(messages, input_ids)
    if int(loss_mask.sum()) == 0:
        return None
    outs = model(input_ids[None, :].to(model.device), output_hidden_states=True)
    return {
        "inputs_embeds": outs.hidden_states[0].cpu()[0],
        "input_ids": input_ids.cpu(),
        "hidden_state": outs.hidden_states[-1].cpu()[0],
        "loss_mask": loss_mask.cpu(),
    }


@torch.no_grad()
def generate_multimodal_sample(record):
    """ViSpec's rollout capture: generate, then keep the whole span with its
    image rows intact and an `image_mask` marking them."""
    prompt, image_refs = build_multimodal_prompt(record)
    if prompt is None:
        return None
    images = [load_image_ref(ref, args.image_root) for ref in image_refs]
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(model.device)

    outs = model.generate(
        **inputs,
        output_hidden_states=True,
        return_dict_in_generate=True,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature != 0,
        temperature=args.temperature,
    )

    inputs_embeds = torch.cat([step[0] for step in outs.hidden_states], dim=1)
    hidden_state = torch.cat([step[-1] for step in outs.hidden_states], dim=1)

    image_mask = (outs.sequences == model.config.image_token_id)[..., :-1]
    loss_mask = torch.ones_like(outs.sequences[:, :-1], dtype=bool)
    # Everything up to the end of the prompt is context, not a training target.
    loss_mask[:, : inputs["input_ids"].shape[-1] - 1] = 0
    return {
        "inputs_embeds": inputs_embeds.cpu()[0],
        "hidden_state": hidden_state.cpu()[0],
        "loss_mask": loss_mask.cpu()[0],
        "image_mask": image_mask.cpu()[0],
    }


def write_sample(outdir, sample, idx):
    os.makedirs(outdir, exist_ok=True)
    torch.save(sample, os.path.join(outdir, f"data_{idx}.ckpt"))


processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
tokenizer = processor.tokenizer
model = AutoModelForImageTextToText.from_pretrained(
    args.model, device_map="auto", torch_dtype=torch.bfloat16
)
model.eval()

text_outdir = os.path.join(args.outdir, str(args.index)) if args.outdir else None
mm_outdir = (
    os.path.join(args.multimodal_outdir, str(args.index))
    if args.multimodal_outdir
    else None
)

written_text = written_mm = skipped = failed = 0
for line_index, record in tqdm(read_records(args.datapath, args.start, args.end)):
    has_image = any(
        block.get("type") == "image"
        for turn in record["conversations"]
        for block in turn["content"]
    )
    outdir = mm_outdir if has_image else text_outdir
    if outdir is None:
        skipped += 1
        continue
    try:
        sample = (
            generate_multimodal_sample(record)
            if has_image
            else generate_text_sample(record)
        )
    except Exception as exc:  # one bad record must not kill a multi-hour shard
        print(f"[worker {args.index}] record {line_index} failed: {type(exc).__name__}: {exc}")
        failed += 1
        continue
    if sample is None:
        skipped += 1
        continue
    write_sample(outdir, sample, line_index)
    if has_image:
        written_mm += 1
    else:
        written_text += 1

print(
    f"[worker {args.index}] text={written_text} multimodal={written_mm} "
    f"skipped={skipped} failed={failed}"
)
