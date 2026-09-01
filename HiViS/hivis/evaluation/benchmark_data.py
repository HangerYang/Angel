"""Benchmark loading and multimodal input construction for HiViS evaluation."""

import ast
import json
import re
from pathlib import Path

from datasets import Dataset, Image, concatenate_datasets, load_dataset


SAMPLE_COUNT = 80
SEED = 42
_DATA_DIR = Path(__file__).resolve().parent / "data"

_HF_SPECS = {
    "ScienceQA": ("derek-thomas/ScienceQA", None, "validation"),
    "ChartQA": ("HuggingFaceM4/ChartQA", None, "test"),
    "MathVista": ("AI4Math/MathVista", None, "testmini"),
    "DocVQA": ("lmms-lab-encoder/DocVQA", "DocVQA", "test"),
}

# Benchmarks we added so an AngelSlim drafter and a HiViS/ViSpec drafter can be
# measured on the same rows. These are sampled FIRST-N, not shuffled, because
# tools/vllm_offline_eagle3_vlm_batch.py samples first-N -- keeping the same
# rows is what makes a PyTorch number comparable to a vLLM number. Do not
# "fix" them to use SEED.
_ANGELSLIM_SPECS = {
    "omnidocbench": ("opendatalab/OmniDocBench", None, "train"),
    "mmmu_history": ("MMMU/MMMU", "History", "test"),
}

# Byte-identical to _OCR_PROMPT in tools/vllm_offline_eagle3_vlm_batch.py.
_ANGELSLIM_OCR_PROMPT = (
    "Perform an OCR task on the provided image. Extract the text accurately "
    "and provide a detailed explanation of the process. Ensure the response "
    "is comprehensive and well-structured."
)

_IMG_REF_RE = re.compile(r"<image\s*(\d+)\s*>", flags=re.IGNORECASE)
_OCR_SUFFIX = " Perform an OCR task on the provided image. Please extract the text accurately and provide a detailed explanation of the process. Ensure the response is comprehensive and well-structured."


def _read_jsonl(filename):
    with (_DATA_DIR / filename).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _load_mmmu(sample_count):
    datasets = [
        load_dataset("MMMU/MMMU", subject, split="test")
        for subject in ("Accounting", "Art", "Biology", "Math")
    ]
    dataset = concatenate_datasets(datasets)
    dataset = dataset.filter(
        lambda row: sum(row.get(f"image_{index}") is not None for index in range(1, 8)) <= 3
    )
    return dataset.shuffle(seed=SEED).select(
        range(min(sample_count, len(dataset)))
    )


def _load_gqa(sample_count):
    records = Dataset.from_list(
        _read_jsonl("llava_gqa_testdev_balanced.jsonl")
    )
    records = records.shuffle(seed=SEED).select(
        range(min(sample_count, len(records)))
    )
    image_root = Path(__file__).resolve().parents[2] / "eval_data/llava_v1_5_mix665k/images/gqa/images"
    records = records.map(
        lambda row: {"image_data": str(image_root / row["image"])}
    )
    return records.cast_column("image_data", Image())


def _load_textvqa(sample_count):
    with (_DATA_DIR / "TextVQA_0.5.1_test.json").open(encoding="utf-8") as file:
        records = Dataset.from_list(json.load(file)["data"][:sample_count])
    image_root = Path(__file__).resolve().parents[2] / "eval_data/llava_v1_5_mix665k/images/textvqa/test_images"
    records = records.map(lambda row: {"image": str(image_root / f"{row['image_id']}.jpg")})
    return records.cast_column("image", Image())


def _load_mme(sample_count):
    records = Dataset.from_list(_read_jsonl("llava_mme.jsonl"))
    records = records.shuffle(seed=SEED).select(
        range(min(sample_count, len(records)))
    )
    image_root = _DATA_DIR / "MME_Benchmark_release_version/MME_Benchmark"
    records = records.map(
        lambda row: {"image_data": str(image_root / row["image"])}
    )
    return records.cast_column("image_data", Image())


def _load_mmvet(sample_count):
    with (_DATA_DIR / "mm-vet.json").open(encoding="utf-8") as file:
        source = json.load(file)
    records = []
    for question_id, row in source.items():
        record = dict(row)
        record["id"] = question_id
        records.append(record)
    records = Dataset.from_list(records)
    records = records.shuffle(seed=SEED).select(
        range(min(sample_count, len(records)))
    )

    image_root = _DATA_DIR / "mm-vet/images"
    records = records.map(
        lambda row: {"image_data": str(image_root / row["imagename"])}
    )
    return records.cast_column("image_data", Image())


def _load_seedbench(sample_count):
    """Load the bundled LLaVA questions and locally extracted SEED images."""
    image_root = _DATA_DIR / "SEED-Bench-image"
    records = []
    for row in _read_jsonl("llava-seed-bench.jsonl"):
        question_id = row.get("question_id", "")
        if isinstance(question_id, int) or str(question_id).isdigit():
            row["question_id"] = str(question_id)
            records.append(row)
    dataset = Dataset.from_list(records)
    dataset = dataset.shuffle(seed=SEED).select(
        range(min(sample_count, len(dataset)))
    )
    dataset = dataset.map(
        lambda row: {"image": str(image_root / Path(row["image"]).name)}
    )
    return dataset.cast_column("image", Image())


def supported_benchmarks():
    """Every name load_benchmark accepts."""
    return sorted(
        list(_ANGELSLIM_SPECS)
        + list(_HF_SPECS)
        + ["gqa", "mme", "mmvet", "seedbench", "vqav2", "textvqa", "mmmu"]
    )


def load_benchmark(name, sample_count=SAMPLE_COUNT):
    """Load and deterministically sample one supported benchmark."""
    if name in _ANGELSLIM_SPECS:
        repo_id, config_name, split = _ANGELSLIM_SPECS[name]
        dataset = load_dataset(repo_id, config_name, split=split)
        return dataset.select(range(min(sample_count, len(dataset))))
    if name in _HF_SPECS:
        repo_id, config_name, split = _HF_SPECS[name]
        dataset = load_dataset(repo_id, config_name, split=split)
        if name == "ScienceQA":
            dataset = dataset.filter(lambda row: row["image"] is not None)
        if name == "MathVista":
            dataset = dataset.filter(
                lambda row: row["decoded_image"] is not None
                and not bool(re.search(r"[\u4e00-\u9fff]", row.get("question", "")))
            )
        return dataset.shuffle(seed=SEED).select(
            range(min(sample_count, len(dataset)))
        )

    if name == "gqa":
        return _load_gqa(sample_count)
    if name == "mme":
        return _load_mme(sample_count)
    if name == "mmvet":
        return _load_mmvet(sample_count)
    if name == "seedbench":
        return _load_seedbench(sample_count)
    if name == "vqav2":
        return load_dataset(
            "lmms-lab-encoder/VQAv2", split=f"test[:{sample_count}]"
        )
    if name == "textvqa":
        return _load_textvqa(sample_count)
    if name == "mmmu":
        return _load_mmmu(sample_count)
    raise ValueError(f"Unsupported dataset: {name}")


def _message(text, image_count=1, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    content = [{"type": "image"} for _ in range(image_count)]
    content.append({"type": "text", "text": text})
    messages.append({"role": "user", "content": content})
    return messages


def _parse_options(options):
    if options is None:
        return []
    if isinstance(options, str):
        try:
            parsed = ast.literal_eval(options)
            return list(parsed) if isinstance(parsed, (list, tuple)) else [str(parsed)]
        except Exception:
            return [options]
    return list(options) if isinstance(options, (list, tuple)) else [str(options)]


def _prepare_mmmu(row):
    used = []
    question = row.get("question", "").strip()
    for match in _IMG_REF_RE.finditer(question):
        index = int(match.group(1))
        if index not in used:
            used.append(index)
    question = _IMG_REF_RE.sub(lambda match: f"Image{int(match.group(1))}", question)

    options = []
    for index, option in enumerate(_parse_options(row.get("options"))):
        option = str(option)
        for match in _IMG_REF_RE.finditer(option):
            image_index = int(match.group(1))
            if image_index not in used:
                used.append(image_index)
        option = _IMG_REF_RE.sub(
            lambda match: f"Image{int(match.group(1))}", option
        ).strip()
        tag = chr(65 + index) if index < 26 else f"Option{index + 1}"
        options.append(f"({tag}) {option}")
    if options:
        question += "\nOptions: " + " ".join(options)
    images = [row.get(f"image_{index}") for index in used if row.get(f"image_{index}") is not None]
    return _message(question, len(images)), images


def prepare_inputs(model, dataset, index, dataset_name, truncation=False):
    """Convert one benchmark row into processor inputs on the model device."""
    row = dataset[index]
    if dataset_name == "omnidocbench":
        messages, image = _message(_ANGELSLIM_OCR_PROMPT), row["image"]
    elif dataset_name == "mmmu_history":
        question = _IMG_REF_RE.sub("", row["question"]).strip()
        messages = _message(
            f"Answer this question: {question} "
            "Then describe the image in detail to justify your answer."
        )
        image = row["image_1"]
    elif dataset_name == "ScienceQA":
        choices = " ".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(row["choices"]))
        messages, image = _message(f"{row['question']} {choices}"), row["image"]
    elif dataset_name == "vqav2":
        messages = _message(row["question"])
        image = row["image"]
    elif dataset_name == "textvqa":
        messages = _message(row["question"] + _OCR_SUFFIX)
        image = row["image"]
    elif dataset_name == "mme":
        messages = _message(row["text"].partition("\n")[0])
        image = row["image_data"]
    elif dataset_name == "mmvet":
        messages = _message(row["question"].partition("\n")[0])
        image = row["image_data"]
    elif dataset_name == "seedbench":
        messages, image = _message(row["question"]), row["image"]
    elif dataset_name == "gqa":
        messages = _message(row["text"].partition("\n")[0])
        image = row["image_data"]
    elif dataset_name == "ChartQA":
        messages, image = _message(row["query"]), row["image"]
    elif dataset_name == "MathVista":
        messages = _message(row["question"] + "\nPlease answer with an explanation.")
        image = row["decoded_image"]
    elif dataset_name == "DocVQA":
        messages, image = _message(row["question"] + _OCR_SUFFIX), row["image"]
    elif dataset_name == "mmmu":
        messages, image = _prepare_mmmu(row)
    else:
        raise ValueError(f"Unsupported dataset input adapter: {dataset_name}")

    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    return model.processor(
        images=image,
        text=prompt,
        truncation=truncation,
        return_tensors="pt",
    ).to(model.base_model.device)
