# HiViS: Hiding Visual Tokens from the Drafter for Speculative Decoding in Vision-Language Models

Official implementation of **HiViS**, a speculative decoding framework for vision-language models.

[Paper](https://arxiv.org/abs/2509.23928)

## Contents

- [Installation](#installation)
- [Evaluation](#1-evaluation)
  - [Prepare benchmarks](#11-prepare-benchmarks)
  - [Run HiViS](#12-run-hivis)
  - [Autoregressive baseline and speedup](#13-autoregressive-baseline-and-speedup)
- [Training](#2-training)
  - [Prepare JSON data](#21-prepare-json-data)
  - [Optional response regeneration](#22-optional-response-regeneration)
  - [Generate training checkpoints](#23-generate-training-checkpoints)
  - [Two-stage training](#24-two-stage-training)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Installation

```bash
conda create -n hivis python=3.9 -y
conda activate hivis
cd HiViS
pip install -r requirements.txt
```

HiViS uses Torch 2.6.

## 1. Evaluation

Released HiViS checkpoints:

| Base Model | HiViS Drafter |
| --- | --- |
| [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | [Irisssme/HiViS-llava-1.5-7b-hf](https://huggingface.co/Irisssme/HiViS-llava-1.5-7b-hf) |
| [llava-hf/llava-1.5-13b-hf](https://huggingface.co/llava-hf/llava-1.5-13b-hf) | [Irisssme/HiViS-llava-1.5-13b-hf](https://huggingface.co/Irisssme/HiViS-llava-1.5-13b-hf) |
| [llava-hf/llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf) | [Irisssme/HiViS-llava-v1.6-vicuna-7b-hf](https://huggingface.co/Irisssme/HiViS-llava-v1.6-vicuna-7b-hf) |
| [llava-hf/llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) | [Irisssme/HiViS-llava-v1.6-vicuna-13b-hf](https://huggingface.co/Irisssme/HiViS-llava-v1.6-vicuna-13b-hf) |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | [Irisssme/HiViS-Qwen2.5-VL-7B-Instruct](https://huggingface.co/Irisssme/HiViS-Qwen2.5-VL-7B-Instruct) |

For EAGLE-2 baseline evaluation with [`llava-hf/llava-v1.6-vicuna-7b-hf`](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf), use the released [`Irisssme/EAGLE-2-llava-v1.6-vicuna-7b-hf`](https://huggingface.co/Irisssme/EAGLE-2-llava-v1.6-vicuna-7b-hf) checkpoint with `--draft-method eagle`.

### 1.1 Prepare benchmarks

ScienceQA, ChartQA, MathVista, DocVQA, VQAv2, and MMMU are downloaded automatically from Hugging Face. GQA, TextVQA, MME, MM-Vet, and SEED-Bench use the metadata included in `hivis/evaluation/data` and locally downloaded images.

Download the remaining images from [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), [TextVQA](https://textvqa.org/dataset/), [MME](https://huggingface.co/datasets/darkyarding/MME), [MM-Vet](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip), and [SEED-Bench](https://huggingface.co/datasets/AILab-CVC/SEED-Bench), then arrange them as follows:

```text
hivis/evaluation/data/
├── llava_gqa_testdev_balanced.jsonl
├── TextVQA_0.5.1_test.json
├── llava_mme.jsonl
├── mm-vet.json
├── llava-seed-bench.jsonl
├── MME_Benchmark_release_version/
│   └── MME_Benchmark/
├── mm-vet/
│   └── images/
└── SEED-Bench-image/

eval_data/llava_v1_5_mix665k/images/
├── gqa/
│   └── images/
└── textvqa/
    ├── train_images/
    └── test_images/
```

### 1.2 Run HiViS

```bash
CUDA_VISIBLE_DEVICES=0 python -m hivis.evaluation.ge_hivis_answer \
  --draft-method={hivis,eagle,vispec} \
  --base-model-path "$BASE_MODEL_PATH" \
  --ea-model-path "$DRAFT_PATH" \
  --dataset={ChartQA,ScienceQA,MathVista,DocVQA,vqav2,textvqa,gqa,mme,mmvet,seedbench,mmmu} \
  --answer-file outputs/<dataset>_<draft-method>.jsonl
```

The evaluation code also supports ViSpec with its corresponding checkpoints. We thank [ViSpec](https://github.com/KangJialiang/ViSpec/tree/main) for making its code publicly available.

### 1.3 Autoregressive baseline and speedup

```bash
CUDA_VISIBLE_DEVICES=0 python -m hivis.evaluation.ge_baseline_answer_hivis \
  --base-model-path "$BASE_MODEL_PATH" \
  --ea-model-path "$DRAFT_PATH" \
  --dataset={ChartQA,ScienceQA,MathVista,DocVQA,vqav2,textvqa,gqa,mme,mmvet,seedbench,mmmu} \
  --answer-file outputs/<dataset>_baseline.jsonl

python -m hivis.evaluation.speed \
  --model-path "$BASE_MODEL_PATH" \
  --baseline-json outputs/<dataset>_baseline.jsonl \
  --hivis-json outputs/<dataset>_<draft-method>.jsonl
```

Each benchmark run uses 80 samples.

## 2. Training

HiViS training uses text-only ShareGPT data and multimodal LLaVA-v1.5-mix665k data in both stages.

### 2.1 Prepare JSON data

Download [ShareGPT](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V4.3_unfiltered_cleaned_split.json) to `eval_data/sharegpt`, then clean it:

```bash
python eval_data/sharegpt/prepare_data.py --input eval_data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json
```

Prepare the LLaVA multimodal index from [`liuhaotian/LLaVA-Instruct-150K`](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json):

```bash
python eval_data/llava_v1_5_mix665k/prepare_data.py
```

Place the corresponding LLaVA-mix images under `eval_data/llava_v1_5_mix665k/images`.

### 2.2 Optional response regeneration

Regenerating assistant responses with the target VLM can improve Stage-2 training. After starting a vLLM server, regenerate ShareGPT with:

```bash
python -m hivis.ge_data.regenerate_sharegpt \
  --vllm_model_name <served_model_name> \
  --vllm_urls <vllm_url>
```

Regenerate LLaVA-mix with:

```bash
python -m hivis.ge_data.regenerate_llava_mix \
  --base_dataset_path eval_data/llava_v1_5_mix665k/images \
  --vllm_model_name <served_model_name> \
  --vllm_urls <vllm_url>
```

The regenerated JSONL files are stored under model-specific subdirectories.

### 2.3 Generate training checkpoints

Generate text and multimodal `.ckpt` samples with the target VLM:

```bash
python -m hivis.ge_data.allocation \
  --model={llava,qwen} \
  --data-type={text,multimodal} \
  --model-path={Qwen/Qwen2.5-VL-7B-Instruct,llava-hf/llava-v1.6-vicuna-7b-hf,llava-hf/llava-v1.6-vicuna-13b-hf,llava-hf/llava-1.5-7b-hf,llava-hf/llava-1.5-13b-hf} \
  --start=0 \
  --end=68000 \
  --gpus 0 1 2 3
```

Pass `--data-file` with the model-specific regenerated JSONL path to generate checkpoints from regenerated responses. To keep them separate from Stage-1 data, set `--outdir` to `eval_data/generated/<model>/sharegpt_regenerated` or `eval_data/generated/<model>/llava_v1_5_mix665k_regenerated`, then use those directories for Stage 2.

The complete training data layout is:

```text
eval_data/
├── sharegpt/
│   ├── ShareGPT_V4.3_unfiltered_cleaned_split.json
│   ├── sharegpt.jsonl
│   └── <model>/
│       ├── sharegpt_regenerated.jsonl
│       └── sharegpt_regenerated_failed.jsonl
├── llava_v1_5_mix665k/
│   ├── llava_v1_5_mix665k_long_context.jsonl
│   ├── images/
│   │   ├── coco/
│   │   ├── gqa/
│   │   ├── ocr_vqa/
│   │   ├── textvqa/
│   │   └── vg/
│   └── <model>/
│       ├── llava_v1_5_mix665k_regenerated.jsonl
│       └── llava_v1_5_mix665k_regenerated_failed.jsonl
└── generated/
    ├── <model>/
    │   ├── sharegpt/
    │   │   ├── code/
    │   │   └── non_code/
    │   ├── llava_v1_5_mix665k/
    │   ├── sharegpt_regenerated/               # optional
    │   │   ├── code/
    │   │   └── non_code/
    │   └── llava_v1_5_mix665k_regenerated/     # optional
    └── ...
```

`<model>` is one of `llava16_7b`, `llava16_13b`, `llava15_7b`, `llava15_13b`, or `qwen25vl_7b`.

### 2.4 Two-stage training

Stage 1:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch -m --mixed_precision=bf16 hivis.train.main_mix \
  --basepath=<base_model_path> \
  --configpath=hivis/train/<model_config>.json \
  --text-data-dir=eval_data/generated/<model>/sharegpt \
  --multimodal-data-dir=eval_data/generated/<model>/llava_v1_5_mix665k \
  --cpdir=checkpoints/<model>/stage1
```

Stage 2:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch -m --mixed_precision=bf16 hivis.train.main_mix_topk_dyn_res \
  --basepath=<base_model_path> \
  --configpath=hivis/train/<model_config>.json \
  --text-data-dir=eval_data/generated/<model>/sharegpt/non_code \
  --multimodal-data-dir=eval_data/generated/<model>/llava_v1_5_mix665k \
  --ckpt_path=checkpoints/<model>/stage1/state_<epoch> \
  --cpdir=checkpoints/<model>/stage2
```

## Acknowledgements

We would like to acknowledge the foundational work of previous projects that inspired our approach, especially [EAGLE](https://github.com/SafeAILab/EAGLE) and [HASS](https://github.com/HArmonizedSS/HASS).

## Citation

```bibtex
@inproceedings{xie2026hivis,
  title={Hivis: Hiding visual tokens from the drafter for speculative decoding in vision-language models},
  author={Xie, Zhinan and Wang, Peisong and Qiu, Shuang and Cheng, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8952--8961},
  year={2026}
}
```

## License

Apache-2.0
