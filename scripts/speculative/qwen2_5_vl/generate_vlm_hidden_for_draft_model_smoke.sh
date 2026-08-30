#!/usr/bin/env bash
# Smoke test: generate EAGLE3 draft-training hidden states for Qwen2.5-VL-7B
# using the angelslim vLLM-backend pipeline, distributed across GPUs via Ray.
#
# VLM + vLLM only works through tools/ray_generate_hidden_for_draft_model.py --
# the torchrun-based tools/generate_hidden_for_draft_model.py explicitly does
# not support it (torchrun's env vars collide with vLLM's own NCCL process
# groups; see VLMVLLMBackend.load_model's docstring). Ray sidesteps this by
# giving each GPU its own actor with its own independent vLLM engine, so no
# `ray start --head` dance is needed on a single node -- the script's own
# `ray.init()` boots a local cluster automatically.
set -euo pipefail

cd /home/hyang/Angel
conda_env=/home/hyang/anaconda3/envs/angel
python_bin="$conda_env/bin/python"
# `python tools/script.py` puts tools/ (not repo root) first on sys.path, so
# `import angelslim` would otherwise resolve to whatever stale copy is pip
# installed in the conda env's site-packages instead of this repo's source.
export PYTHONPATH="/home/hyang/Angel${PYTHONPATH:+:$PYTHONPATH}"
# We invoke the env's python by absolute path (no `conda activate`), so its
# bin/ isn't on PATH -- flashinfer's JIT kernel build shells out to `ninja`,
# which then can't be found without this.
export PATH="$conda_env/bin:$PATH"

TARGET_MODEL_PATH=/home/hyang/HiViS/models/Qwen2.5-VL-7B-Instruct
TRAIN_DATASET_PATH=/home/hyang/Angel/dataset/preprocessed/mixed_sharegpt_llava665k_70k70k.jsonl
OUTPUT_DIR=/home/hyang/Angel/dataset/qwen2_5_vl_7b_target_gen_smoke
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/qwen2.5-vl-7b-eagle3-mrope.json

SAMPLE_NUM=${SAMPLE_NUM:-32}
TOTAL_GPUS=${TOTAL_GPUS:-2}
TP_SIZE=1
MODEL_MAX_LENGTH=4096
MAX_MODEL_LEN=4096
MAX_PIXELS=153664
MIN_PIXELS=3136
GPU_MEMORY_UTILIZATION=0.85
LIMIT_MM_PER_PROMPT='{"image": 10, "video": 10}'

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TOTAL_GPUS - 1)))
export MAX_PIXELS MIN_PIXELS
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export TOKENIZERS_PARALLELISM=false

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "[$(date --iso-8601=seconds)] Starting smoke test: sample_num=$SAMPLE_NUM, total_gpus=$TOTAL_GPUS, gpus=$CUDA_VISIBLE_DEVICES"
"$python_bin" tools/ray_generate_hidden_for_draft_model.py \
    --modal_type VLM \
    --dataset_path "$TRAIN_DATASET_PATH" \
    --target_model_name_or_path "$TARGET_MODEL_PATH" \
    --draft_model_config_path "$DRAFT_MODEL_CONFIG_PATH" \
    --target_backend vllm \
    --torch_dtype bfloat16 \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --outdir "$OUTPUT_DIR" \
    --num_proc 8 \
    --sample_num "$SAMPLE_NUM" \
    --tensor_parallel_size "$TP_SIZE" \
    --total_gpus "$TOTAL_GPUS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --limit_mm_per_prompt "$LIMIT_MM_PER_PROMPT"
echo "[$(date --iso-8601=seconds)] Smoke test finished."
