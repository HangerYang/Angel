cd /home/hyang/AngelSlim
conda activate angel


TRAIN_MODE=nccl \
NPROC=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-banded-mix-uninit.json \
OUTPUT_DIR=output/smolvlm-256m-eagle3-progressive-banded-mix-uninit \
NUM_TRAIN_EPOCHS=2 \
SAVE_STRATEGY=epoch \
EVAL_STRATEGY=epoch \
LOAD_FROM_CACHE_FILE=true \
TARGET_HS_WARMUP_STEPS=0 \
bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh



# Phase 1 (plan: peaceful-dancing-prism.md) -- progressive_staged, but every
# draft layer gets its own independent 3H->H FC over ALL aux streams (instead
# of the raw one-stream-per-layer inject above), plus EAGLE 3.1 fc_norm +
# norm_output. Uninit base (no draft_layer_init_from_target), so this isolates
# the new FC routing cleanly -- see smolvlm-256m-eagle3-progressive-per-layer-fc-3.1.json.
# Uses the same mixed_70k70k dataset already gist-prewarmed earlier, so this
# should hit warm .map_cache immediately (no gist_conditioning in this config).
TRAIN_MODE=nccl \
NPROC=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1.json \
OUTPUT_DIR=output/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1 \
NUM_TRAIN_EPOCHS=2 \
SAVE_STRATEGY=epoch \
EVAL_STRATEGY=epoch \
LOAD_FROM_CACHE_FILE=true \
TARGET_HS_WARMUP_STEPS=0 \
bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh


DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-3.1-oracle-remaining-gist-r7.json \
TRAIN_DATA_PATH=/home/hyang/AngelSlim/dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl \
EVAL_DATA_PATH=/home/hyang/AngelSlim/dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl \
OUTPUT_DIR=output/smolvlm_256m_eagle3_oracle_gist_r7_2ep \
TRAIN_MODE=nccl \
NUM_TRAIN_EPOCHS=2 \
SAVE_STRATEGY=epoch \
EVAL_STRATEGY=epoch \
LOAD_FROM_CACHE_FILE=true \
bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh


# Phase 2 (plan: peaceful-dancing-prism.md) -- Medusa-style MTP heads (K=3:
# predict t+1 as before, plus 2 new heads for t+2/t+3), frozen trunk, trained
# only on the new heads. Ported to the ONLINE pipeline (no offline .ckpt dump
# needed) -- reuses the exact same train_eagle3_online.py dataset path, so it
# hits the already-warm .map_cache. Checkpoint below is Block B's completed
# run (progressive_banded_mix, epoch 2.0, eval_loss 1.93).
#
# Also fixes a real bug found while wiring this up: LlamaRotaryEmbedding's
# non-persistent inv_freq/cos_cached/sin_cached buffers come back as
# uninitialized garbage memory after Eagle3LlamaForCausalLM.from_pretrained
# (HF's low_cpu_mem_usage load path doesn't rerun their __init__-time
# computation) -- this script force-recomputes them after load
# (_refresh_rope_caches). Confirmed fixed: real losses + up to ~45% per-step
# accuracy on a 32-sample smoke run, vs. NaN loss before the fix.
CUDA_VISIBLE_DEVICES=0 \
python tools/train_eagle3_mtp_heads.py \
  --draft_checkpoint_path output/smolvlm-256m-eagle3-progressive-banded-mix-uninit/checkpoint-66466 \
  --target_model_name_or_path HuggingFaceTB/SmolVLM-256M-Instruct \
  --train_data_path dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl \
  --eval_data_path dataset/smolvlm_256m_target_gen_mixed_70k70k/eval.jsonl \
  --num_extra_heads 2 \
  --num_train_epochs 1 \
  --logging_steps 50 \
  --load_from_cache_file true \
  --output_dir output/smolvlm-256m-eagle3-mtp-heads