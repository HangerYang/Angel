#!/bin/bash
# Quick smoke test: verify no crashes on minimal runs
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TARGET_MODEL="HuggingFaceTB/SmolVLM-256M-Instruct"
TRAIN_DATA="dataset/smolvlm_256m_target_gen/train.jsonl"
EVAL_DATA="dataset/smolvlm_256m_target_gen/eval.jsonl"

echo "================================================================================"
echo "SMOKE TEST: Updated Plan B (progressive_fc_draft_feedback)"
echo "================================================================================"

OUTPUT_UPDATED_B="outputs/smoke_updated_b"
rm -rf "$OUTPUT_UPDATED_B"
mkdir -p "$OUTPUT_UPDATED_B"

echo "Training 10 steps..."
python tools/train_eagle3_online.py \
    --modal_type VLM \
    --training_mode online \
    --target_model_name_or_path "$TARGET_MODEL" \
    --draft_model_config_path angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-draft-feedback.json \
    --train_data_path "$TRAIN_DATA" \
    --eval_data_path "$EVAL_DATA" \
    --chat_template_type smolvlm \
    --output_dir "$OUTPUT_UPDATED_B" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --max_steps 10 \
    --save_steps 10 \
    --eval_steps 10 \
    --logging_steps 1 \
    --bf16 \
    2>&1 | tail -30

echo ""
echo "Eval on 10 samples (temp=0)..."
LATEST_CKPT=$(find "$OUTPUT_UPDATED_B" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
if [ -z "$LATEST_CKPT" ]; then LATEST_CKPT="$OUTPUT_UPDATED_B"; fi

python tools/eval_smolvlm_eagle3_acceptance.py \
    --target_model "$TARGET_MODEL" \
    --draft_model "$LATEST_CKPT" \
    --draft_model_config_path angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-draft-feedback.json \
    --data_path "$EVAL_DATA" \
    --output_file "${OUTPUT_UPDATED_B}/eval_temp0.json" \
    --num_samples 10 \
    --num_spec_tokens 3 \
    --torch_dtype bfloat16 \
    --chat_template_type smolvlm \
    2>&1 | tail -20

echo "✓ Updated Plan B smoke test passed"

echo ""
echo "================================================================================"
echo "SMOKE TEST: Plan B with Bridges"
echo "================================================================================"

OUTPUT_BRIDGES="outputs/smoke_bridges"
rm -rf "$OUTPUT_BRIDGES"
mkdir -p "$OUTPUT_BRIDGES"

echo "Training 10 steps..."
python tools/train_eagle3_online.py \
    --modal_type VLM \
    --training_mode online \
    --target_model_name_or_path "$TARGET_MODEL" \
    --draft_model_config_path angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-bridges.json \
    --train_data_path "$TRAIN_DATA" \
    --eval_data_path "$EVAL_DATA" \
    --chat_template_type smolvlm \
    --output_dir "$OUTPUT_BRIDGES" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --max_steps 10 \
    --save_steps 10 \
    --eval_steps 10 \
    --logging_steps 1 \
    --bf16 \
    2>&1 | tail -30

echo ""
echo "Eval on 10 samples (temp=0)..."
LATEST_CKPT=$(find "$OUTPUT_BRIDGES" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
if [ -z "$LATEST_CKPT" ]; then LATEST_CKPT="$OUTPUT_BRIDGES"; fi

python tools/eval_smolvlm_eagle3_acceptance.py \
    --target_model "$TARGET_MODEL" \
    --draft_model "$LATEST_CKPT" \
    --draft_model_config_path angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-bridges.json \
    --data_path "$EVAL_DATA" \
    --output_file "${OUTPUT_BRIDGES}/eval_temp0.json" \
    --num_samples 10 \
    --num_spec_tokens 3 \
    --torch_dtype bfloat16 \
    --chat_template_type smolvlm \
    2>&1 | tail -20

echo "✓ Plan B with Bridges smoke test passed"

echo ""
echo "================================================================================"
echo "✓ ALL SMOKE TESTS PASSED"
echo "================================================================================"
echo "Results:"
echo "  Updated Plan B: $OUTPUT_UPDATED_B"
echo "  Plan B Bridges: $OUTPUT_BRIDGES"
