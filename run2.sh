for DS in lmms-lab/textvqa HuggingFaceH4/MATH-500 Lin-Chen/MMStar; do
  CUDA_VISIBLE_DEVICES=1 \
  EARLY_EXIT_THRESHOLD=0.8 \
  DRAFT_MODEL=output/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1/checkpoint-66466 \
  DRAFT_MODEL_CONFIG_PATH=angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1.json \
  DATASET=$DS \
  OUTPUT_FILE=output/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1/checkpoint-66466/eval/early_exit_0.8/$(basename $DS)/results.jsonl \
  bash scripts/speculative/smolvlm/eval_eagle3_vlm_batch.sh
done