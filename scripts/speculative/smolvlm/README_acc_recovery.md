# SmolVLM Speculative Acc Recovery

This note records the end-to-end workflow used to compare AngelSlim training
accuracy with a no-teacher-forcing offline rollout for SmolVLM Eagle3 variants.

The recovery helper is:

```bash
tools/recover_hf_train_acc_smolvlm_hawk.py
```

Despite the historical file name, the helper supports:

- stock Eagle3, `eagle_aux_injection_mode=fused_fc`
- progressive Eagle, `eagle_aux_injection_mode=progressive_staged`
- Hawk, `eagle_aux_injection_mode=hawk`

Real-hawk / LoRA checkpoints currently need a loader fix before using this
helper. A smoke run showed missing/unexpected LoRA/base weights when loading an
unmerged real-hawk checkpoint, so do not trust real-hawk numbers from this
helper until that is fixed.

## Future SmolVLM Config Convention

The train configs in this repo now use:

```json
"aux_hidden_states_layer_ids": [1, 14, 26],
"eagle_aux_hidden_state_layer_ids": [2, 15, 27],
"draft_layer_init_from_target": [2, 15, 27]
```

For the progressive uninitialized config, `draft_layer_init_from_target` is
intentionally absent.

## Train A Progressive Eagle Smoke Checkpoint

Use a separate output directory so an existing run is not resumed or overwritten.
The command below trains 1000 optimizer steps and saves `checkpoint-1000`.

```bash
PYTHONPATH=/home/hyang/AngelSlim \
CUDA_VISIBLE_DEVICES=2 \
/home/hyang/miniconda3/envs/angel/bin/python tools/train_eagle3_online.py \
  --modal_type VLM \
  --target_model_name_or_path HuggingFaceTB/SmolVLM-256M-Instruct \
  --draft_model_config_path angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive.json \
  --train_data_path dataset/smolvlm_256m_target_gen_mixed_70k70k/train.jsonl \
  --output_dir output/smolvlm_256m_eagle3_progressive_smoke_1k \
  --num_train_epochs 2 \
  --max_steps 1000 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_proc 1 \
  --load_from_cache_file true \
  --save_strategy steps \
  --save_steps 1000 \
  --eval_strategy no \
  --eval_steps 5000 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type constant \
  --logging_steps 100 \
  --model_max_length 4096 \
  --embed_weight_key model.text_model.embed_tokens.weight \
  --chat_template_type smolvlm \
  --bf16 \
  --report_to none \
  --run_name smolvlm-256m-progressive-smoke-1k \
  --sample_num 4096
```

Expected branch log:

```text
Eagle3 progressive_staged: target HS only on first draft token; steps 1+ use draft outs only
```

## Recover Teacher Acc And Free-Rollout Acc

For an existing Hawk checkpoint:

```bash
PYTHONPATH=/home/hyang/AngelSlim \
CUDA_VISIBLE_DEVICES=0 \
/home/hyang/miniconda3/envs/angel/bin/python tools/recover_hf_train_acc_smolvlm_hawk.py \
  --checkpoint output/smolvlm_256m_hawk_nccl/checkpoint-66466 \
  --sample-num 256 \
  --max-batches 256 \
  --batch-size 1 \
  --num-proc 1 \
  --load-from-cache-file \
  --rollout both \
  --print-every 64
```

For the progressive smoke checkpoint:

```bash
PYTHONPATH=/home/hyang/AngelSlim \
CUDA_VISIBLE_DEVICES=2 \
/home/hyang/miniconda3/envs/angel/bin/python tools/recover_hf_train_acc_smolvlm_hawk.py \
  --checkpoint output/smolvlm_256m_eagle3_progressive_smoke_1k/checkpoint-1000 \
  --sample-num 256 \
  --max-batches 256 \
  --batch-size 1 \
  --num-proc 1 \
  --load-from-cache-file \
  --rollout both \
  --print-every 128
```

Use `--rollout teacher` to reproduce the trainer metric only, `--rollout free`
for the no-teacher-forcing rollout only, and `--rollout both` for both tables.

## What The Helper Measures

Teacher mode calls the normal AngelSlim trainer loss path and reports
`train/acc_0` through `train/acc_6` on the selected dataset slice.

Free mode keeps the same target/logit/hidden-state preparation path, but removes
teacher forcing inside the draft rollout. After each speculative step, the next
draft input token is the draft argmax mapped back into the target vocabulary.
It reports:

- `free_acc_i`: per-position draft-vs-target argmax agreement after draft
  feedback.
- `free_acceptance_rate_pos_i`: prefix survival through position `i`.
- `free_mean_acceptance_length`: `1 + mean(number of accepted speculative tokens)`.

This is an offline training-tape proxy. It is useful for debugging training vs
rollout mismatch, but it is not a replacement for the full benchmark/vLLM
acceptance-length eval.

## Verified Smoke Results

Hawk nccl, `output/smolvlm_256m_hawk_nccl/checkpoint-66466`, 256 train samples:

| metric | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher acc | 0.6685 | 0.6488 | 0.6429 | 0.6373 | 0.6350 | 0.6272 | 0.6240 |
| free acc | 0.7040 | 0.5668 | 0.4748 | 0.4186 | 0.3766 | 0.3464 | 0.3190 |
| prefix acceptance | 0.7040 | 0.4859 | 0.3662 | 0.3001 | 0.2575 | 0.2260 | 0.2024 |

Free mean acceptance length: `3.5420`.

Stock Eagle3, `output/smolvlm_256m_eagle3_nccl/checkpoint-50000`, 256 train samples:

| metric | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher acc | 0.6472 | 0.6095 | 0.5904 | 0.5790 | 0.5697 | 0.5554 | 0.5406 |
| free acc | 0.6811 | 0.5096 | 0.4050 | 0.3339 | 0.2798 | 0.2294 | 0.1991 |
| prefix acceptance | 0.6811 | 0.4336 | 0.3001 | 0.2163 | 0.1623 | 0.1237 | 0.0956 |

Free mean acceptance length: `3.0128`.

Progressive Eagle smoke,
`output/smolvlm_256m_eagle3_progressive_smoke_1k/checkpoint-1000`, 256 train samples:

| metric | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher acc | 0.2246 | 0.2295 | 0.2288 | 0.2236 | 0.2201 | 0.2130 | 0.2067 |
| free acc | 0.2247 | 0.1131 | 0.0850 | 0.0717 | 0.0656 | 0.0572 | 0.0531 |
| prefix acceptance | 0.2247 | 0.0694 | 0.0262 | 0.0039 | 0.0023 | 0.0006 | 0.0003 |

Free mean acceptance length: `1.3274`.
