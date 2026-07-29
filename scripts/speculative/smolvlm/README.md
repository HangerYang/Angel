# SmolVLM speculative data generation

Target: `HuggingFaceTB/SmolVLM-256M-Instruct`

## Recommended: dp=4 (4 GPU replicas)

```bash
GPU_NUM=4 BASE_PORT=6000 bash scripts/speculative/smolvlm/run_vllm_server.sh

MAX_CLIENTS=4 NUM_THREADS=32 bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
```

Starts vLLM on ports **6000–6003** (one replica per GPU); the generator load-balances across them.

---

## 1. Start vLLM server

```bash
bash scripts/speculative/smolvlm/run_vllm_server.sh
# defaults: GPU_NUM=4 BASE_PORT=6000
# override e.g. GPU_NUM=1 MODEL_LOCAL_PATH=HuggingFaceTB/SmolVLM-256M-Instruct
```

## 2. Generate target-model samples

Uses `dataset/mixed_text_vl_36/mixed_text_vl_36.jsonl` (`openai_vl` format) by default.

```bash
bash scripts/speculative/smolvlm/generate_data_for_target_model.sh
# defaults: MAX_CLIENTS=4 NUM_THREADS=32
# optional: DATA_NAME_OR_PATH=... OUTPUT_DIR=... MAX_TOKENS=512
```

Output: `dataset/smolvlm_256m_target_gen/data_*.jsonl`

## 3. Online Eagle3 training

```bash
bash scripts/speculative/smolvlm/train_eagle3_vlm_online.sh
# defaults: 1 GPU, 1 epoch on dataset/smolvlm_256m_target_gen/data_0-36.jsonl
# optional: CUDA_VISIBLE_DEVICES=0,1 NPROC=2 SAMPLE_NUM=8 OUTPUT_DIR=...
```

Draft config: `angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3.json`
(`target_model_type: smolvlm`, also accepts `idefics3`).
