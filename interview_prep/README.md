# AI Researcher Interview Prep

Each notebook: **Lesson → Implementation (TODOs) → Quiz → Final Answers & Explanations**.
Try to fill in the TODOs and answer the quiz yourself before reading the final section — the
sanity-check cells will tell you if your implementation is right.

1. `01_cross_attention.ipynb` — cross-attention from scratch (Q from one sequence, K/V from
   another). Covers Flamingo-style gated cross-attention vs. LLaVA/Qwen2.5-VL-style token
   concatenation, masking, complexity, and KV-cache implications for frozen vision encoders.
2. `02_speculative_decoding.ipynb` — draft/verify/accept-reject from scratch, with a statistical
   proof-by-simulation that the output distribution exactly matches the target model regardless
   of draft quality. Covers speedup math, tree-based drafting, and EAGLE/hidden-state-style
   multimodal drafting (closest to your own area).
3. `03_attention_kv_cache_basics.ipynb` — causal self-attention with an incremental KV cache,
   verified to exactly match a full forward pass. Covers MHA vs. GQA vs. MQA, cache memory
   footprint, prefill vs. decode, RoPE on Q/K but not V, and cache rollback under speculative
   rejection.

Suggested order: 3 → 1 → 2 (basics, then VLM-specific, then your specialty) if you want the
easiest ramp-up, or 1 → 2 → 3 to front-load what's most likely to come up first in a VLM/decoding
role. More notebooks can be added the same way if you want to keep going (e.g. RoPE from scratch,
LoRA/PEFT, quantization basics, RLHF/DPO objectives, vision encoders/patch embeddings).
