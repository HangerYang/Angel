# AI Researcher Interview Prep

Each notebook: **Lesson → Implementation (TODOs) → Quiz → Final Answers & Explanations**.
Try to fill in the TODOs and answer the quiz yourself before reading the final section — the
sanity-check cells will tell you if your implementation is right.

## 02–05: broad industry-trend set (2026 interview research)

These four came from checking what's actually being asked in AI/ML interviews right now, rather
than just what's specific to your own project work.

2. `02_rope.ipynb` — Rotary Position Embeddings from scratch. The rotate-half construction, why
   $Q_i \cdot K_j$ ends up a function of relative position only, why Q/K but not V, and why naive
   context-length extrapolation breaks (+ what position interpolation / NTK scaling do about it).
3. `03_lora_peft.ipynb` — LoRA from scratch. The $W + \frac{\alpha}{r}BA$ decomposition, why $B$
   is zero-initialized (same trick as Flamingo's gate in notebook 1), merging for zero-latency
   inference vs. multi-adapter serving, and a QLoRA pointer.
4. `04_moe_routing.ipynb` — Mixture-of-Experts routing. Router + top-k dispatch, the
   load-balancing auxiliary loss (and what expert collapse looks like without it), capacity/token
   dropping, and why MoE saves compute but not memory (ties back to notebook 07's KV-cache
   framing).
5. `05_vit_patch_embed_projector.ipynb` — ViT patch embedding (+ the conv-equivalence gotcha) and
   the two VLM projector families: MLP-style (LLaVA, token count scales with resolution) vs.
   resampler/Q-Former-style (fixed token count, fixed information bottleneck) — a direct
   extension of notebook 1's cross-attention material.

## 01 and 06–07: your specialty deep dive

1. `01_cross_attention.ipynb` — cross-attention from scratch (Q from one sequence, K/V from
   another). Flamingo-style gated cross-attention vs. LLaVA/Qwen2.5-VL-style token concatenation,
   masking, complexity, and KV-cache implications for frozen vision encoders.
6. `06_speculative_decoding.ipynb` — draft/verify/accept-reject from scratch, with a statistical
   proof-by-simulation that the output distribution exactly matches the target model regardless
   of draft quality. Covers speedup math, tree-based drafting, and EAGLE/hidden-state-style
   multimodal drafting — closest to your own area of work.
7. `07_attention_kv_cache_basics.ipynb` — causal self-attention with an incremental KV cache,
   verified to exactly match a full forward pass. Covers MHA vs. GQA vs. MQA, cache memory
   footprint, prefill vs. decode, and cache rollback under speculative rejection.

## Suggested order

**02 → 03 → 04 → 05 → 01 → 06 → 07** — front-loads the broad, currently-popular topics an
interviewer at any AI lab is likely to ask regardless of team, then moves into cross-attention
and finally your own specialty (VLMs + speculative decoding), where you should already be
strongest. Feel free to reorder — every notebook is self-contained.

More notebooks can be added the same way if you want to keep going (e.g. quantization basics,
DPO/GRPO post-training objectives, RoPE extrapolation methods in depth, diffusion vs.
autoregressive generation).
