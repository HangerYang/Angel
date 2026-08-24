**Technical Field & Background**

Smart TV devices have severely limited resources: typically 1–4GB RAM along with extreme storage and power limitations. With advancements in AI and customer demand for this technology, running these models efficiently on-device becomes critical. Standard autoregressive decoding used by LLMs for text generation compounds this challenge, as each token requires a full model forward pass, and this per-token cost accumulates over the length of the generated sequence. This complexity intensifies with multimodal models that process both language and vision inputs, as each modality adds separate memory overhead and compute bottlenecks.

In this work, we present a multimodal speculative decoding framework that enables efficient inference for vision-language models on TV devices. Our approach runs a tiny draft model locally while offloading expensive verification to a server, or alternatively stacks two on-device models of different sizes to reduce per-token compute. Existing speculative decoding techniques largely target large language models; fewer works address large vision-language models, and none account for the memory, power, and connectivity constraints of on-device TV deployment.

Speculative decoding accelerates token generation by decoupling proposal from verification. A small draft model proposes $\gamma$ tokens cheaply and independently; the large target model then verifies all proposals in a single parallel forward pass, replacing multiple expensive sequential decoding steps. At verification, each proposed token is stochastically accepted based on the ratio of target to draft probabilities; the first rejection triggers sampling a correction token from the normalized residual distribution $\text{norm}(\max(0, p_{\text{target}} - p_{\text{draft}}))$. This rejection sampling is lossless: the final token distribution matches exactly what the target model would have produced alone, guaranteeing correctness despite the speedup. The efficiency win comes from amortizing the target model's cost across multiple candidates, reducing per-token latency (reported at up to 2–3× for text-only LLMs), with multimodal settings showing more variable gains depending on how vision tokens are handled. For resource-constrained devices like TVs, this makes LLM inference practical by minimizing expensive forward passes.


**Problem**

Static Hidden-State Fusion: Draft models reconstruct target hidden states by fusing a small, fixed set of layers (e.g., low, mid, and high), assuming that representations distributed across depth provide complementary information. We hypothesize that sparsely sampling only a few fixed points creates an information bottleneck, as intermediate target layers contain predictive signals that cannot be fully recovered from a static subset. This bottleneck is compounded in VLMs, where vision and language pathways interact through cross-attention and fusion layers; because the depth carrying the relevant signal shifts dynamically with token modality and position, a fixed layer selection is even less likely to capture it.

Single-Path Supervision: Draft models are typically trained on target-generated trajectories, receiving supervision only along a small set of realized continuations. At inference, however, speculative decoding can accept alternative tokens that are also well supported by the target distribution. As a result, plausible branches that are rarely or never observed during training receive little explicit supervision, creating a mismatch between reproducing target trajectories during training and maximizing target acceptance during drafting.

Visual Token Cost: Vision tokens dominate prefill cost and KV cache footprint, but the speedup from speculative decoding comes entirely from parallelizing language token verification. Vision tokens must still be encoded and processed at full computational cost regardless of draft-target agreement.

Weak Visual Grounding: Small draft models lack the capacity to learn reliable cross-attention over image regions, so their token proposals are not grounded in the same visual evidence the target attends to. This produces systematic disagreement on vision-dependent tokens specifically, where high acceptance rates matter most.

On-Device Parameter Duplication: Speculative decoding requires holding both draft and target parameters resident in memory simultaneously. On TV-class hardware with 1–4GB RAM, even a lightweight draft head adds a persistent memory footprint that competes directly with the target model's limited memory budget.

# Novelty Statements

**[Novelty 1]: Learnable Multi-Layer Fusion** - Use hidden states from a broader set of target-model layers instead of selecting only a few fixed layers. The selected layers are grouped into low-, middle-, and high-level representations, with learnable weights used to combine the layers within each group. These weights are learned jointly during training, allowing the model to determine which target layers provide the most useful information for drafting while keeping the drafter lightweight.

**[Novelty 2]: Branch-Aware Distillation** - Extend standard distillation on teacher-generated trajectories with additional supervision on plausible branches produced by the drafter. When the drafter predicts a token that differs from the teacher’s top prediction but is still highly ranked by the teacher, the target model is evaluated from this alternative branch and its subsequent distribution is used as an additional training signal. This exposes the drafter to likely off-trajectory states that can occur during speculative decoding.

**[Novelty 3]: Learned-Query Hidden-State Compression** - Introduce a single learned-query cross-attention compressor that reduces the number of visual hidden-state rows passed to the drafter. For each image tile, a small set of learned queries attends over the tile’s visual rows and produces a compact set of summary rows. A learned mixture over the auxiliary target-layer streams is used to determine the reference representation for computing the attention weights, and the same weights are then applied across the corresponding hidden states from all selected target layers. This compresses the visual portion of the target hidden-state sequence while preserving aligned information across layers, without introducing a multi-layer Q-Former or a separate visual encoder.

**[Novelty 4]: Visual Attention Distillation** - Add an auxiliary attention-distillation objective that encourages the drafter to match the target model’s attention over visual tokens. In addition to matching the target’s output distribution, the drafter is therefore supervised to attend to similar visual evidence when generating tokens, reducing disagreement caused by different visual grounding between the draft and target models.

**[Novelty 5]: LoRA-Based Lightweight Drafter** - Reuse a transformer layer from the existing VLM as the basis of the EAGLE draft layer and adapt it using LoRA instead of maintaining a separately parameterized transformer block. The LoRA parameters are only activated when speculative decoding is used, providing additional drafting capability with a small number of extra parameters and a low memory overhead suitable for on-device deployment.


# Prior Works

**EAGLE-3 — Scaling up Inference Acceleration of Large Language Models via Training-Time Test.**
Summary: EAGLE-3 is a speculative decoding method for LLMs that trains a lightweight draft model using hidden states from the target model. Compared with earlier EAGLE variants, it replaces feature prediction with direct token prediction and uses hidden states from multiple target layers, specifically low-, middle-, and high-level representations, to provide richer information to the drafter. It also introduces training-time testing to better expose the drafter to its own predicted states during training.

Learnable Multi-Layer Fusion: EAGLE-3 uses a small fixed set of target layers and concatenates their hidden states. Our method uses a broader set of layers and learns weighted combinations of multiple layers within low-, middle-, and high-level groups before fusion.

Branch-Aware Distillation: EAGLE-3's training-time test exposes the drafter to its own predicted states, but does not explicitly substitute plausible alternative draft tokens into the sequence, rerun the target model from those branches, and distill the resulting post-branch target distribution.

Learned-Query Hidden-State Compression: EAGLE-3 does not reduce the number of visual hidden-state rows using learned-query cross-attention or shared compression weights across multiple target-layer streams.

Visual Attention Distillation: EAGLE-3 does not explicitly train the drafter to match the target model's attention distribution over visual tokens.

LoRA-Based Lightweight Drafter: EAGLE-3 uses a separately trained draft model. Our approach instead reuses an existing VLM transformer layer and adapts it using LoRA, reducing the additional parameter and memory cost of the drafter.



**Accelerating Vision-Language Models with Vision-Aware Speculative Decoding.**

Summary: ViSpec adapts speculative decoding specifically to VLMs by addressing the difficulty of processing long visual-token sequences with a lightweight draft model. It introduces a vision adaptor that uses learnable queries to compress the original visual embeddings into a smaller set of visual tokens before they are processed by the drafter. ViSpec also extracts a global visual feature from the compressed representation and injects it into subsequent text hidden states so that the drafter retains access to visual context during generation. Its training procedure further incorporates sampled target outputs and draft-model hidden states to improve robustness during multi-token prediction.

Learnable Multi-Layer Fusion: ViSpec primarily uses the target model’s last-layer hidden state as target-aware information. Our method instead collects hidden states from a broader set of target layers and learns weighted combinations within low-, middle-, and high-level layer groups.

Branch-Aware Distillation: ViSpec uses sampling and draft hidden-state feedback to improve self-correction during training, but does not explicitly identify plausible draft-token disagreements, substitute those tokens into the sequence, rerun the target model on the resulting branch, and distill the target’s post-branch distribution.

Learned-Query Hidden-State Compression: ViSpec uses learnable queries to compress visual embeddings from the vision encoder before they enter the draft model. Our compressor instead operates on the target model’s intermediate visual hidden states and uses shared attention weights to compress corresponding visual rows across multiple target layers.

Visual Attention Distillation: ViSpec improves access to visual information through compressed visual embeddings and global-feature injection, but does not explicitly supervise the drafter to match the target model’s attention distribution over visual tokens.

LoRA-Based Lightweight Drafter: ViSpec introduces a separately parameterized draft model together with an additional vision adaptor. Our approach instead reuses an existing VLM transformer layer and adapts it using LoRA to reduce the incremental parameter and memory footprint of speculative drafting.


**Hiding Visual Tokens from the Drafter for Speculative Decoding in Vision-Language Models.**

Summary: HiViS is a VLM-specific speculative decoding framework that removes visual tokens from the drafter entirely to reduce the large prefill and attention cost introduced by image tokens. Instead of requiring the drafter to process the original visual-token sequence, HiViS uses the target VLM as a semantic fusion model and transfers multimodal information to the drafter through target-model hidden representations. It further introduces a time-step-aware aligned training scheme with self-feedback so that the drafter can maintain and refine the transferred visual-text information during multi-token drafting.

Learnable Multi-Layer Fusion: HiViS uses the target VLM’s last-layer hidden states to transfer multimodal information to the drafter. Our method instead uses hidden states from multiple target layers and learns weighted combinations within low-, middle-, and high-level layer groups, allowing the drafter to incorporate complementary information distributed across the target model’s layers.

Branch-Aware Distillation: HiViS uses time-step-aware aligned training to reduce hidden-state mismatch as the drafter performs multiple decoding steps without target feedback. Our method instead introduces explicit supervision on alternative token trajectories: when the drafter produces a plausible non-top-1 target token, we condition the target model on that token and distill its subsequent prediction distribution.

Learned-Query Hidden-State Compression: HiViS eliminates visual tokens from the drafter and transfers their information implicitly through fused target representations. Our method takes a different approach: it retains visual hidden-state information but compresses the visual rows with learned-query cross-attention, using shared compression weights across corresponding hidden states from multiple target layers.

Visual Attention Distillation: HiViS transfers visual-text semantics through target hidden representations but does not explicitly supervise the drafter to reproduce the target model’s attention distribution over visual tokens. Our method adds this direct attention-level supervision to encourage consistent visual grounding between the drafter and verifier.

LoRA-Based Lightweight Drafter: HiViS reduces drafting cost primarily by removing visual tokens from the drafter. Our approach additionally reduces the parameter overhead of the drafter itself by reusing an existing VLM transformer layer and adapting it with LoRA rather than introducing a fully separate draft layer.
