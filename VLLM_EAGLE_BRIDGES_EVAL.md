# vLLM Eagle3 Bridges + Updated Run B Eval Implementation

This document outlines the vLLM changes needed to support early-exit bridges and progressive_fc_draft_feedback eval.

## Changes Required

### 1. Add EarlyExitBridge class to `third_party/vllm/vllm/model_executor/models/llama_eagle3.py`

Add this before the `LlamaDecoderLayer` class definition (around line 40):

```python
class EarlyExitBridge(nn.Module):
    """Residual MLP: approx_h_next = h + w2(act(w1(norm(h)))).
    
    w2 is zero-initialized so the bridge starts as pure identity.
    """

    def __init__(
        self, hidden_size: int, intermediate_size: int, hidden_act: str, rms_norm_eps: float
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]
        nn.init.zeros_(self.w2.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.w2(self.act_fn(self.w1(self.norm(h))))
```

Need to import `ACT2FN`:
```python
from transformers.activations import ACT2FN
```

### 2. Update `LlamaModel.__init__` to add bridge support

After the progressive_fc initialization block (around line 535), add:

```python
        # Early-exit bridges: residual MLPs for approximating deeper layers
        self.early_exit_bridges = bool(
            getattr(self.config, "early_exit_bridges", False)
        )
        self.bridges: nn.ModuleList | None = None
        if self.early_exit_bridges:
            if not self.progressive_staged:
                raise ValueError("early_exit_bridges requires progressive_staged mode")
            bridge_mid = int(
                getattr(self.config, "bridge_intermediate_size", None)
                or self.config.hidden_size
            )
            self.bridges = nn.ModuleList(
                [
                    EarlyExitBridge(
                        self.config.hidden_size,
                        bridge_mid,
                        self.config.hidden_act,
                        self.config.rms_norm_eps,
                    )
                    for _ in range(self.config.num_hidden_layers - 1)
                ]
            )
        
        # progressive_fc_draft_feedback: step 1+ mirrors step 0 via FC
        self.progressive_fc_draft_feedback = bool(
            getattr(self.config, "progressive_fc_draft_feedback", False)
        )
        if self.progressive_fc_draft_feedback and not self.progressive_per_layer_fc:
            raise ValueError(
                "progressive_fc_draft_feedback requires progressive_per_layer_fc=True"
            )
```

### 3. Update `Eagle3LlamaForCausalLM.take_progressive_draft_feedback()` 

Replace the entire method (around line 1110) with:

```python
    def take_progressive_draft_feedback(
        self,
        token_indices: torch.Tensor | None = None,
        num_tokens: int | None = None,
    ) -> torch.Tensor | None:
        """After a draft forward, build next-step feedback from draft layer outs.

        Default (Run B): sets per-layer injects to raw ``[h0, h1, h2, ...]``.

        When ``progressive_fc_draft_feedback=True`` (Updated Run B): mirrors
        step-0 by concatenating all draft layer outputs and projecting through
        the same per-layer FCs, so every layer sees all depths at step 1+.

        When ``early_exit_bridges`` is on: optionally uses bridges to approximate
        missing layers when a shallow exit is detected.

        Returns L0 seed: post-norm of the last layer out when ``norm_output``
        is set (EAGLE 3.1), otherwise ``h0``.
        """
        with angelslim_time_block(
            "progressive_draft_feedback",
            num_tokens=int(num_tokens or 0),
            has_token_indices=bool(token_indices is not None),
        ):
            outs = getattr(self.model, "_last_layer_outs", None)
            if not outs or self.model.disable_progressive_feedback:
                return None
            gathered: list[torch.Tensor] = []
            for h in outs:
                if token_indices is not None:
                    gathered.append(h[token_indices])
                elif num_tokens is not None:
                    gathered.append(h[:num_tokens])
                else:
                    gathered.append(h)
            
            # Bridge-based approximation: if bridges are available and gathered is
            # shallow (e.g., early exit), use bridges to approximate missing layers.
            # Optional: bridges are for training robustness; inference can also
            # just reuse the last layer.
            if (
                getattr(self.model, "early_exit_bridges", False)
                and self.model.bridges is not None
                and len(gathered) < len(self.model.layers)
            ):
                exit_depth = len(gathered) - 1
                h_cur = gathered[-1]
                for i in range(exit_depth, len(self.model.layers) - 1):
                    h_cur = self.model.bridges[i](h_cur)
                    gathered.append(h_cur)
            
            # Updated Run B: concat all and project through per-layer FCs
            if (
                getattr(self.model, "progressive_fc_draft_feedback", False)
                and self.model.progressive_fc is not None
            ):
                fc_norm = getattr(self.model, "fc_norm", None)
                if fc_norm is not None:
                    if len(fc_norm) != len(gathered):
                        raise RuntimeError(
                            f"progressive_fc_draft_feedback: fc_norm length "
                            f"({len(fc_norm)}) must equal number of draft "
                            f"layer outputs ({len(gathered)})"
                        )
                    normed = [norm(h) for norm, h in zip(fc_norm, gathered)]
                else:
                    normed = gathered
                draft_all = torch.cat(normed, dim=-1)
                fused = [fc(draft_all) for fc in self.model.progressive_fc]
                self.model.set_aux_inject(fused)
            else:
                # Run B: raw per-layer feedback
                self.model.set_aux_inject(gathered)
            
            if self.model.norm_output:
                return self.model.norm(gathered[-1])
            return gathered[0]
```

### 4. Update `prepare_draft_config_for_vllm_eval.py`

Add bridge config propagation after the progressive_per_layer_fc block (around line 146):

```python
        # progressive_fc_draft_feedback flag
        progressive_fc_draft_feedback = bool(
            cfg.get(
                "progressive_fc_draft_feedback",
                train_cfg.get("progressive_fc_draft_feedback", False),
            )
        )
        updates["progressive_fc_draft_feedback"] = progressive_fc_draft_feedback
        
        # early_exit_bridges and bridge configs
        early_exit_bridges = bool(
            cfg.get("early_exit_bridges", train_cfg.get("early_exit_bridges", False))
        )
        updates["early_exit_bridges"] = early_exit_bridges
        if early_exit_bridges:
            if not self.progressive_staged:
                raise ValueError("early_exit_bridges requires progressive_staged mode")
            bridge_mid = int(
                cfg.get(
                    "bridge_intermediate_size",
                    train_cfg.get("bridge_intermediate_size", config.hidden_size),
                )
            )
            updates["bridge_intermediate_size"] = bridge_mid
            # Propagate multi-depth CE weights for reference
            multi_depth_ce = cfg.get(
                "multi_depth_ce_weights",
                train_cfg.get("multi_depth_ce_weights", []),
            )
            if multi_depth_ce:
                updates["multi_depth_ce_weights"] = multi_depth_ce
```

Add to the logging output (around line 285):

```python
    if cfg.get("progressive_fc_draft_feedback"):
        print(f"  progressive_fc_draft_feedback: True")
    if cfg.get("early_exit_bridges"):
        print(f"  early_exit_bridges: True")
        print(f"    bridge_intermediate_size: {cfg.get('bridge_intermediate_size')}")
        if cfg.get("multi_depth_ce_weights"):
            print(f"    multi_depth_ce_weights: {cfg.get('multi_depth_ce_weights')}")
```

## Summary

These changes enable:

1. **Updated Run B eval**: symmetric FC feedback at step 1+ (fc_norm → concat → progressive_fc)
2. **Plan B with Bridges eval**: optional bridge-based approximation for early exits, plus multi-depth CE head support
3. **Config propagation**: all training flags automatically carried through to vLLM eval via `prepare_draft_config_for_vllm_eval.py`

At inference, both modes:
- Cascade with confidence thresholds on the shared lm_head
- Exit early if confident at any depth
- For bridges: optionally approximate missing layers; or simply reuse last available layer
- Propagate the last layer's hidden state as the L0 seed for next step

## Testing

After implementing, verify with:

```bash
python scripts/speculative/smolvlm/prepare_draft_config_for_vllm_eval.py \
    --draft_model outputs/smolvlm-256m-eagle3-updated-runb/checkpoint-5000 \
    --draft_model_config_path angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-draft-feedback.json \
    --dry_run

python scripts/speculative/smolvlm/prepare_draft_config_for_vllm_eval.py \
    --draft_model outputs/smolvlm-256m-eagle3-runb-bridges/checkpoint-5000 \
    --draft_model_config_path angelslim/compressor/speculative/train/configs/smolvlm-256m-eagle3-progressive-per-layer-fc-3.1-bridges.json \
    --dry_run
```

Both should show the respective flags enabled. Then run eval as normal.
