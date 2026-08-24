#!/usr/bin/env python3
"""Apply AngelSlim oracle-gist vLLM edits after syncing third_party/vllm.

The script is intentionally conservative: it is idempotent, anchor based, and
fails before writing when the expected upstream text is not found.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
VLLM = ROOT / "vllm" / "vllm"
PATCHES = ROOT / "patches"


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise SystemExit(f"anchor not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1))
    return True


def insert_after(path: Path, anchor: str, snippet: str) -> bool:
    text = path.read_text()
    if snippet in text:
        return False
    if anchor not in text:
        raise SystemExit(f"anchor not found in {path}: {anchor[:120]!r}")
    path.write_text(text.replace(anchor, anchor + snippet, 1))
    return True


def write_eagle_gist() -> bool:
    dst = VLLM / "model_executor" / "models" / "eagle_gist.py"
    src = PATCHES / "eagle_gist.py.txt"
    content = src.read_text()
    if dst.exists() and dst.read_text() == content:
        return False
    dst.write_text(content)
    return True


def patch_llama_eagle3() -> int:
    path = VLLM / "model_executor" / "models" / "llama_eagle3.py"
    n = 0
    n += insert_after(path, "from . import eagle_miracle\n", "from . import eagle_gist\n")
    n += insert_after(
        path,
        """        qkv_input_size = (
            (1 + self.n_aux_bands) * self.hidden_size
            if (self.banded_mix_wide and layer_idx == 0)
            else self.hidden_size
            if self.hawk
            else (
                2 * self.hidden_size
                if (
                    layer_idx == 0
                    or self.progressive_staged
                    or self.per_layer_weighted_sum
                )
                else self.hidden_size
            )
        )
""",
        """        # Oracle gist qkv/both: layer 0 takes [embed | aux | gist].
        self.gist_qkv_stream = (
            layer_idx == 0
            and bool(getattr(config, "gist_conditioning", False))
            and str(getattr(config, "gist_injection", "qkv")) in ("qkv", "both")
        )
        if self.gist_qkv_stream:
            qkv_input_size += self.hidden_size
""",
    )
    n += insert_after(
        path,
        "        self.layer_idx = layer_idx\n",
        """        self.gist_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.gist_qkv_stream
            else None
        )
        self.gist_projector = None
""",
    )
    n += replace_once(
        path,
        """        elif self.layer_idx == 0:
            # First layer: concatenate embeds with hidden_states
            embeds = self.input_layernorm(embeds)
            hidden_states, residual = self._residual_norm(hidden_states=hidden_states)
            hidden_states = torch.cat([embeds, hidden_states], dim=-1)
""",
        """        elif self.layer_idx == 0:
            embeds = self.input_layernorm(embeds)
            hidden_states, residual = self._residual_norm(hidden_states=hidden_states)
            streams = [embeds, hidden_states]
            if self.gist_qkv_stream:
                streams.append(self._gist_stream(hidden_states))
            hidden_states = torch.cat(streams, dim=-1)
""",
    )
    n += insert_after(
        path,
        '''    def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
        """Use drafter's quantization config instead of verifier's."""
        return get_draft_quant_config(vllm_config)
''',
        """
    def _gist_stream(self, like: torch.Tensor) -> torch.Tensor:
        \"\"\"Projected+normed oracle gist for this step, zeros when unarmed.\"\"\"
        num_rows = int(like.shape[0])
        gist = eagle_gist.current(num_rows, like.device, like.dtype)
        if gist is None or self.gist_projector is None or self.gist_norm is None:
            return torch.zeros_like(like)
        return self.gist_norm(self.gist_projector(gist))
""",
    )
    n += insert_after(
        path,
        """        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

""",
        """        if self.gist_conditioning:
            self.gist_projector = ReplicatedLinear(
                input_size=self.gist_embedding_dim,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=current_vllm_config.model_config.dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "gist_projector"),
                return_bias=False,
            )
            if self.gist_fc_stream:
                self.gist_stream_norm = RMSNorm(
                    self.config.hidden_size, eps=self.config.rms_norm_eps
                )

""",
    )
    n += insert_after(
        path,
        """        self.layers = nn.ModuleList(
""",
        """            # gist_projector is shared with layer 0 for qkv/both injection.
""",
    )
    n += replace_once(
        path,
        """        )
        if self.use_aux_hidden_state:
""",
        """        )
        if self.gist_qkv_stream:
            self.layers[0].gist_projector = self.gist_projector
        if self.use_aux_hidden_state:
""",
    )
    n += insert_after(
        path,
        """        if self.model.norm_before_fc:
            hidden_states = self.model.input_norm(hidden_states)

""",
        """        if getattr(self.model, "gist_fc_stream", False):
            gist = eagle_gist.current(
                int(hidden_states.shape[0]), hidden_states.device, hidden_states.dtype
            )
            if gist is None or self.model.gist_projector is None:
                gist_stream = torch.zeros(
                    (hidden_states.shape[0], self.model.config.hidden_size),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            else:
                gist_stream = self.model.gist_projector(gist)
            hidden_states = torch.cat((hidden_states, gist_stream), dim=-1)

""",
    )
    n += insert_after(
        path,
        """        if self.model.fc_norm is not None:
            chunks = hidden_states.chunk(len(self.model.fc_norm), dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.model.fc_norm, chunks)],
                dim=-1,
            )

""",
        """        if getattr(self.model, "gist_fc_stream", False):
            gist = eagle_gist.current(
                int(hidden_states.shape[0]), hidden_states.device, hidden_states.dtype
            )
            if gist is None or self.model.gist_projector is None:
                gist_stream = torch.zeros(
                    (hidden_states.shape[0], self.model.config.hidden_size),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            else:
                gist_stream = self.model.gist_projector(gist)
            if self.model.gist_stream_norm is not None:
                gist_stream = self.model.gist_stream_norm(gist_stream)
            hidden_states = torch.cat((hidden_states, gist_stream), dim=-1)

""",
    )
    n += replace_once(
        path,
        """        # Training registers gist_norm on every draft layer even when gist
        # conditioning is disabled. It is unused by B/C and has no vLLM peer.
        skip_substrs.append("gist_norm.")
""",
        """        if not getattr(self.model, "gist_qkv_stream", False):
            skip_substrs.append("gist_norm.")
        if not getattr(self.model, "gist_conditioning", False):
            skip_substrs.append("gist_projector.")
        if not getattr(self.model, "gist_fc_stream", False):
            skip_substrs.append("gist_stream_norm.")
        if getattr(self.model, "gist_fc_stream", False):
            for required in ("model.gist_projector.weight", "model.gist_stream_norm.weight"):
                if required not in model_weights:
                    raise ValueError(f"gist_injection='fc' draft is missing {required}")
""",
    )
    return n


def patch_speculator() -> int:
    path = VLLM / "v1" / "worker" / "gpu" / "spec_decode" / "autoregressive" / "speculator.py"
    n = 0
    n += insert_after(
        path,
        "from vllm.model_executor.models import eagle_miracle\n",
        "from vllm.model_executor.models import eagle_gist\n",
    )
    n += insert_after(
        path,
        """            self._maybe_compress_visual_rows(input_batch, _aux_concat)
            hidden_states = self.model.combine_hidden_states(_aux_concat)
""",
        """            if not dummy_run and eagle_gist.is_enabled():
                eagle_gist.arm(
                    req_ids=list(input_batch.req_ids),
                    query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
                    num_tokens=int(input_batch.num_tokens),
                    num_reqs=int(num_reqs),
                    device=self.hidden_states.device,
                    dtype=self.hidden_states.dtype,
                )
""",
    )
    return n


def main() -> None:
    changes = 0
    changes += write_eagle_gist()
    changes += patch_llama_eagle3()
    changes += patch_speculator()
    print(f"oracle gist vLLM patch applied; changed {changes} section(s)")


if __name__ == "__main__":
    main()
