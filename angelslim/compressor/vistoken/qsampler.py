"""Q-Sampler: compress a SmolVLM tile's visual tokens from 64 to num_queries.

SmolVLM (Idefics3) emits 64 tokens *per tile* after pixel shuffle, and image
splitting is on, so a real prompt carries 13-17 tiles (832-1088 image tokens).
The sampler runs per tile with shared weights, so the ``<row_i_col_j>`` grid
markers around each tile stay valid and only the run length inside each tile
changes: 64 -> N.

Architecture: N learned queries attend over the tile's 64 keys through a stack
of (cross-attention -> MLP) blocks, pre-norm with residuals. The 64 keys carry a
learned 8x8 2D positional embedding -- without it the queries cannot recover
spatial layout, since pixel shuffle flattens an 8x8 grid into an unordered set
as far as attention is concerned.

Output lives in the LM's embedding space (hidden_size), i.e. it is a drop-in
replacement for ``Idefics3Connector`` output.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QSamplerConfig:
    hidden_size: int = 576
    num_queries: int = 4
    num_blocks: int = 1
    num_heads: int = 9
    # MLP(576 -> 1536 -> 576): mirrors SmolVLM's own text intermediate_size
    # rather than the usual 4x, keeping the sampler proportionate to a 256M model.
    mlp_hidden: int = 1536
    dropout: float = 0.0
    tile_tokens: int = 64  # 8x8 after pixel shuffle
    grid: int = 8
    init_std: float = 0.02
    # "mean_pool": block 0 starts as uniform average pooling over the tile and
    # later blocks start as identity, so the student begins as a sane summary of
    # the tile instead of noise. "random": plain init.
    init_mode: str = "mean_pool"
    # Idefics3's connector output has per-token RMS ~5.16 -- ~40x the text
    # embedding scale (0.129). A LayerNorm'd output would enter the LM 5x too
    # small and read as a near-blank image. out_norm's gain is filled with this
    # instead; ``calibrate_output_scale`` replaces it with the measured value.
    out_scale: float = 5.0

    def __post_init__(self):
        if self.grid * self.grid != self.tile_tokens:
            raise ValueError(
                f"tile_tokens ({self.tile_tokens}) must be grid^2 (grid={self.grid})"
            )
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must divide by num_heads "
                f"({self.num_heads})"
            )


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention (queries attend to tile tokens) + MLP."""

    def __init__(self, cfg: QSamplerConfig):
        super().__init__()
        h = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = h // cfg.num_heads
        self.q_norm = nn.LayerNorm(h)
        self.kv_norm = nn.LayerNorm(h)
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.mlp_norm = nn.LayerNorm(h)
        inner = cfg.mlp_hidden
        self.mlp = nn.Sequential(
            nn.Linear(h, inner),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(inner, h),
        )
        self.dropout = cfg.dropout

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: [T, N, H]   kv: [T, 64, H]
        T, N, H = q.shape
        S = kv.shape[1]
        qn = self.q_norm(q)
        kvn = self.kv_norm(kv)

        def split(x, L):
            return x.view(T, L, self.num_heads, self.head_dim).transpose(1, 2)

        a = F.scaled_dot_product_attention(
            split(self.q_proj(qn), N),
            split(self.k_proj(kvn), S),
            split(self.v_proj(kvn), S),
            dropout_p=self.dropout if self.training else 0.0,
        )
        a = a.transpose(1, 2).reshape(T, N, H)
        q = q + self.o_proj(a)
        q = q + self.mlp(self.mlp_norm(q))
        return q


class QSampler(nn.Module):
    """64 tile tokens -> ``num_queries`` tokens, shared across tiles."""

    def __init__(self, cfg: Optional[QSamplerConfig] = None, **kwargs):
        super().__init__()
        self.cfg = cfg or QSamplerConfig(**kwargs)
        c = self.cfg
        self.queries = nn.Parameter(torch.empty(c.num_queries, c.hidden_size))
        self.kv_pos = nn.Parameter(torch.empty(c.tile_tokens, c.hidden_size))
        self.blocks = nn.ModuleList(CrossAttentionBlock(c) for _ in range(c.num_blocks))
        self.out_norm = nn.LayerNorm(c.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        c = self.cfg
        nn.init.normal_(self.queries, std=c.init_std)
        nn.init.normal_(self.kv_pos, std=c.init_std)
        nn.init.ones_(self.out_norm.weight)
        nn.init.zeros_(self.out_norm.bias)
        with torch.no_grad():
            self.out_norm.weight.fill_(c.out_scale)
        if c.init_mode != "mean_pool":
            return
        with torch.no_grad():
            # Block 0 == uniform average pooling: zero query projection makes
            # every attention logit 0, so softmax is uniform over the 64 tile
            # tokens, and identity v/o projections pass that mean straight
            # through. The random `queries` still break symmetry between the N
            # slots, so they differentiate as soon as gradients arrive.
            b0 = self.blocks[0]
            nn.init.zeros_(b0.q_proj.weight)
            nn.init.eye_(b0.v_proj.weight)
            nn.init.eye_(b0.o_proj.weight)
            nn.init.zeros_(b0.mlp[-1].weight)
            nn.init.zeros_(b0.mlp[-1].bias)
            # Later blocks start as exact identity (zeroed residual branches),
            # so depth costs nothing at init and is learned into.
            for b in self.blocks[1:]:
                nn.init.zeros_(b.o_proj.weight)
                nn.init.zeros_(b.mlp[-1].weight)
                nn.init.zeros_(b.mlp[-1].bias)

    @torch.no_grad()
    def calibrate_output_scale(self, tile_features: torch.Tensor) -> float:
        """Match the output RMS to the real connector output on a real batch.

        Removes the hard-coded 5.0: the student's image tokens then enter the LM
        at the same scale the frozen backbone was trained to expect.
        """
        target = tile_features.float().pow(2).mean(-1).sqrt().mean()
        cur = self(tile_features).float().pow(2).mean(-1).sqrt().mean()
        self.out_norm.weight.mul_(float(target / cur.clamp_min(1e-6)))
        return float(target)

    @property
    def num_queries(self) -> int:
        return self.cfg.num_queries

    def forward(self, tile_features: torch.Tensor) -> torch.Tensor:
        """``[T, 64, H]`` connector output -> ``[T, num_queries, H]``.

        ``T`` is the total tile count in the batch (Idefics3 flattens tiles
        across images before the connector), so every tile is sampled
        independently with the same weights.
        """
        c = self.cfg
        if tile_features.dim() != 3:
            raise ValueError(
                f"expected [tiles, {c.tile_tokens}, {c.hidden_size}], got "
                f"{tuple(tile_features.shape)}"
            )
        T, S, H = tile_features.shape
        if S != c.tile_tokens or H != c.hidden_size:
            raise ValueError(
                f"expected [tiles, {c.tile_tokens}, {c.hidden_size}], got "
                f"{tuple(tile_features.shape)}"
            )
        kv = tile_features + self.kv_pos.to(tile_features.dtype)
        q = self.queries.to(tile_features.dtype).unsqueeze(0).expand(T, -1, -1)
        for block in self.blocks:
            q = block(q, kv)
        return self.out_norm(q)
