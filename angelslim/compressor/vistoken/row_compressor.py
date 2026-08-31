"""Row compressor: 64 image rows per tile -> k rows, in target-aux-HS space.

Idea 1. The drafter's single attention layer sees every target aux stream at
every position, but at ~900 image rows it cannot route over them. This module
compresses the image rows to k per tile while keeping the banded depth
structure intact: one routing decision, applied to all aux streams, so the same
image region reaches the drafter at every depth it was read at.

Contrast with ``qsampler.py`` (dropped): that compressed the *connector* output
in LM embedding space against a target-invariance objective. This one lives
after the target forward, compresses the aux hidden states the drafter actually
consumes, and is trained by the draft loss alone.

Shape contract
--------------
Input   ``[B, T, tile_tokens, n*d]``  T tiles, n aux streams concatenated
Output  ``[B, T, k, n*d]``            same stream layout, k rows per tile

There is no value projection. The output is a convex combination of real target
hidden states, so it stays inside the distribution ``fc_norm``/``fc`` expects.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class VisRowCompressorConfig:
    hidden_size: int = 576
    # Aux streams carried per position (9 for the 3x3 banded-mix config).
    num_streams: int = 9
    # Compressed rows per tile. k=1 on a 13-17 tile prompt gives 13-17 rows.
    num_queries: int = 1
    # SmolVLM/Idefics3 emits 64 rows per tile after pixel shuffle.
    tile_tokens: int = 64
    # Upper bound on tiles per prompt; sizes the tile-index table.
    max_tiles: int = 32
    # Routing-only projection width. temperature = sqrt(key_dim).
    key_dim: int = 64
    temperature: float = 8.0
    init_std: float = 0.02
    # "shared": one routing decision for every aux stream (the depth-
    # correspondence claim). "per_band": one query set, one k_proj per band --
    # isolates routing from query content.
    routing: str = "shared"
    # Stream grouping for per_band routing, e.g. [[0,1,2,3],[4,5,6],[7,8]].
    stream_bands: Optional[Tuple[Tuple[int, ...], ...]] = None
    # "learned": learned queries. "mean": uniform average over the tile's rows
    # (the null baseline; no routing params are used).
    query_mode: str = "learned"
    # Compressor param-group LR; the drafter keeps args.learning_rate.
    lr: float = 1e-3

    def __post_init__(self):
        if self.routing not in ("shared", "per_band"):
            raise ValueError(f"routing must be shared|per_band, got {self.routing!r}")
        if self.query_mode not in ("learned", "mean"):
            raise ValueError(
                f"query_mode must be learned|mean, got {self.query_mode!r}"
            )
        if self.routing == "per_band":
            if not self.stream_bands:
                raise ValueError("per_band routing requires stream_bands")
            flat = [s for band in self.stream_bands for s in band]
            if sorted(flat) != list(range(self.num_streams)):
                raise ValueError(
                    "stream_bands must partition range(num_streams); got "
                    f"{self.stream_bands} for num_streams={self.num_streams}"
                )


class VisRowCompressor(nn.Module):
    """Cross-attention over a tile's rows with an identity value path."""

    def __init__(self, cfg: VisRowCompressorConfig):
        super().__init__()
        self.cfg = cfg
        d, k = cfg.hidden_size, cfg.num_queries

        if cfg.query_mode == "learned":
            self.queries = nn.Parameter(torch.empty(k, d))
            nn.init.normal_(self.queries, std=cfg.init_std)
            # One vector per tile index: without it no summary knows where its
            # tile sat on the page. Zero init, so tiles start indistinguishable.
            self.tile_embed = nn.Parameter(torch.zeros(cfg.max_tiles, d))
        else:
            self.register_parameter("queries", None)
            self.register_parameter("tile_embed", None)

        # Routing only -- never applied to the values.
        n_routes = (
            len(cfg.stream_bands) if cfg.routing == "per_band" else 1
        )
        if cfg.query_mode == "learned":
            self.k_proj = nn.ModuleList(
                nn.Linear(d, cfg.key_dim, bias=False) for _ in range(n_routes)
            )
            for proj in self.k_proj:
                nn.init.normal_(proj.weight, std=cfg.init_std)
        else:
            self.k_proj = None

        # Reference stream for routing: its own mix, deliberately NOT the fc's
        # band mix -- one knob doing two jobs makes the routing ablations
        # unreadable. Uniform init.
        self.ref_mix_logits = nn.Parameter(torch.zeros(cfg.num_streams))

        # The embedding half of the drafter's input on a compressed row. Held
        # as a delta on the <image> embedding and zero-initialised, so the
        # drafter starts from exactly the vector it already saw on image rows.
        self.row_embed_delta = nn.Parameter(torch.zeros(d))

    @property
    def num_queries(self) -> int:
        return self.cfg.num_queries

    @staticmethod
    def slot_offsets(num_queries: int, tile_tokens: int = 64) -> tuple:
        """Which rows inside a tile the k summaries occupy.

        Evenly spaced centres: k=1 -> (32,), k=4 -> (8, 24, 40, 56). These are
        real target positions, so a summary keeps the RoPE angle the target
        computed at that slot -- no invented position, nothing to round.

        Fixed and data-independent on purpose: vLLM builds the draft's slot
        mapping from ``input_ids`` alone, before the model runs, so the kept
        rows cannot depend on the routing weights.
        """
        if not 0 < num_queries <= tile_tokens:
            raise ValueError(
                f"num_queries must be in 1..{tile_tokens}, got {num_queries}"
            )
        return tuple(
            (2 * i + 1) * tile_tokens // (2 * num_queries) for i in range(num_queries)
        )

    @property
    def offsets(self) -> tuple:
        return self.slot_offsets(self.cfg.num_queries, self.cfg.tile_tokens)

    def _reference(self, tiles: torch.Tensor) -> torch.Tensor:
        """[B, T, N, n, d] -> [B, T, N, d]: softmax mix over aux streams."""
        w = torch.softmax(self.ref_mix_logits.float(), dim=0).to(tiles.dtype)
        return torch.einsum("btnsd,s->btnd", tiles, w)

    def _weights(self, ref: torch.Tensor, n_tiles: int) -> torch.Tensor:
        """Routing weights [B, T, R, k, N] (R=1 shared, R=bands per_band)."""
        cfg = self.cfg
        b, t, n, _ = ref.shape
        if cfg.query_mode == "mean":
            return ref.new_full((b, t, 1, cfg.num_queries, n), 1.0 / n)
        if n_tiles > cfg.max_tiles:
            raise ValueError(
                f"{n_tiles} tiles exceeds max_tiles={cfg.max_tiles}; raise it "
                "in the config (the tile-index table is sized by it)"
            )
        # [T, k, d]: one query set per tile, shifted by the tile-index vector.
        q = self.queries.unsqueeze(0) + self.tile_embed[:n_tiles].unsqueeze(1)
        outs = []
        for proj in self.k_proj:
            qh = proj(q.to(proj.weight.dtype))            # [T, k, key]
            kh = proj(ref.to(proj.weight.dtype))          # [B, T, N, key]
            logits = torch.einsum("tkc,btnc->btkn", qh, kh) / cfg.temperature
            outs.append(torch.softmax(logits.float(), dim=-1).to(ref.dtype))
        return torch.stack(outs, dim=2)

    def forward(
        self, tiles: torch.Tensor, n_tiles: int, expected_rows: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress ``[B, T, N, n*d]`` tile rows to ``[B, T, k, n*d]``.

        Also returns the routing weights ``[B, T, k, N]`` (band-averaged when
        routing is per_band) so the caller can place each output row's RoPE
        position at the weighted mean of its sources.

        ``N`` need not equal ``cfg.tile_tokens`` -- nothing below this check
        actually depends on ``tile_tokens`` (that field only sizes
        ``slot_offsets`` for the *unpruned* case). A caller that pre-prunes
        each tile to ``M < tile_tokens`` rows (e.g. target-attention row
        pruning, before this module ever sees them) passes
        ``expected_rows=M`` to assert against the real row count instead of
        the config default.
        """
        cfg = self.cfg
        b, t, n, flat = tiles.shape
        if flat != cfg.num_streams * cfg.hidden_size:
            raise ValueError(
                f"expected {cfg.num_streams}*{cfg.hidden_size} feature dims, got {flat}"
            )
        want = cfg.tile_tokens if expected_rows is None else expected_rows
        if n != want:
            raise ValueError(f"expected {want} rows per tile, got {n}")

        streams = tiles.view(b, t, n, cfg.num_streams, cfg.hidden_size)
        ref = self._reference(streams)
        w = self._weights(ref, n_tiles)                    # [B, T, R, k, N]

        if cfg.routing == "shared":
            out = torch.einsum("btkn,btnsd->btksd", w[:, :, 0], streams)
            pos_w = w[:, :, 0]
        else:
            parts = [None] * cfg.num_streams
            for band_idx, band in enumerate(cfg.stream_bands):
                idx = torch.as_tensor(band, device=streams.device)
                sub = streams.index_select(3, idx)          # [B,T,N,|band|,d]
                mixed = torch.einsum("btkn,btnsd->btksd", w[:, :, band_idx], sub)
                for slot, stream_id in enumerate(band):
                    parts[stream_id] = mixed[:, :, :, slot]
            out = torch.stack(parts, dim=3)
            pos_w = w.mean(dim=2)
        return out.reshape(b, t, cfg.num_queries, flat), pos_w
