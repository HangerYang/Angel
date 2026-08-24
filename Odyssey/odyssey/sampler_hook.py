"""Env-gated pure-torch replacement for vLLM's Triton rejection sampler.

Activated by `ODYSSEY_BRANCH` in {baseline, stale_corr, stale_skip, recompute}.
`block` deliberately does NOT route here -- it is vLLM's own
`rejection_sample_method="block"` running on the stock kernels.

The signature mirrors `rejection_sample()` in
`third_party/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`
so the call site needs a two-line diff and nothing else.
"""

from __future__ import annotations

import os
import time

import torch

from . import events
from .branches import BRANCHES, verify_one_request


def branch() -> str | None:
    b = os.environ.get("ODYSSEY_BRANCH", "").strip()
    if not b or b == "block":
        # "block" and unset both mean: leave the stock kernels alone.
        return None
    if b not in BRANCHES:
        raise ValueError(f"ODYSSEY_BRANCH={b!r} not in {BRANCHES + ('block',)}")
    return b


_ROUND = {"n": 0}


def odyssey_rejection_sample(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor | None,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
    branch_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = int(cu_num_logits.shape[0]) - 1
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)

    cu = cu_num_logits.tolist()
    idx_map = idx_mapping.tolist()
    temps = temperature.tolist()
    seeds = seed.tolist()

    t0 = time.perf_counter()
    for r in range(num_reqs):
        start, end = cu[r], cu[r + 1]
        n = end - start
        req_slot = idx_map[r]
        temp = float(temps[req_slot])

        tl = target_logits[start:end]
        # vLLM stores the proposal for target row i at draft_sampled[i + 1]
        # (see _rejection_kernel: `draft_sampled_ptr + logit_idx + 1`), so
        # shift once here and keep branches.py on a 1:1 alignment.
        dt = draft_sampled[start + 1 : end]
        dl = None
        if draft_logits is not None and n > 1:
            # draft_logits is [max_num_reqs, K, V]; gather the rows this
            # request's positions map to, matching the Triton path's indexing.
            rows = expanded_idx_mapping[start : end - 1]
            locs = expanded_local_pos[start : end - 1]
            dl = draft_logits[rows, locs]

        # Per-request generator so a rerun with the same seed reproduces the
        # same accept/reject coin flips across branches.
        gen = None
        if temp > 0.0:
            gen = torch.Generator(device=target_logits.device)
            gen.manual_seed(int(seeds[req_slot]) + _ROUND["n"] * 1_000_003 + r)

        out = verify_one_request(branch_name, tl, dt, dl, temp, gen)
        # Absolute position of the first verified token. The offline re-scorer
        # needs it to rebuild the conditioning prefix (and to spot where one
        # prompt ends and the next begins, since positions reset per request).
        first_pos = int(pos[start])

        toks = out["tokens"]
        k = min(len(toks), num_speculative_steps + 1)
        num_sampled[r] = k
        for j in range(k):
            sampled[r, j] = toks[j]

        if events.enabled():
            events.emit(
                "round",
                round=_ROUND["n"],
                req=r,
                branch=branch_name,
                num_spec=out["num_spec"],
                first_pos=first_pos,
                tau=out["tau"],
                first_reject=out["first_reject"],
                correction=out["correction"],
                reject_entropy=out["reject_entropy"],
                salvaged=out["salvaged"],
                salvage_trace=out["salvage_trace"],
                temperature=temp,
                # Branch 3's offline re-scorer needs the proposals and the
                # correction to rebuild the corrected tail.
                draft_tokens=dt[: out["num_spec"]].tolist(),
                emitted=toks,
            )

    if events.enabled():
        events.emit(
            "round_timing",
            round=_ROUND["n"],
            branch=branch_name,
            num_reqs=num_reqs,
            sampler_ms=(time.perf_counter() - t0) * 1000.0,
        )
    _ROUND["n"] += 1
    return sampled, num_sampled
