"""Per-rejection resync policies, in pure torch.

Standard speculative decoding throws away every draft token after the first
rejection. The target already scored all of them in one forward, so those
distributions are sitting in memory unused -- they are just *stale*: position
i+1 was scored assuming the rejected token stood at i.

The branches here differ in what they do with that stale tail:

  baseline      discard the tail (Leviathan et al.); the control for tau.
  stale_corr    splice the correction at i, keep verifying i+1.. against the
                stale distributions. Off-policy, free.
  stale_skip    same continuation, but keep the *rejected* draft token at i
                rather than the correction -- i.e. pretend nothing changed.
                Maximally off-policy, also free.
  recompute     emit like baseline, but log the event so an offline pass can
                re-score the corrected tail with fresh target distributions.
                This is the ground truth #1/#2 are approximating.
  block         handled by vLLM's own `rejection_sample_method="block"`, not
                here -- see README.

`block` is absent from this module on purpose: vLLM 0.25 already implements
joint-suffix verification in `_compute_cumulative_log_p_kernel`, so running it
means flipping a config field, not duplicating the kernel.

Everything is written per-request in plain torch. That is slower than the Triton
path it replaces, which is fine: these runs measure acceptance behaviour, and
the wall-clock numbers that matter come from the `baseline` vs `block` A/B
running on the untouched kernels.
"""

from __future__ import annotations

import torch

BRANCHES = ("baseline", "stale_corr", "stale_skip", "recompute")

# Branches whose emitted token sequence is byte-identical to standard
# speculative decoding. Anything outside this set breaks the distributional
# guarantee by construction -- that is the thing being measured.
EXACT_BRANCHES = frozenset({"baseline", "recompute"})


def _probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Distribution at one position.

    No temperature division here: vLLM's `apply_sampling_params` has already
    scaled the target logits in place, and the Triton kernel this replaces
    likewise takes a plain softmax of whatever it is handed (temperature only
    selects the greedy branch). Dividing again would double-apply it.
    """
    if temperature <= 0.0:
        out = torch.zeros_like(logits, dtype=torch.float32)
        out[int(logits.argmax())] = 1.0
        return out
    return torch.softmax(logits.float(), dim=-1)


def entropy(logits: torch.Tensor, temperature: float) -> float:
    """Shannon entropy (nats) of the target distribution.

    Always computed at temperature 1 regardless of the sampling temperature:
    at temp 0 the sampling distribution is a point mass with entropy 0, which
    would make the entropy-vs-outcome scatter degenerate. What we want is the
    model's own uncertainty at that position.
    """
    p = torch.softmax(logits.float(), dim=-1)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum())


def _residual_sample(
    p: torch.Tensor,
    q: torch.Tensor | None,
    temperature: float,
    gen: torch.Generator | None,
) -> int:
    """Draw the correction after a rejection.

    Greedy takes the target argmax. Otherwise this is the standard residual
    norm(max(0, p - q)); with no draft distribution on hand we fall back to p,
    which is what vLLM does when draft logits are not returned.
    """
    if temperature <= 0.0:
        return int(p.argmax())
    resid = p if q is None else (p - q).clamp_min(0.0)
    total = float(resid.sum())
    if total <= 1e-9:
        resid = p
        total = float(resid.sum())
    return int(torch.multinomial(resid / total, 1, generator=gen))


def verify_one_request(
    branch: str,
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_logits: torch.Tensor | None,
    temperature: float,
    gen: torch.Generator | None,
) -> dict:
    """Run one verification round for a single request.

    Args:
        target_logits: [n, V]. Row j is the target's distribution for the token
            following the first j draft tokens, so rows 0..n-2 verify the n-1
            proposals and row n-1 is the bonus slot.
        draft_tokens: the proposals, already aligned so that entry i is the
            token verified against `target_logits[i]`. vLLM stores these
            shifted by one (the kernel reads `draft_sampled[logit_idx + 1]`);
            the caller in sampler_hook does that slice.
        draft_logits: [n-1, V] or None. Only needed for the residual draw.
        temperature: this request's sampling temperature (0 => greedy).

    Returns a dict with the emitted tokens and the per-round telemetry the
    analysis pass groups on.
    """
    n = int(target_logits.shape[0])
    num_spec = n - 1
    emitted: list[int] = []

    first_reject = -1
    correction = -1
    reject_entropy = float("nan")
    # Draft tokens accepted *after* the first rejection. This is the whole
    # point of the experiment: baseline salvages zero by definition.
    salvaged = 0
    salvage_trace: list[int] = []

    i = 0
    while i < num_spec:
        p = _probs(target_logits[i], temperature)
        t = int(draft_tokens[i])

        if temperature <= 0.0:
            accept = t == int(target_logits[i].argmax())
        else:
            q = None if draft_logits is None else torch.softmax(
                draft_logits[i].float(), dim=-1
            )
            qt = 1.0 if q is None else float(q[t].clamp_min(1e-12))
            ratio = float(p[t]) / qt
            if ratio >= 1.0:
                accept = True
            else:
                r = torch.rand(1, generator=gen, device=p.device)
                accept = bool(float(r) < ratio)

        if accept:
            emitted.append(t)
            i += 1
            continue

        # --- rejection ---
        q = None
        if draft_logits is not None and temperature > 0.0:
            q = torch.softmax(draft_logits[i].float(), dim=-1)
        corr = _residual_sample(p, q, temperature, gen)

        first_reject = i
        correction = corr
        reject_entropy = entropy(target_logits[i], temperature)

        if branch in ("baseline", "recompute"):
            # Standard behaviour: the correction terminates the round.
            emitted.append(corr)
            return _result(
                emitted, num_spec, first_reject, correction, reject_entropy,
                salvaged, salvage_trace, branch,
            )

        if branch == "stale_corr":
            emitted.append(corr)
        elif branch == "stale_skip":
            # Keep the rejected token: "ignore that position i changed at all".
            emitted.append(t)
        else:
            raise ValueError(f"unknown branch {branch!r}")

        # Continue verifying the tail against distributions that were computed
        # under the *rejected* token. Every acceptance from here is salvage.
        j = i + 1
        while j < num_spec:
            pj = _probs(target_logits[j], temperature)
            tj = int(draft_tokens[j])
            if temperature <= 0.0:
                acc_j = tj == int(target_logits[j].argmax())
            else:
                qj = None if draft_logits is None else torch.softmax(
                    draft_logits[j].float(), dim=-1
                )
                qtj = 1.0 if qj is None else float(qj[tj].clamp_min(1e-12))
                rj = float(pj[tj]) / qtj
                if rj >= 1.0:
                    acc_j = True
                else:
                    r = torch.rand(1, generator=gen, device=pj.device)
                    acc_j = bool(float(r) < rj)
            if not acc_j:
                break
            emitted.append(tj)
            salvaged += 1
            salvage_trace.append(j)
            j += 1

        return _result(
            emitted, num_spec, first_reject, correction, reject_entropy,
            salvaged, salvage_trace, branch,
        )

    # No rejection: every proposal accepted, so the bonus token is free.
    bonus_p = _probs(target_logits[num_spec], temperature)
    bonus = (
        int(bonus_p.argmax())
        if temperature <= 0.0
        else int(torch.multinomial(bonus_p, 1, generator=gen))
    )
    emitted.append(bonus)
    return _result(
        emitted, num_spec, -1, -1, float("nan"), 0, [], branch,
    )


def _result(emitted, num_spec, first_reject, correction, reject_entropy,
            salvaged, salvage_trace, branch) -> dict:
    return {
        "tokens": emitted,
        "num_spec": num_spec,
        # tau: tokens emitted per verification round, the throughput number.
        "tau": len(emitted),
        "first_reject": first_reject,
        "correction": correction,
        "reject_entropy": reject_entropy,
        "salvaged": salvaged,
        "salvage_trace": salvage_trace,
        "branch": branch,
        "exact": branch in EXACT_BRANCHES,
    }
