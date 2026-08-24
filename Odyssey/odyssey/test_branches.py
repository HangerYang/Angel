"""Sanity checks for the branch policies.

Run: python3 -m odyssey.test_branches   (from the Odyssey dir)
"""

from __future__ import annotations

import torch

from .branches import verify_one_request


def _logits(rows, vocab=8):
    """Build [n, V] logits whose argmax at row j is rows[j]."""
    out = torch.full((len(rows), vocab), -10.0)
    for j, tok in enumerate(rows):
        out[j, tok] = 10.0
    return out


def test_all_accepted():
    # Target argmax matches every proposal -> full acceptance + bonus.
    tl = _logits([1, 2, 3, 4])
    dt = torch.tensor([1, 2, 3, 0])
    for b in ("baseline", "stale_corr", "stale_skip", "recompute"):
        r = verify_one_request(b, tl, dt, None, 0.0, None)
        assert r["tokens"] == [1, 2, 3, 4], (b, r["tokens"])
        assert r["tau"] == 4 and r["first_reject"] == -1
        assert r["salvaged"] == 0
    print("ok  all_accepted")


def test_baseline_stops_at_rejection():
    # Proposal 1 (index 1) mismatches: baseline emits [1, correction] and stops.
    tl = _logits([1, 5, 3, 4])
    dt = torch.tensor([1, 2, 3, 0])
    r = verify_one_request("baseline", tl, dt, None, 0.0, None)
    assert r["tokens"] == [1, 5], r["tokens"]
    assert r["first_reject"] == 1 and r["correction"] == 5
    assert r["salvaged"] == 0
    print("ok  baseline_stops")


def test_stale_corr_salvages_tail():
    # Same rejection, but proposals 2 and 3 still match the stale argmaxes,
    # so stale_corr keeps going and salvages both.
    tl = _logits([1, 5, 3, 4])
    dt = torch.tensor([1, 2, 3, 0])
    r = verify_one_request("stale_corr", tl, dt, None, 0.0, None)
    # emitted: accepted [1], correction 5, then salvaged proposal 3
    assert r["tokens"] == [1, 5, 3], r["tokens"]
    assert r["salvaged"] == 1 and r["salvage_trace"] == [2]
    assert r["tau"] == 3
    print("ok  stale_corr_salvages")


def test_stale_skip_keeps_rejected_token():
    tl = _logits([1, 5, 3, 4])
    dt = torch.tensor([1, 2, 3, 0])
    r = verify_one_request("stale_skip", tl, dt, None, 0.0, None)
    # Keeps the REJECTED proposal 2 rather than the correction 5.
    assert r["tokens"] == [1, 2, 3], r["tokens"]
    assert r["salvaged"] == 1
    print("ok  stale_skip_keeps_rejected")


def test_recompute_matches_baseline_online():
    # Branch 3 is a control: online it must be byte-identical to baseline,
    # the ground truth comes from the offline rescore pass.
    tl = _logits([1, 5, 3, 4])
    dt = torch.tensor([1, 2, 3, 0])
    a = verify_one_request("baseline", tl, dt, None, 0.0, None)
    b = verify_one_request("recompute", tl, dt, None, 0.0, None)
    assert a["tokens"] == b["tokens"]
    assert b["exact"] is True
    print("ok  recompute_matches_baseline")


def test_salvage_stops_at_second_mismatch():
    # Reject at 0; proposal 1 matches (salvage 1), proposal 2 does not -> stop.
    tl = _logits([7, 2, 7, 4])
    dt = torch.tensor([1, 2, 3, 0])
    r = verify_one_request("stale_corr", tl, dt, None, 0.0, None)
    assert r["tokens"] == [7, 2], r["tokens"]
    assert r["salvaged"] == 1 and r["salvage_trace"] == [1]
    print("ok  salvage_stops_at_second_mismatch")


def test_tau_never_exceeds_buffer():
    # sampled is [num_reqs, K+1]; no branch may emit more than that.
    torch.manual_seed(0)
    for _ in range(200):
        k = int(torch.randint(1, 9, (1,)))
        tl = torch.randn(k + 1, 32)
        dt = torch.randint(0, 32, (k + 1,))
        for b in ("baseline", "stale_corr", "stale_skip", "recompute"):
            r = verify_one_request(b, tl, dt, None, 0.0, None)
            assert 1 <= r["tau"] <= k + 1, (b, k, r["tau"])
    print("ok  tau_within_buffer")


def test_entropy_is_temp_independent():
    # Entropy must reflect model uncertainty, not the sampling temperature,
    # or the temp-0 scatter collapses to all zeros.
    tl = torch.tensor([[1.0, 1.0, 1.0, 1.0], [10.0, -10.0, -10.0, -10.0]])
    dt = torch.tensor([1, 0])
    r = verify_one_request("baseline", tl, dt, None, 0.0, None)
    assert r["first_reject"] == 0
    assert r["reject_entropy"] > 1.3, r["reject_entropy"]  # ~ln(4) = 1.386
    print(f"ok  entropy_temp_independent (H={r['reject_entropy']:.3f})")


def test_sampling_mode_runs():
    # temp > 0 exercises the residual-sampling path.
    torch.manual_seed(0)
    gen = torch.Generator()
    gen.manual_seed(1234)
    tl = torch.randn(5, 64)
    dl = torch.randn(4, 64)
    dt = torch.randint(0, 64, (5,))
    for b in ("baseline", "stale_corr", "stale_skip", "recompute"):
        r = verify_one_request(b, tl, dt, dl, 1.0, gen)
        assert r["tau"] >= 1
    print("ok  sampling_mode_runs")


if __name__ == "__main__":
    test_all_accepted()
    test_baseline_stops_at_rejection()
    test_stale_corr_salvages_tail()
    test_stale_skip_keeps_rejected_token()
    test_recompute_matches_baseline_online()
    test_salvage_stops_at_second_mismatch()
    test_tau_never_exceeds_buffer()
    test_entropy_is_temp_independent()
    test_sampling_mode_runs()
    print("\nall branch tests passed")
