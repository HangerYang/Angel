"""Alignment test for branch distillation: does the branch step get scored
against the teacher's continuation of the SUBSTITUTED sequence, at the right
absolute position?  Uses a fake teacher whose logits at position p are a
deterministic function of the token at p, so a misalignment cannot pass.
"""
import sys, types, torch
from torch import nn
sys.path.insert(0, "/home/hyang/AngelSlim")
from angelslim.compressor.speculative.train.trainer.eagle3_trainer import Eagle3Trainer

B, S, V, DV = 2, 9, 20, 12
torch.manual_seed(0)

def teacher_fn(ids):
    """Ranked logits at p, all a function of the token AT p: top-1 is
    (t*7+1)%DV, then +1, then +2 — so a top-3 exists to branch into."""
    out = torch.full((B, S, V), -10.0)
    first = (ids * 7 + 1) % DV
    for rank, score in enumerate((10.0, 8.0, 6.0)):
        out.scatter_(2, ((first + rank) % DV)[..., None], score)
    return out

class FakeDraft:
    def __init__(self):
        used = list(range(DV))                      # draft vocab = target ids 0..DV-1
        self.d2t = torch.tensor([used[i] - i for i in range(DV)])
        self.t2d = torch.tensor([i in set(used) for i in range(V)])
        self.calls = []
    def embed_input_ids(self, ids):
        self.last_ids = ids
        return torch.zeros(B, S, 4)
    def encode_layers(self, **kw):
        self.calls.append(kw)
        return torch.zeros(B, S, 4), kw["cache_hidden"]
    def compute_logits(self, h):
        return torch.zeros(B, S, DV)

trainer = object.__new__(type("T", (Eagle3Trainer,), {"prepare_data_for_draft_model": None,
                                                      "compute_loss": None}))
trainer.branch_distill_loss_weight = 0.1
trainer.branch_distill_target_top_k = 3
trainer.branch_distill_prob_ratio_threshold = 0.0
trainer.branch_distill_objective = "change"
trainer.branch_distill_change_delta_weight = False
trainer.branch_distill_warmup_steps = 0
trainer.branch_distill_ramp_steps = 0
trainer.branch_distill_synthetic = False
trainer.branch_distill_synthetic_ratio = 1.0
trainer.branch_distill_steps = 1
trainer.branch_teacher_logits = teacher_fn

orig_ids = torch.randint(0, DV, (B, S))
trainer._branch_ctx = {"input_ids": orig_ids}
draft = FakeDraft()

# --- Build a step-idx frame: target_logits = teacher on the real sequence,
#     shifted left once (pre-loop) + idx times, as the trainer does.
idx = 0
real_logits = teacher_fn(orig_ids)
target_logits = Eagle3Trainer._shift_left(real_logits, 1 + idx)
loss_mask = torch.ones(B, S, 1, dtype=torch.long)

# --- Draft logits: force the teacher's 2nd choice at even positions (should
#     branch) and its 1st choice at odd ones (should not).
second = target_logits.topk(3, -1).indices[..., 1]
first = target_logits.argmax(-1)
even = (torch.arange(S) % 2 == 0)[None, :].expand(B, S)
# ties in the zero-padded tail can rank a non-draft id; those columns
# are excluded by the bounds check anyway.
draft_choice = torch.where(even, second, first).clamp(max=DV - 1)
logits = torch.full((B, S, DV), -5.0).scatter_(2, draft_choice[..., None], 5.0)

pending = trainer._branch_decide(idx, draft, logits, target_logits, loss_mask)
assert pending is not None, "no branch positions found — retune the fixture"

# The teacher fixture scores top-2 two logits below top-1, so p2/p1 = exp(-2).
# A 0.2 plausibility threshold rejects those branches; 0.1 keeps them.
trainer.branch_distill_prob_ratio_threshold = 0.2
blocked = trainer._branch_decide(idx, draft, logits, target_logits, loss_mask)
assert blocked is None, "ratio threshold failed to reject weak alternatives"
assert trainer._branch_last_stats["candidates"] > 0
assert trainer._branch_last_stats["survivors"] == 0
trainer.branch_distill_prob_ratio_threshold = 0.1
pending = trainer._branch_decide(idx, draft, logits, target_logits, loss_mask)
assert pending is not None, "ratio threshold failed to keep plausible alternatives"

# 1. mask == draft top-1 differs from teacher top-1 but is in teacher top-3
teacher_top1 = target_logits.argmax(-1)
topk = target_logits.topk(3, -1).indices
want = ((topk == draft_choice[..., None]).any(-1) & (draft_choice != teacher_top1)
        & (torch.arange(S)[None, :] + idx + 2 < S))
assert torch.equal(pending["mask"], want), "branch mask wrong"
print(f"branch positions: {int(want.sum())} of {B*S}")

# 2. the branch target delta at index j must be the teacher's logit change
#    AFTER the draft token was placed at absolute position a = j + idx + 2.
rows, cols = torch.nonzero(pending["mask"], as_tuple=True)
sub_ids = orig_ids.clone()
sub_ids[rows, cols + idx + 2] = draft_choice[rows, cols].to(sub_ids.dtype)
branch_expected_logits = Eagle3Trainer._shift_left(
    teacher_fn(sub_ids), idx + 2
)[..., draft.t2d].float()
real_expected_logits = Eagle3Trainer._shift_left(
    target_logits, 1
)[..., draft.t2d].float()
expected_delta = (
    branch_expected_logits - branch_expected_logits.mean(dim=-1, keepdim=True)
    - (real_expected_logits - real_expected_logits.mean(dim=-1, keepdim=True))
)
assert torch.equal(
    pending["target_delta"][rows, cols],
    expected_delta[rows, cols],
), "target delta is not branch teacher minus real teacher"
print("branch target delta follows the substituted token at the right position")

# 3. the forked step feeds the branch token at exactly those positions.
next_ids = Eagle3Trainer._shift_left(orig_ids, 1 + idx + 1)
base_logits = torch.zeros(B, S, DV)
loss, n, target_rms, draft_rms, denom = trainer._branch_loss(
    pending, draft, next_ids, torch.zeros(B, S, 4), [[[], []]], None, None,
    loss_mask, None, base_logits
)
fed = draft.last_ids
assert torch.equal(fed[rows, cols], draft_choice[rows, cols]), "branch token not fed"
keep = ~pending["mask"]
assert torch.equal(fed[keep], next_ids[keep]), "non-branch positions were disturbed"
assert torch.isfinite(loss) and n == int(want.sum())
# Diagnostics: RMS values are masked means over the same active branch
# positions used by the MSE. The fake draft emits identical branch/base logits,
# so its delta RMS is exactly zero while the teacher delta is non-zero.
assert torch.isfinite(target_rms) and target_rms.item() > 0.0, "target delta RMS bad"
assert torch.isfinite(draft_rms) and draft_rms.item() == 0.0, "draft delta RMS bad"
# denom is the active-position count the branch-change MSE divides by; it must
# be positive and no larger than the number of branch positions.
assert 0 < denom <= n, f"denom {denom} outside (0, {n}]"
assert torch.isfinite(loss) and loss.item() > 0.0
# 4. delta weighting: alpha_i = RMS(delta_T^i)/mean(RMS) turns the plain mean
#    into a weighted mean, so the loss must equal sum(s*mse)/sum(s) over the
#    active positions -- computed here independently from the pending target.
trainer.branch_distill_change_delta_weight = True
wloss, _, _, _, _ = trainer._branch_loss(
    pending, draft, next_ids, torch.zeros(B, S, 4), [[[], []]], None, None,
    loss_mask, None, base_logits
)
active = (pending["mask"][..., None].int() * loss_mask).float()
td = pending["target_delta"].float()
# The fake draft's delta is zero, so the per-position MSE is just E[delta_T^2].
mse = td.pow(2).mean(dim=2, keepdim=True)
s_i = mse.sqrt()
want_loss = (active * s_i * mse).sum() / (active * s_i).sum()
assert torch.allclose(wloss, want_loss, atol=1e-5), (
    f"weighted loss {wloss.item()} != sum(s*mse)/sum(s) {want_loss.item()}"
)
assert wloss.item() > loss.item(), "weighting should upweight the big-delta branches"
trainer.branch_distill_change_delta_weight = False
print(f"delta-weighted loss ok — {wloss.item():.4f} vs unweighted {loss.item():.4f}")

print(f"forked step ok — loss={loss.item():.4f}, n={n}, "
      f"target_delta_rms={target_rms.item():.4f}, draft_delta_rms={draft_rms.item():.4f}, denom={denom}")
# 5. curriculum: 0 before warmup, linear through the ramp, flat after.
class _S:
    global_step = 0
trainer.state = _S()
trainer.branch_distill_warmup_steps = 20000
trainer.branch_distill_ramp_steps = 15000
for step, want in [(0, 0.0), (19999, 0.0), (20000, 0.0), (27500, 0.05),
                   (35000, 0.1), (66000, 0.1)]:
    trainer.state.global_step = step
    got = trainer.branch_weight_now()
    assert abs(got - want) < 1e-9, f"step {step}: weight {got} != {want}"
print("curriculum ramp ok")
trainer.branch_distill_warmup_steps = 0
trainer.branch_distill_ramp_steps = 0

# 6. top_k=2 selects exactly the draft-took-teacher-rank-2 positions, and
#    synthetic mode adds rank-2 substitutions elsewhere, capped by the ratio.
trainer.branch_distill_target_top_k = 2
p2 = trainer._branch_decide(idx, draft, logits, target_logits, loss_mask)
rank2 = target_logits.topk(2, -1).indices[..., 1]
natural = (draft_choice == rank2) & (draft_choice != target_logits.argmax(-1))
natural = natural & loss_mask[..., 0].bool()
cols_ok = (torch.arange(S) + idx + 2 < S)[None, :]
natural = natural & cols_ok
assert torch.equal(p2["mask"], natural), "top_k=2 mask is not the rank-2 set"
assert torch.equal(p2["tokens"][natural], draft_choice[natural]), "natural token wrong"
print(f"rank-2 only: {int(natural.sum())} branches")

trainer.branch_distill_synthetic = True
trainer.branch_distill_synthetic_ratio = 1.0
p3 = trainer._branch_decide(idx, draft, logits, target_logits, loss_mask)
added = p3["mask"] & ~natural
n_nat = int(natural.sum())
assert int(added.sum()) == trainer._branch_last_stats["synthetic"]
assert int(added.sum()) <= n_nat, "synthetic budget exceeded the natural count"
assert torch.equal(p3["tokens"][added], rank2[added]), "synthetic token is not rank-2"
assert torch.equal(p3["tokens"][natural], draft_choice[natural]), "natural token changed"
assert bool((added & ~loss_mask[..., 0].bool()).sum() == 0), "synthetic hit a masked position"
print(f"synthetic: {int(added.sum())} added on top of {n_nat} natural")
trainer.branch_distill_synthetic = False
trainer.branch_distill_target_top_k = 3

print("PASS")
