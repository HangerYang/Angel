# Odyssey — resync-after-rejection experiments

Inference-time only. No training. Measures what happens if speculative decoding
stops throwing away the draft tail after the first rejection.

## The five branches

At the position where draft token *i* is rejected:

| # | name | what it does | cost | exact? |
|---|------|--------------|------|--------|
| 5 | `baseline` | discard tokens *i+1..γ* (Leviathan et al.) | free | yes |
| 1 | `stale_corr` | splice the correction at *i*, keep verifying *i+1..γ* against the **stale** target distributions | free | **no** |
| 2 | `stale_skip` | same continuation, but keep the *rejected* token at *i* — pretend nothing changed | free | **no** |
| 3 | `recompute` | online: identical to baseline. Offline: re-score the corrected tail with fresh target distributions | 1 extra target forward | yes |
| 4 | `block` | joint-suffix verification — accept/reject the tail as a unit | free | yes |

Branches 1 and 2 were folded in as separate branches rather than merged,
because they differ in what actually gets *emitted*: `stale_corr` emits the
correction, `stale_skip` emits the rejected token.

**Branch 4 is not reimplemented here.** vLLM 0.25 already ships joint-suffix
verification (`_compute_cumulative_log_p_kernel` in
`third_party/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`),
reached by `speculative_config={"rejection_sample_method": "block"}`. The sweep
flips that field instead of duplicating the kernel.

## Layout

```
odyssey/branches.py       the branch policies, pure torch, no vLLM imports
odyssey/sampler_hook.py   env-gated replacement for vLLM's rejection_sample()
odyssey/events.py         per-round JSONL telemetry
odyssey/rescore.py        branch 3 control: offline fresh-distribution re-score
odyssey/analyze.py        end-of-run aggregation (token-level drift)
odyssey/utility.py        answer-level scoring: accuracy, delta, agreement
odyssey/test_branches.py  unit tests for the policies
scripts/run_branch_sweep.sh
results/<tag>/            per-branch, per-benchmark outputs + report.json
logs/<tag>/               per-round event JSONL + eval stdout
```

## Integration

Two hooks, both minimal and both env-gated so the default path is untouched:

1. `third_party/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
   — `RejectionSampler.__call__` dispatches to the Odyssey python sampler when
   `ODYSSEY_BRANCH` is set, otherwise runs the stock Triton kernels.
2. `tools/vllm_offline_eagle3_vlm_batch.py` — passes
   `ODYSSEY_REJECTION_METHOD` through to `speculative_config`.

Note that vLLM 0.25's live code is under `v1/worker/gpu/`. The files
`v1/spec_decode/eagle.py` and `v1/worker/gpu_model_runner.py` are dead in this
version and editing them does nothing.

### Correctness of the python sampler

`baseline` reproduces the stock Triton kernel **exactly** — identical
per-position acceptance rates on textvqa at γ=8:

```
stock triton : pos_0 0.1881  pos_1 0.0594  pos_2 0.0149  pos_3 0.005  pos_4..7 0.0
odyssey base : pos_0 0.1881  pos_1 0.0594  pos_2 0.0149  pos_3 0.005  pos_4..7 0.0
```

Two alignment details the reimplementation has to get right, both easy to get
wrong:

- vLLM stores the proposal for target row `i` at `draft_sampled[i + 1]`, not
  `draft_sampled[i]` (see `_rejection_kernel`). Getting this wrong silently
  drops acceptance from 1.86 to 1.07.
- `apply_sampling_params` has **already** applied temperature to the logits
  handed to the sampler. Dividing by temperature again double-applies it.

## Running

```bash
TAG=g8_t0_main GAMMA=8 TEMP=0 NUM_PROMPTS=40 OUTPUT_LEN=256 GPU=0 \
  bash Odyssey/scripts/run_branch_sweep.sh
```

Then `Odyssey/results/<tag>/report.json` holds everything.

### IGNORE_EOS

Defaults to 0 (natural generation). Setting it to 1 makes every request emit
exactly `OUTPUT_LEN` tokens so wall-clock is comparable across runs — but on
short-answer benchmarks it measures acceptance mostly on post-EOS text. On
textvqa at γ=8 it drops acceptance 1.86 → 1.27 (natural output there averages
9.75 tokens). Long-generation benchmarks like MATH-500 are unaffected in
practice, which is why the sweep defaults to them.

## Caveats on the branch-3 control

The re-score only means something if it conditions on the same prefix the
online target saw, so:

- **Text-only benchmarks only.** SmolVLM is a VLM; on an image benchmark the
  online distribution is conditioned on image embeddings the offline pass
  cannot rebuild. MATH-500 is text-only.
- Prefixes are rebuilt as `prompt tokens + tokens emitted in earlier rounds`
  and validated against the `first_pos` the sampler logged. Groups that fail
  to line up are skipped and counted (`groups_skipped_prefix_mismatch`) rather
  than silently scored against the wrong context. The `g8_t0_main` run matched
  36/37; the one skip is vLLM's warmup round.

## Two kinds of "did it change the output"

Keep these apart -- they disagree sharply in the `g8_t0_main` run:

- **Token identity** (`analyze.py`): `exact_match_rate` is string equality with
  the baseline generation; `mean_prefix_agreement` is the longest common
  *character* prefix over the baseline's length. Both are exact-match metrics --
  prefix agreement is a graded exact match, not a semantic one. They answer
  "did the branch reproduce the target's tokens".
- **Utility** (`utility.py`): extracts the answer (option letter, `\boxed{}`,
  last number) and reports accuracy, accuracy delta vs. baseline, and
  `utility_agreement` -- how often the branch lands on the same answer as the
  baseline regardless of wording. This answers "is the branch still useful".

On MMStar `stale_skip` scores 0.000 exact match and 0.975 utility agreement.
Reporting only the first would have been misleading.

## Prompt style matters more than anything else here

`PROMPT_STYLE=answer_then_describe` ("ATD") appends *"Then describe the image in
detail to justify your answer"*. It lengthens MMStar output 8.8 -> 80.1 tokens
while keeping the answer at the front. That is the regime this experiment needs:
long enough for post-rejection salvage to compound, and still answer-extractable.

Under raw prompts the same branch looks utility-neutral; under ATD it flips 10%
of answers and rewrites the justification. Run ATD.

## Results

- [`RESULTS_ATD.md`](RESULTS_ATD.md) — **read this one** for utility.
- [`RESULTS.md`](RESULTS.md) — raw prompts; holds the over-salvage control and
  the temp-0 `block` finding.
