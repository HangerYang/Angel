"""Acceptance length + throughput for a speculative drafter, measured inside
HiViS's own EaModel loop.

Works for both drafter families, so a paper table can be filled from one
backend:

    --draft_method angelslim_eagle3   our AngelSlim EAGLE-3 drafts (SmolVLM)
    --draft_method hivis|vispec|eagle HiViS's / ViSpec's own checkpoints

and for both benchmark families (see hivis/evaluation/benchmark_data.py):
our vLLM-matched rows (`omnidocbench`, `mmmu_history`) and HiViS's eleven.
Rows and prompts come from the shared adapter, so "their model on our
benchmark" and "our model on their benchmark" are the same code path.

Every mode -- tree, chain, --naive -- runs through the same target backbone and
the same generation loop, so tok/s ratios across runs are a real speedup and not
a comparison of kernel stacks. See README_ANGELSLIM.md.

    # our drafter, our benchmark
    python run_angelslim_eval.py --draft <ckpt> --n 40 --max_new_tokens 1024 \
        --total_token 60 --depth 5 --top_k 10           # tree
        --total_token 5  --depth 4 --top_k 1            # chain (K=4)
        --naive                                         # AR baseline

    # their drafter, our benchmark
    python run_angelslim_eval.py --draft_method hivis \
        --base Qwen/Qwen2.5-VL-7B-Instruct \
        --draft Irisssme/HiViS-Qwen2.5-VL-7B-Instruct --dataset omnidocbench
"""
import argparse, json, time
import torch
from hivis.model.model_hivis import EaModel
from hivis.evaluation.benchmark_data import (
    load_benchmark, prepare_inputs, supported_benchmarks,
)

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="HuggingFaceTB/SmolVLM-256M-Instruct",
                help="target VLM; must match the drafter it was trained against")
ap.add_argument("--draft", required=True, help="drafter checkpoint dir or HF repo id")
ap.add_argument("--draft_method", default="angelslim_eagle3",
                choices=["angelslim_eagle3", "hivis", "vispec", "eagle"])
ap.add_argument("--dataset", default="omnidocbench", choices=supported_benchmarks())
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--max_new_tokens", type=int, default=256)
ap.add_argument("--total_token", type=int, default=60)
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--top_k", type=int, default=10)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max_pixels", type=int, default=None,
                help="cap the image resolution the processor emits, in pixels "
                     "(Qwen2.5-VL: ~1 visual token per 28x28 px). Its default "
                     "turns one OmniDocBench page into ~16.3k tokens, and this "
                     "harness's attention is eager/quadratic, so a 7B target "
                     "OOMs on a 24GB card. 1605632 gives ~2k tokens. Changes "
                     "what the model sees -- hold it fixed across compared runs.")
ap.add_argument("--cache_len", type=int, default=None,
                help="cap the preallocated static KV cache (default: the model's "
                     "max_position_embeddings). Qwen2.5-VL's 128k default costs "
                     "~7.3GB and OOMs a 7B target on a 24GB card; 4096 is ample "
                     "for these prompts. Must be >= prompt + max_new_tokens.")
ap.add_argument("--out", default=None)
ap.add_argument("--naive", action="store_true",
                help="autoregressive baseline in the same harness (speedup denominator)")
a = ap.parse_args()

model = EaModel.from_pretrained(
    base_model_path=a.base, ea_model_path=a.draft,
    total_token=a.total_token, depth=a.depth, top_k=a.top_k,
    draft_method=a.draft_method,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=a.device,
)
model.eval()
model.cache_max_len = a.cache_len
if a.max_pixels is not None:
    image_processor = getattr(model.processor, "image_processor", None)
    if image_processor is None:
        raise SystemExit("--max_pixels: this processor has no image_processor")
    image_processor.max_pixels = a.max_pixels
    if isinstance(getattr(image_processor, "size", None), dict):
        image_processor.size["longest_edge"] = a.max_pixels
    print("max_pixels ->", a.max_pixels)
print("drafter:", type(model.ea_layer).__name__,
      "| method:", a.draft_method, "| aux layers:", getattr(model, "aux_layer_ids", None))

dataset = load_benchmark(a.dataset, sample_count=a.n)
tokenizer = getattr(model.processor, "tokenizer", None) or model.processor


def row_metadata(row):
    """Everything about a benchmark row that is worth keeping beside the
    generation -- the question and the gold answer, so two runs can be diffed
    on what they actually produced, not only on tau. Images and any other
    non-JSON value are dropped; the row schema differs per benchmark, so this
    keeps whatever scalar/list fields there are rather than naming them."""
    keep = {}
    for k, v in row.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            keep[k] = v
        elif isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
            keep[k] = v
    return keep


rows, all_acc = [], []
for i in range(len(dataset)):
    inputs = prepare_inputs(model, dataset, i, a.dataset)
    prompt_len = inputs["input_ids"].shape[1]
    if a.cache_len is not None and prompt_len + a.max_new_tokens > a.cache_len:
        # The cache is preallocated, so overflowing it surfaces deep inside
        # KVCache as "start (0) + length (N) exceeds dimension size". Say what
        # is actually wrong. Qwen2.5-VL does not downsample image tokens the way
        # SmolVLM does -- one OmniDocBench page is ~16k visual tokens.
        raise SystemExit(
            "--cache_len %d is too small for prompt %d (row %d) + --max_new_tokens %d. "
            "Use --cache_len %d or more."
            % (a.cache_len, prompt_len, i, a.max_new_tokens,
               prompt_len + a.max_new_tokens)
        )
    torch.cuda.synchronize(); t0 = time.time()
    if a.naive:
        out_ids, new_token, idx = model.naivegenerate(
            inputs, temperature=0.0, max_new_tokens=a.max_new_tokens, log=True)[:3]
        acc = []
    else:
        out_ids, new_token, idx, acc = model.eagenerate(
            inputs, temperature=0.0, max_new_tokens=a.max_new_tokens, log=True)
    torch.cuda.synchronize(); dt = time.time() - t0
    tau = sum(x + 1 for x in acc) / max(len(acc), 1) if acc else 1.0
    all_acc += acc
    generated_ids = out_ids[0, prompt_len:].tolist()
    rows.append({
        "index": i,
        "tokens": int(new_token), "rounds": len(acc), "tau": tau, "time": dt,
        "prompt_tokens": int(prompt_len),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "generated_ids": generated_ids,
        # Per-round accepted counts (drafted tokens that survived verification,
        # excluding the always-free bonus). The distribution, not just its mean,
        # is what separates "accepts a little every round" from "accepts a lot
        # sometimes" -- two very different drafters with the same tau.
        "accept_lengths": [int(x) for x in acc],
        "row": row_metadata(dataset[i]),
    })
    print("  [%d] tokens=%d rounds=%d tau=%.3f  %.2fs" % (i, new_token, len(acc), tau, dt))

tau = sum(x + 1 for x in all_acc) / max(len(all_acc), 1) if all_acc else 1.0
total_tok = sum(r["tokens"] for r in rows)
total_time = sum(r["time"] for r in rows)
print("\n%s | %s | total_token=%d depth=%d top_k=%d"
      % (a.draft_method, a.dataset, a.total_token, a.depth, a.top_k))
print("mean acceptance length = %.4f over %d rounds | %.2f tok/s | avg out %.1f"
      % (tau, len(all_acc), total_tok / total_time, total_tok / len(rows)))
# Acceptance rate at each speculative position, the way vLLM's
# `acceptance_rates` reports it: fraction of rounds in which at least k drafted
# tokens were accepted, k = 1..max. A tau can be reached either by a short
# chain that almost always lands or a long one that usually breaks at depth 1,
# and only this curve tells those apart.
max_k = max(all_acc) if all_acc else 0
acceptance_rates = [
    sum(1 for x in all_acc if x >= k) / len(all_acc) for k in range(1, max_k + 1)
] if all_acc else []

if a.out:
    json.dump({
        "metrics": {
            "draft": a.draft, "draft_method": a.draft_method, "base": a.base,
            "dataset": a.dataset, "num_prompts": len(rows),
            "total_token": a.total_token, "depth": a.depth, "top_k": a.top_k,
            "max_new_tokens": a.max_new_tokens, "naive": a.naive, "temperature": 0.0,
            "tau": tau, "rounds": len(all_acc),
            "tok_per_s": total_tok / total_time,
            "total_output_tokens": total_tok, "total_time_s": total_time,
            "avg_input_tokens": sum(r["prompt_tokens"] for r in rows) / max(len(rows), 1),
            "avg_output_tokens": total_tok / max(len(rows), 1),
            "acceptance_rates": acceptance_rates,
        },
        # Kept for the older readers that indexed these at the top level.
        "tau": tau, "rounds": len(all_acc), "tok_per_s": total_tok / total_time,
        "per_prompt": rows, "cfg": vars(a),
    }, open(a.out, "w"), indent=2, ensure_ascii=False)
