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
print("drafter:", type(model.ea_layer).__name__,
      "| method:", a.draft_method, "| aux layers:", getattr(model, "aux_layer_ids", None))

dataset = load_benchmark(a.dataset, sample_count=a.n)
rows, all_acc = [], []
for i in range(len(dataset)):
    inputs = prepare_inputs(model, dataset, i, a.dataset)
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
    rows.append({"tokens": int(new_token), "rounds": len(acc), "tau": tau, "time": dt})
    print("  [%d] tokens=%d rounds=%d tau=%.3f  %.2fs" % (i, new_token, len(acc), tau, dt))

tau = sum(x + 1 for x in all_acc) / max(len(all_acc), 1) if all_acc else 1.0
total_tok = sum(r["tokens"] for r in rows)
total_time = sum(r["time"] for r in rows)
print("\n%s | %s | total_token=%d depth=%d top_k=%d"
      % (a.draft_method, a.dataset, a.total_token, a.depth, a.top_k))
print("mean acceptance length = %.4f over %d rounds | %.2f tok/s | avg out %.1f"
      % (tau, len(all_acc), total_tok / total_time, total_tok / len(rows)))
if a.out:
    json.dump({"tau": tau, "rounds": len(all_acc), "tok_per_s": total_tok / total_time,
               "per_prompt": rows, "cfg": vars(a)}, open(a.out, "w"), indent=2)
