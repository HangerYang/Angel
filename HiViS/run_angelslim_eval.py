"""Acceptance length + throughput for an AngelSlim EAGLE-3 draft, measured
inside HiViS's own EaModel loop.

Every mode (tree, chain, --naive) runs through the same target backbone and the
same generation loop, so tok/s ratios across runs are a real speedup and not a
comparison of kernel stacks. See README_ANGELSLIM.md.

    python run_angelslim_eval.py --draft <ckpt> --n 40 --max_new_tokens 1024 \
        --total_token 60 --depth 5 --top_k 10       # tree
        --total_token 5  --depth 4 --top_k 1        # chain (K=4)
        --naive                                     # AR baseline
"""
import argparse, json, sys, time
import torch
from PIL import Image
from hivis.model.model_hivis import EaModel

OCR_PROMPT = ("Perform an OCR task on the provided image. Extract the text accurately "
              "and provide a detailed explanation of the process. Ensure the response "
              "is comprehensive and well-structured.")

def load_prompts(name, n):
    from datasets import load_dataset
    if name == "opendatalab/OmniDocBench":
        ds = load_dataset(name, split="train").select(range(n))
        return [(it["image"], OCR_PROMPT) for it in ds]
    if name == "MMMU/MMMU":
        ds = load_dataset(name, "History", split="test").select(range(n))
        return [(it["image_1"],
                 "Answer this question: %s Then describe the image in detail to justify your answer."
                 % it["question"].replace("<image 1>", "")) for it in ds]
    raise ValueError(name)

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="HuggingFaceTB/SmolVLM-256M-Instruct")
ap.add_argument("--draft", required=True)
ap.add_argument("--dataset", default="opendatalab/OmniDocBench")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--max_new_tokens", type=int, default=256)
ap.add_argument("--total_token", type=int, default=60)
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--top_k", type=int, default=10)
ap.add_argument("--out", default=None)
ap.add_argument("--naive", action="store_true",
                help="autoregressive baseline in the same harness (speedup denominator)")
a = ap.parse_args()

model = EaModel.from_pretrained(
    base_model_path=a.base, ea_model_path=a.draft,
    total_token=a.total_token, depth=a.depth, top_k=a.top_k,
    draft_method="angelslim_eagle3",
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda:0",
)
model.eval()
print("drafter:", type(model.ea_layer).__name__, "| aux layers:", model.aux_layer_ids)

proc = model.processor
rows, all_acc = [], []
for i, (img, q) in enumerate(load_prompts(a.dataset, a.n)):
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=text, images=[img.convert("RGB")], return_tensors="pt").to("cuda:0")
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

tau = sum(x + 1 for x in all_acc) / max(len(all_acc), 1)
print("\nmean acceptance length (tree, total_token=%d depth=%d top_k=%d) = %.4f  over %d rounds"
      % (a.total_token, a.depth, a.top_k, tau, len(all_acc)))
if a.out:
    json.dump({"tau": tau, "rounds": len(all_acc), "per_prompt": rows,
               "cfg": vars(a)}, open(a.out, "w"), indent=2)
