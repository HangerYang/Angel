import argparse
import copy
import json
import os
import queue
import random
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock

import base64
import requests
from PIL import Image
from tqdm import tqdm

from .model_names import model_directory_name


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate assistant texts with vLLM and write jsonl.")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="eval_data/sharegpt/sharegpt.jsonl",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--failed_jsonl",
        type=str,
        default=None,
    )
    parser.add_argument("--base_dataset_path", type=str, default="eval_data")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=68000)
    parser.add_argument("--vllm_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument(
        "--vllm_urls",
        type=str,
        nargs="+",
        default=[
            # "http://127.0.0.1:8000",
            "http://127.0.0.1:8001",
            "http://127.0.0.1:8002",
            "http://127.0.0.1:8003",
        ],
        help=(
            "One or more vLLM base URLs. When provided, requests are dynamically balanced across these endpoints. Defaults to localhost ports 8000-8003."
        ),
    )
    parser.add_argument("--vllm_model_name", type=str, default="llava-v1.6-vicuna-13b-hf")
    parser.add_argument("--model-path", "--model_path", dest="model_path", type=str, default=None)
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num_workers", type=int, default=192)
    parser.add_argument("--max_inflight", type=int, default=786)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    args = parser.parse_args()
    model_name = model_directory_name(args.model_path or args.vllm_model_name)
    output_dir = Path("eval_data/sharegpt") / model_name
    if args.output_jsonl is None:
        args.output_jsonl = str(output_dir / "sharegpt_regenerated.jsonl")
    if args.failed_jsonl is None:
        args.failed_jsonl = str(output_dir / "sharegpt_regenerated_failed.jsonl")
    return args


def normalize_vllm_urls(args):
    raw_urls = args.vllm_urls if args.vllm_urls else [args.vllm_url]
    urls = []
    for raw_url in raw_urls:
        # Also accept --vllm_urls url1,url2,url3 for convenience.
        for url in raw_url.split(","):
            url = url.strip().rstrip("/")
            if url and url not in urls:
                urls.append(url)
    if not urls:
        raise ValueError("At least one vLLM URL must be provided")
    return urls


class VLLMEndpointPool:
    """Dynamically balance concurrent HTTP requests across vLLM endpoints."""

    def __init__(self, urls, num_slots):
        self.urls = list(urls)
        self._available = queue.Queue()
        self._request_counts = {url: 0 for url in self.urls}
        self._count_lock = Lock()
        for slot_idx in range(num_slots):
            self._available.put(self.urls[slot_idx % len(self.urls)])

    @contextmanager
    def acquire(self):
        url = self._available.get()
        with self._count_lock:
            self._request_counts[url] += 1
        try:
            yield url
        finally:
            self._available.put(url)

    def request_counts(self):
        with self._count_lock:
            return dict(self._request_counts)


def load_image(base_path, image_path):
    if image_path is None:
        return None
    full_path = f"{base_path}/{image_path}"
    try:
        return Image.open(full_path).convert("RGB")
    except Exception as e:
        print(f"[warn] cannot open image: {full_path}, err={e}")
        return None


def strip_image_token(text):
    if text is None:
        return ""
    return text.replace("<image>", "").strip()


def build_messages(history_conversations, image=None):
    image_url = None
    if image is not None:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{image_b64}"

    messages = []
    image_attached = False
    for msg in history_conversations:
        role = msg.get("from", "")
        text = msg.get("value", "")
        if role == "human":
            content = []
            if image_url is not None and not image_attached:
                content.append({"type": "image_url", "image_url": {"url": image_url}})
                image_attached = True
            cleaned = strip_image_token(text)
            if cleaned:
                content.append({"type": "text", "text": cleaned})
            if not content:
                content.append({"type": "text", "text": ""})
            messages.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": text})
    return messages


def call_vllm_chat(messages, args, endpoint_pool):
    model_name = args.vllm_model_name if args.vllm_model_name else "default"
    max_tokens = args.max_tokens
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.vllm_api_key}",
    }
    for _ in range(3):
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": args.temperature,
        }
        with endpoint_pool.acquire() as base_url:
            if base_url.endswith("/v1"):
                endpoint = f"{base_url}/chat/completions"
            else:
                endpoint = f"{base_url}/v1/chat/completions"
            resp = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=args.timeout,
            )
        if resp.status_code < 400:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        err_text = resp.text
        if "maximum context length is 4096" in err_text and "in the messages" in err_text:
            m = re.search(r"\((\d+)\s+in the messages,\s+(\d+)\s+in the completion\)", err_text)
            if m is not None:
                message_tokens = int(m.group(1))
                allowed_completion = max(1, 4096 - message_tokens)
                if allowed_completion < max_tokens:
                    print(
                        f"[adjust] context overflow fixed: "
                        f"messages_tokens={message_tokens}, "
                        f"max_tokens {max_tokens} -> {allowed_completion}"
                    )
                    max_tokens = allowed_completion
                    continue
        if len(err_text) > 1200:
            err_text = err_text[:1200] + "...(truncated)"
        raise RuntimeError(f"vLLM HTTP {resp.status_code}: {err_text}")
    raise RuntimeError("vLLM HTTP 400: context too long after max_tokens adjustment")


def regenerate_record(record, args, endpoint_pool):
    conversations = record.get("conversations", None)
    if conversations is None:
        return record

    image = None
    image_relpath = record.get("image", None)
    if image_relpath is not None:
        image = load_image(args.base_dataset_path, image_relpath)

    rewritten = []
    for msg in conversations:
        role = msg.get("from", "")
        if role == "human":
            rewritten.append(copy.deepcopy(msg))
            continue

        if role not in ("gpt", "assistant"):
            rewritten.append(copy.deepcopy(msg))
            continue

        if len(rewritten) == 0 or rewritten[-1].get("from") != "human":
            rewritten.append(copy.deepcopy(msg))
            continue

        messages = build_messages(rewritten, image)
        generated_text = call_vllm_chat(messages, args, endpoint_pool)
        new_msg = copy.deepcopy(msg)
        new_msg["value"] = generated_text
        rewritten.append(new_msg)

    out = copy.deepcopy(record)
    out["conversations"] = rewritten
    return out


def process_record_line(task_idx, src_idx, line, args, endpoint_pool):
    line = line.strip()
    if not line:
        return task_idx, src_idx, "", None
    record = json.loads(line)
    try:
        new_record = regenerate_record(record, args, endpoint_pool)
        return task_idx, src_idx, json.dumps(new_record, ensure_ascii=False) + "\n", None
    except Exception as e:
        print(f"[warn] idx={src_idx} failed: {e}")
        failed_item = {
            "idx": src_idx,
            "error": str(e),
            "record": record,
        }
        # Keep output row count aligned with input range by writing original record.
        return task_idx, src_idx, json.dumps(record, ensure_ascii=False) + "\n", failed_item


def main():
    args = parse_args()
    for output_path in (args.output_jsonl, args.failed_jsonl):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    vllm_urls = normalize_vllm_urls(args)
    num_workers = max(1, args.num_workers, len(vllm_urls))
    endpoint_pool = VLLMEndpointPool(vllm_urls, num_slots=num_workers)
    print(f"vLLM endpoints ({len(vllm_urls)}): {vllm_urls}")
    print(f"HTTP workers: {num_workers}")

    with open(args.input_jsonl, "r", encoding="utf-8") as fcount:
        total_lines = sum(1 for _ in fcount)
    effective_end = total_lines if args.end == -1 else min(args.end, total_lines)
    effective_start = min(args.start, effective_end)
    total_to_process = max(0, effective_end - effective_start)

    selected_lines = []
    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if idx < args.start:
                continue
            if args.end != -1 and idx >= args.end:
                break
            selected_lines.append((idx, line))
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(selected_lines)

    written = 0
    failed_items = []
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        pbar = tqdm(total=total_to_process, desc="regenerating", unit="sample")
        max_inflight = max(1, args.max_inflight)
        next_write_idx = 0
        buffered = {}
        futures = set()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for task_idx, (src_idx, line) in enumerate(selected_lines):
                futures.add(
                    executor.submit(
                        process_record_line,
                        task_idx,
                        src_idx,
                        line,
                        args,
                        endpoint_pool,
                    )
                )
                if len(futures) < max_inflight:
                    continue

                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    done_task_idx, _, out_line, failed_item = fut.result()
                    buffered[done_task_idx] = out_line
                    if failed_item is not None:
                        failed_items.append(failed_item)
                    pbar.update(1)
                    while next_write_idx in buffered:
                        to_write = buffered.pop(next_write_idx)
                        if to_write:
                            fout.write(to_write)
                            written += 1
                        next_write_idx += 1

            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    done_task_idx, _, out_line, failed_item = fut.result()
                    buffered[done_task_idx] = out_line
                    if failed_item is not None:
                        failed_items.append(failed_item)
                    pbar.update(1)
                    while next_write_idx in buffered:
                        to_write = buffered.pop(next_write_idx)
                        if to_write:
                            fout.write(to_write)
                            written += 1
                        next_write_idx += 1
        pbar.close()
    if failed_items:
        with open(args.failed_jsonl, "w", encoding="utf-8") as ff:
            for item in sorted(failed_items, key=lambda x: x["idx"]):
                ff.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"done, wrote {written} lines to {args.output_jsonl}")
    print(f"failed: {len(failed_items)} lines, saved to {args.failed_jsonl}")
    print(f"requests by endpoint: {endpoint_pool.request_counts()}")


if __name__ == "__main__":
    main()
