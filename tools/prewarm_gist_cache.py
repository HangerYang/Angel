# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pre-warm one shard of the oracle gist embedding cache.

Online Eagle3 training with ``gist_conditioning=true`` encodes text spans with
a SentenceTransformer on first use and caches the vectors on disk. That cache
is keyed only by (encoder model, gist_refresh_every, text) -- not by which
file the text came from -- so it can be built in parallel, sharded across
GPUs, ahead of time, then merged.

This script deliberately bypasses ``builder.build_dataset()``'s
``ds.map(cache_file_name=...)`` path: that cache_file_name is derived from the
*whole* input file (path/size/mtime), so every shard worker would compute the
same cache_file_name and race on the same output file. Instead it loads the
raw conversations, shards them with ``Dataset.shard()``, and calls the
builder's own per-example ``_process_single_conversation`` directly -- the
exact function ``build_dataset()`` calls internally -- so the resulting gist
cache is byte-identical to what a normal run would produce, just computed
concurrently across GPUs.

Driven by scripts/speculative/smolvlm/prewarm_gist_cache.sh -- see that script
for sharding, parallel launch, and merging the per-shard caches together.
"""

import argparse
import json
import time
from pathlib import Path

from datasets import Features, Value, load_dataset
from transformers import AutoProcessor

from angelslim.compressor.speculative.train.data.chat_templates import (
    string_to_chat_template_type,
)
from angelslim.compressor.speculative.train.data.dataset_builder import (
    DatasetBuilderFactory,
)

# Must match the Features schema in OnlineSmolVLMDatasetBuilder.build_dataset
# (online_dataset_builder.py) so shard workers parse rows identically.
CONVERSATION_FEATURES = Features(
    {
        "id": Value("string"),
        "conversations": [
            {
                "role": Value("string"),
                "content": [
                    {
                        "type": Value("string"),
                        "text": Value("string"),
                        "image": Value("string"),
                        "image_url": {"url": Value("string")},
                    }
                ],
            }
        ],
    }
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft_model_config_path", required=True)
    p.add_argument("--target_model_name_or_path", required=True)
    p.add_argument(
        "--data_path", nargs="+", required=True, help="jsonl file(s) forming the shard pool"
    )
    p.add_argument(
        "--gist_cache_dir", required=True, help="Unique scratch dir for this shard's cache file"
    )
    p.add_argument("--device", required=True, help="e.g. cuda:0 (after CUDA_VISIBLE_DEVICES)")
    p.add_argument("--model_max_length", type=int, default=4096)
    p.add_argument("--chat_template_type", default="smolvlm")
    p.add_argument("--modal_type", default="VLM")
    p.add_argument("--shuffle_seed", type=int, default=42)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument(
        "--save_every", type=int, default=2000, help="Flush cache to disk every N conversations"
    )
    p.add_argument(
        "--row_offset",
        type=int,
        default=0,
        help="Skip this many rows within the selected shard before processing "
        "(for resuming/re-splitting an interrupted shard's tail)",
    )
    p.add_argument(
        "--row_limit",
        type=int,
        default=None,
        help="Process at most this many rows after --row_offset",
    )
    return p.parse_args()


def main():
    args = parse_args()
    tag = f"[shard {args.shard_index}/{args.num_shards}]"

    draft_cfg = json.loads(Path(args.draft_model_config_path).read_text())
    if not draft_cfg.get("gist_conditioning"):
        raise SystemExit(
            f"{args.draft_model_config_path} has gist_conditioning=false; nothing to prewarm"
        )
    target_model_type = draft_cfg.get("target_model_type")

    print(f"{tag} loading processor for {args.target_model_name_or_path}", flush=True)
    tokenizer = AutoProcessor.from_pretrained(args.target_model_name_or_path, trust_remote_code=True)

    gist_kwargs = dict(
        gist_conditioning=True,
        gist_encoder_model_name_or_path=draft_cfg.get(
            "gist_encoder_model_name_or_path", "Qwen/Qwen3-Embedding-0.6B"
        ),
        gist_refresh_every=draft_cfg.get("gist_refresh_every", 4),
        gist_encoder_device=args.device,
        gist_batch_size=draft_cfg.get("gist_batch_size", 32),
        gist_embedding_dim=draft_cfg.get("gist_embedding_dim", 0),
        gist_cache_dir=args.gist_cache_dir,
    )

    builder = DatasetBuilderFactory.create(
        training_mode="online",
        modal_type=args.modal_type,
        target_model_type=target_model_type,
        tokenizer=tokenizer,
        max_length=args.model_max_length,
        shuffle_seed=args.shuffle_seed,
        chat_template_type=string_to_chat_template_type(args.chat_template_type),
        display=False,
        target_model_name_or_path=args.target_model_name_or_path,
        output_dir=None,
        **gist_kwargs,
    )

    builder._prepare_gist_cache(args.data_path)

    print(f"{tag} loading {args.data_path}", flush=True)
    ds = load_dataset("json", data_files=args.data_path, features=CONVERSATION_FEATURES)["train"]
    if args.num_shards > 1:
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_index, contiguous=True)
    if args.row_offset or args.row_limit is not None:
        end = len(ds) if args.row_limit is None else min(len(ds), args.row_offset + args.row_limit)
        ds = ds.select(range(args.row_offset, end))
    print(f"{tag} {len(ds)} conversations to scan on {args.device}", flush=True)

    t0 = time.time()
    n_before = len(builder._gist_cache)
    n_rows = len(ds)
    for i, row in enumerate(ds):
        try:
            builder._process_single_conversation(row["conversations"])
        except Exception as e:  # noqa: BLE001 - keep scanning past bad rows
            print(f"{tag} row {i} error: {e}", flush=True)
        if (i + 1) % args.save_every == 0 or (i + 1) == n_rows:
            builder._save_gist_cache()
            elapsed = time.time() - t0
            n_new = len(builder._gist_cache) - n_before
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"{tag} {i + 1}/{n_rows} rows ({rate:.1f}/s), "
                f"+{n_new} new gist entries, {elapsed:.0f}s elapsed",
                flush=True,
            )

    builder._unload_gist_encoder()
    print(
        f"{tag} done: {len(builder._gist_cache)} total gist entries -> {builder._gist_cache_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
