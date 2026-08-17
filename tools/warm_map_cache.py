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
"""Build the final train/eval .map_cache arrow files in a single process.

train_eagle3_online.py builds the online dataset independently on every
torchrun rank with no rank-0-first gating, so an NPROC>1 launch redundantly
repeats this preprocessing pass on every rank. Running it once here -- via
the same DatasetBuilderFactory.build_dataset() path training uses, so the
pinned cache_file_name matches exactly -- means every rank hits a warm cache
instead of recomputing.

Run this AFTER scripts/speculative/smolvlm/prewarm_gist_cache.sh has merged
the oracle gist embedding cache: with that warm, every gist lookup here is a
cache hit and this pass is CPU-only tokenization (no GPU encoder needed).
"""

import argparse
import json
from pathlib import Path

from transformers import AutoProcessor

from angelslim.compressor.speculative.train.data.chat_templates import (
    string_to_chat_template_type,
)
from angelslim.compressor.speculative.train.data.dataset_builder import (
    DatasetBuilderFactory,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft_model_config_path", required=True)
    p.add_argument("--target_model_name_or_path", required=True)
    p.add_argument("--train_data_path", required=True)
    p.add_argument("--eval_data_path", default="")
    p.add_argument("--model_max_length", type=int, default=4096)
    p.add_argument("--chat_template_type", default="smolvlm")
    p.add_argument("--modal_type", default="VLM")
    p.add_argument("--shuffle_seed", type=int, default=42)
    p.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="HF datasets map workers. Only takes effect for gist-conditioned "
        "configs if --gist_allow_multiproc is also set (otherwise the "
        "builder forces num_proc=None for encoder safety).",
    )
    p.add_argument(
        "--gist_allow_multiproc",
        action="store_true",
        default=False,
        help="Skip the gist_conditioning num_proc=None safety forcing. Only "
        "safe when the on-disk gist cache is already known to cover every "
        "text in train/eval -- verify with the prewarm script first.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    draft_cfg = json.loads(Path(args.draft_model_config_path).read_text())
    target_model_type = draft_cfg.get("target_model_type")

    print(f"Loading processor for {args.target_model_name_or_path}", flush=True)
    tokenizer = AutoProcessor.from_pretrained(args.target_model_name_or_path, trust_remote_code=True)

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
        gist_conditioning=draft_cfg.get("gist_conditioning", False),
        gist_encoder_model_name_or_path=draft_cfg.get(
            "gist_encoder_model_name_or_path", "Qwen/Qwen3-Embedding-0.6B"
        ),
        gist_refresh_every=draft_cfg.get("gist_refresh_every", 4),
        gist_encoder_device=draft_cfg.get("gist_encoder_device", "cuda:0"),
        gist_batch_size=draft_cfg.get("gist_batch_size", 32),
        gist_embedding_dim=draft_cfg.get("gist_embedding_dim", 0),
        gist_cache_dir=draft_cfg.get("gist_cache_dir"),
        gist_allow_multiproc=args.gist_allow_multiproc,
    )

    print(
        f"Building train .map_cache for {args.train_data_path} "
        f"(num_proc={args.num_proc}, gist_allow_multiproc={args.gist_allow_multiproc})",
        flush=True,
    )
    train_ds = builder.build_dataset(
        args.train_data_path,
        num_proc=args.num_proc,
        shuffle=True,
        sample_num=None,
        min_loss_tokens=None,
        load_from_cache_file=True,
    )
    print(f"train: {len(train_ds)} examples cached", flush=True)

    if args.eval_data_path:
        print(f"Building eval .map_cache for {args.eval_data_path}", flush=True)
        eval_ds = builder.build_dataset(
            args.eval_data_path,
            num_proc=args.num_proc,
            shuffle=False,
            sample_num=None,
            min_loss_tokens=None,
            load_from_cache_file=True,
        )
        print(f"eval: {len(eval_ds)} examples cached", flush=True)


if __name__ == "__main__":
    main()
