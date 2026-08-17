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
"""Standalone, cheap training run for Medusa-style MTP heads (Phase 2), on the
ONLINE data pipeline -- same jsonl -> tokenization path as
scripts/speculative/smolvlm/train_eagle3_vlm_online.sh, so pointing this at the
same TRAIN_DATA_PATH/EVAL_DATA_PATH/MODEL_MAX_LENGTH/CHAT_TEMPLATE_TYPE/
SHUFFLE_SEED as an existing real run hits its already-warm .map_cache instead
of rebuilding anything.

Loads an existing, already-trained EAGLE3 draft checkpoint, freezes the whole
trunk (+ the target model, always frozen), and trains only a few small extra
heads (mtp_heads.MTPHeads) that predict t+2, t+3, ... from a SINGLE trunk
forward pass -- no re-encoding loop, unlike Eagle3Trainer's multi-step
training-time-test. Deliberately NOT built on Eagle3Trainer's own training
loop for that reason; it only reuses Eagle3Trainer's (well-tested) per-batch
data-prep methods (prepare_data_for_draft_model / down_project_hidden_states /
prepare_attention_mask_and_position_ids) so the online target-model-forward,
VLM image handling, and hidden-state extraction logic isn't reimplemented.

Explicitly scoped to a standalone accuracy check before any speculative-
pipeline integration -- see mtp_heads.py and the plan doc for what this is /
is not meant to prove.
"""

import argparse
import os

import torch
import transformers
from torch.utils.data import DataLoader

from angelslim.compressor.speculative import (
    DatasetManager,
    Eagle3TrainerFactory,
    create_target_model,
)
from angelslim.compressor.speculative.train.models.draft.llama_eagle3 import (
    Eagle3LlamaForCausalLM,
    LlamaRotaryEmbedding,
)
from angelslim.compressor.speculative.train.models.draft.mtp_heads import (
    MTPHeads,
    apply_mtp_training_setup,
)
from angelslim.compressor.speculative.utils import padding
from angelslim.utils import rank0_print

# Mirrors tools/train_eagle3_online.py's draft-config -> args plumbing for
# gist_conditioning, so DatasetManager builds the SAME dataset a real online
# run for this checkpoint's config would (same .map_cache key).
_GIST_CONFIG_FIELDS = (
    "gist_conditioning",
    "gist_encoder_model_name_or_path",
    "gist_refresh_every",
    "gist_encoder_device",
    "gist_batch_size",
    "gist_embedding_dim",
    "gist_cache_dir",
)


def _refresh_rope_caches(model: torch.nn.Module) -> int:
    """Force-recompute every LlamaRotaryEmbedding's inv_freq AND cos/sin cache.

    ``inv_freq``/``cos_cached``/``sin_cached`` are registered with
    ``persistent=False`` (correctly -- they're deterministic, no need to save
    them). But that means ``Eagle3LlamaForCausalLM.from_pretrained`` (which
    loads via HF's low_cpu_mem_usage / meta-device path) does not reliably
    re-run the ``__init__``-time computation for them: confirmed empirically
    -- after ``from_pretrained`` on a real, fully trained checkpoint,
    ``rotary_emb.inv_freq`` itself came back as uninitialized garbage memory
    (values like 6e19, -4e31 -- not the deterministic 1/base**(...) formula),
    which then poisons ``cos_cached``/``sin_cached`` (and, downstream, every
    attention layer) even though a freshly *constructed* (not loaded)
    LlamaRotaryEmbedding with identical parameters is fine. Recomputing just
    ``cos_cached``/``sin_cached`` from the still-garbage ``inv_freq`` is NOT
    enough (tried that first) -- ``inv_freq`` has to be rebuilt too. Returns
    how many modules were refreshed (for a sanity log line).
    """
    n = 0
    for module in model.modules():
        if isinstance(module, LlamaRotaryEmbedding):
            ref = next(module.buffers(), None)
            device = ref.device if ref is not None else torch.device("cpu")
            dtype = torch.get_default_dtype()
            fresh_inv_freq = 1.0 / (
                module.base
                ** (torch.arange(0, module.dim, 2, device=device).float() / module.dim)
            )
            module.inv_freq = fresh_inv_freq
            module._set_cos_sin_cache(
                seq_len=module.max_position_embeddings, device=device, dtype=dtype
            )
            n += 1
    return n


def parse_args():
    parser = argparse.ArgumentParser(description="Train Medusa-style MTP heads (Phase 2, online)")

    parser.add_argument("--modal_type", type=str, default="VLM", choices=["LLM", "VLM"])
    parser.add_argument(
        "--draft_checkpoint_path",
        type=str,
        required=True,
        help="Path to an existing trained draft checkpoint dir (any variant -- "
        "not tied to a specific Phase-1 config). Its saved config (incl. "
        "eagle_aux_injection_mode, aux_hidden_states_layer_ids, gist_* fields) "
        "is reused as-is to build the dataset, so no separate "
        "--draft_model_config_path is needed.",
    )
    parser.add_argument("--target_model_name_or_path", type=str, required=True)
    parser.add_argument("--num_extra_heads", type=int, default=2, help="K-1, e.g. 2 for K=3")

    parser.add_argument("--train_data_path", type=str, nargs="+", required=True)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--chat_template_type", type=str, default="smolvlm")
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument(
        "--load_from_cache_file",
        type=lambda v: str(v).lower() in ("1", "true", "t", "yes", "y"),
        default=True,
        help="Reuse HF datasets map/filter cache (default true -- this is the "
        "whole point of the online port: hit the cache the real training runs "
        "already built, never rebuild).",
    )
    parser.add_argument("--display", action="store_true", default=False)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--discount", type=float, default=0.8, help="Per-offset loss discount, weight[k]=discount**k"
    )

    return parser.parse_args()


def mtp_train_step(trainer, draft, mtp_heads, batch, num_extra_heads, discount):
    """One MTP training step: one frozen trunk forward (target + draft), K-1
    parallel head losses on top of it.

    Reuses ``trainer``'s per-batch data-prep (the same methods
    Eagle3Trainer.compute_loss calls before its own multi-step loop) so the
    online target-model-forward / VLM handling / hidden-state extraction is
    the exact, already-tested code path -- not reimplemented here.
    """
    # Some draft variants (e.g. progressive_banded_mix) keep a few direct
    # parameters (band mix logits) in fp32 for softmax stability while the
    # rest of the model is bf16. combine_hidden_states picks its cast dtype
    # from next(self.parameters()).dtype, which can land on one of those fp32
    # params (PyTorch yields a module's own direct parameters before its
    # submodules'). Real bf16 training never hits this because Trainer wraps
    # the forward pass in autocast, which tolerates the mismatch; do the same
    # here rather than patching the shared model file for a plain eager call.
    # lm_head is always a plain Linear in the real working dtype -- unlike
    # next(draft.parameters()).dtype, it's never a stray fp32-for-stability
    # direct parameter (e.g. progressive_banded_mix's band mix logits).
    autocast_dtype = draft.lm_head.weight.dtype
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=autocast_dtype):
        # The real Trainer.train() loop moves batch tensors to device via
        # _prepare_inputs before compute_loss; we bypass that loop, so do it
        # ourselves (reusing the same HF Trainer method, not reimplementing it).
        batch = trainer._prepare_inputs(batch)
        data_for_draft = trainer.prepare_data_for_draft_model(batch)
        target_logits = data_for_draft["target_logits"]
        input_ids = data_for_draft["input_ids"]
        loss_mask = data_for_draft["loss_mask"]
        position_ids = data_for_draft["position_ids"]
        attention_mask = data_for_draft["attention_mask"]
        gist_embeddings = data_for_draft.get("gist_embeddings")

        hidden_states = trainer.down_project_hidden_states(
            data_for_draft["hidden_states"], gist_embeddings
        )
        attention_mask, position_ids = trainer.prepare_attention_mask_and_position_ids(
            hidden_states, attention_mask, position_ids
        )

        inputs_embeds = draft.embed_input_ids(input_ids)
        cache_hidden = draft.init_cache_hidden()
        h_final, _ = draft.encode_layers(
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            gist_embeddings=gist_embeddings,
        )
        # Keep it in the real working dtype (not autocast's possibly-fp32
        # internal choice for this op) so it matches mtp_heads' own dtype.
        h_final = h_final.detach().to(autocast_dtype)

    per_head_losses, per_head_acc = [], []
    shifted_target_logits, shifted_loss_mask = target_logits, loss_mask
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        logits_per_head = mtp_heads(h_final, draft.norm, draft.lm_head)

        for k in range(num_extra_heads):
            # head k predicts t+2+k; prepare_data_for_draft_model already
            # shifted target_logits once (t+1 alignment), so shift (k+1)
            # times total.
            shifted_target_logits = padding(shifted_target_logits, left=False)
            shifted_loss_mask = padding(shifted_loss_mask, left=False)

            with torch.no_grad():
                target_max_token = shifted_target_logits.argmax(-1)
                target_mask = draft.t2d[target_max_token][..., None].int()
                position_mask = target_mask * shifted_loss_mask
                target_head_logits = shifted_target_logits[..., draft.t2d].float()
                target_p = torch.nn.functional.softmax(target_head_logits, dim=-1).detach()

            out_logp = torch.nn.functional.log_softmax(logits_per_head[k], dim=-1)
            head_loss = -torch.sum(position_mask * target_p * out_logp, dim=-1).mean()
            per_head_losses.append(head_loss)

            with torch.no_grad():
                correct = (
                    logits_per_head[k].argmax(-1) == target_p.argmax(-1)
                ) * position_mask.squeeze(-1)
                acc = correct.sum().item() / (shifted_loss_mask.sum().item() + 1e-6)
            per_head_acc.append(acc)

    weights = [discount**k for k in range(num_extra_heads)]
    total_loss = sum(w * l for w, l in zip(weights, per_head_losses))
    return total_loss, per_head_losses, per_head_acc


def train():
    args = parse_args()

    rank0_print(f"Loading draft checkpoint: {args.draft_checkpoint_path}")
    draft = Eagle3LlamaForCausalLM.from_pretrained(args.draft_checkpoint_path)
    draft_model_config = draft.config
    target_model_type = getattr(draft_model_config, "target_model_type", None)
    rank0_print(
        f"progressive_staged={draft.progressive_staged} "
        f"progressive_per_layer_fc={getattr(draft, 'progressive_per_layer_fc', False)} "
        f"progressive_banded_mix={draft.progressive_banded_mix} hawk={draft.hawk} "
        f"norm_output={draft.norm_output}"
    )

    mtp_heads = MTPHeads(hidden_size=draft.hidden_size, num_extra_heads=args.num_extra_heads)
    setup_summary = apply_mtp_training_setup(draft, mtp_heads)
    rank0_print(
        f"MTP heads trainable params: {setup_summary['num_trainable']} / "
        f"{setup_summary['num_total']} total "
        f"({100 * setup_summary['num_trainable'] / setup_summary['num_total']:.3f}%)"
    )

    for name in _GIST_CONFIG_FIELDS:
        if hasattr(draft_model_config, name):
            setattr(args, name, getattr(draft_model_config, name))
    # DatasetManager checks data_args.training_mode == "offline" to decide
    # whether to also build an offline_dataset_builder; we only ever want
    # the online path here.
    args.training_mode = "online"

    rank0_print("Loading target model...")
    target_model = create_target_model(
        backend="hf",
        model_path=args.target_model_name_or_path,
        modal_type=args.modal_type,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        target_model_type=target_model_type,
    )
    for p in target_model.model.parameters():
        p.requires_grad_(False)
    target_model.model.eval()

    device = next(target_model.model.parameters()).device
    draft = draft.to(device)
    n_rope = _refresh_rope_caches(draft)
    rank0_print(f"Recomputed {n_rope} RoPE cos/sin cache(s) after from_pretrained load")
    # lm_head.weight.dtype, not next(draft.parameters()).dtype -- see the
    # comment in mtp_train_step on why the latter is unreliable for some
    # draft variants (e.g. progressive_banded_mix's fp32 band mix logits).
    mtp_heads = mtp_heads.to(device=device, dtype=draft.lm_head.weight.dtype)
    rank0_print(f"draft + mtp_heads moved to {device}, dtype={draft.lm_head.weight.dtype}")

    rank0_print("Building online datasets (same pipeline as train_eagle3_vlm_online.sh)...")
    dataset_manager = DatasetManager(
        data_args=args,
        tokenizer=target_model.tokenizer,
        model_max_length=args.model_max_length,
        chat_template_type=args.chat_template_type,
        target_model_type=target_model_type,
    )
    train_dataset, eval_dataset, data_collator = dataset_manager.create_online_datasets()
    rank0_print(f"train={len(train_dataset)} eval={len(eval_dataset) if eval_dataset else 0}")

    # Minimal TrainingArguments purely to satisfy Eagle3Trainer's (HF Trainer)
    # __init__ -- we never call trainer.train(); only its per-batch data-prep
    # helper methods are reused, in our own loop below.
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Eagle3TrainerFactory.create(
        training_mode="online",
        modal_type=args.modal_type,
        draft_model=draft,
        target_model=target_model,
        length=1,
        draft_model_config=draft_model_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    optimizer = torch.optim.AdamW(
        mtp_heads.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    mtp_heads.train()
    draft.eval()
    step = 0
    for epoch in range(args.num_train_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            loss, per_head_losses, per_head_acc = mtp_train_step(
                trainer, draft, mtp_heads, batch, args.num_extra_heads, args.discount
            )
            loss.backward()
            optimizer.step()
            step += 1
            if step % args.logging_steps == 0:
                loss_str = " ".join(f"t+{2+k}={l.item():.4f}" for k, l in enumerate(per_head_losses))
                acc_str = " ".join(f"t+{2+k}={a:.4f}" for k, a in enumerate(per_head_acc))
                rank0_print(f"step={step} epoch={epoch} loss={loss.item():.4f} | {loss_str} | acc: {acc_str}")

    rank0_print(f"Saving mtp_heads to {args.output_dir}")
    torch.save(
        {
            "state_dict": mtp_heads.state_dict(),
            "draft_checkpoint_path": args.draft_checkpoint_path,
            "num_extra_heads": args.num_extra_heads,
            "hidden_size": draft.hidden_size,
        },
        os.path.join(args.output_dir, "mtp_heads.pt"),
    )
    rank0_print("Done.")


if __name__ == "__main__":
    train()
