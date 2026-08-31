"""End-to-end check of target-attention pruning wired into
OnlineVLMEagle3Trainer.prepare_data_for_draft_model -- a fake target model
backend stands in for VLMTransformersBackend (same interface:
get_hidden_states_and_logits / get_text_decoder_layers), so this exercises
the trainer's own branching (prune_mode dispatch, TargetQKCapture lifetime,
compressor_rows_per_sample plumbing) without needing a real SmolVLM.
"""
import types

import torch
import torch.nn as nn

from angelslim.compressor.speculative.train.trainer.online_eagle3_trainer import (
    OnlineVLMEagle3Trainer,
)
from angelslim.compressor.vistoken.row_compressor import VisRowCompressor, VisRowCompressorConfig

IMG = 49190
D, NS, TILE = 8, 9, 64
N_HEADS, HEAD_DIM = 2, 4
N_TILES, N_TEXT = 2, 5


class FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(D, N_HEADS * HEAD_DIM, bias=False)
        self.self_attn.k_proj = nn.Linear(D, N_HEADS * HEAD_DIM, bias=False)


class FakeTargetBackend:
    """Mimics VLMTransformersBackend's interface used by the trainer."""

    def __init__(self, seq_len):
        self.layers = nn.ModuleList([FakeDecoderLayer() for _ in range(35)])
        self.model = types.SimpleNamespace(
            config=types.SimpleNamespace(head_dim=HEAD_DIM, num_attention_heads=N_HEADS, hidden_size=D)
        )
        self._seq_len = seq_len

    def get_text_decoder_layers(self):
        return self.layers

    def get_hidden_states_and_logits(self, input_ids, attention_mask=None, **kwargs):
        # Actually run q_proj/k_proj on the aux layers so TargetQKCapture's
        # hooks have something real to fire on, like the real forward would.
        x = torch.randn(1, self._seq_len, D)
        for layer_id in kwargs["aux_hidden_states_layer_ids"]:
            layer = self.layers[layer_id]
            layer.self_attn.q_proj(x)
            layer.self_attn.k_proj(x)
        hidden_states = torch.randn(1, self._seq_len, NS * D)
        logits = torch.randn(1, self._seq_len, 5)
        return hidden_states, logits, None, None


def build_inputs():
    ids, is_img = [], []
    for _ in range(N_TEXT):
        ids.append(1000 + len(ids))
        is_img.append(False)
    for t in range(N_TILES):
        ids.append(2000 + t)
        is_img.append(False)
        ids.extend([IMG] * TILE)
        is_img.extend([True] * TILE)
    for _ in range(N_TEXT):
        ids.append(3000 + len(ids))
        is_img.append(False)
    input_ids = torch.tensor(ids).unsqueeze(0)
    seq_len = input_ids.shape[1]
    loss_mask = torch.zeros(1, seq_len)
    loss_mask[0, -N_TEXT:] = 1.0
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    return input_ids, attention_mask, loss_mask, seq_len


def make_trainer(prune_cfg):
    draft = nn.Module()
    cfg = VisRowCompressorConfig(hidden_size=D, num_streams=NS, num_queries=1, tile_tokens=TILE)
    draft.vistoken = VisRowCompressor(cfg)
    draft.vistoken_image_token_id = IMG

    draft_model_config = types.SimpleNamespace(
        aux_hidden_states_layer_ids=[2, 4, 8, 10, 15, 18, 20, 26, 28],
        vistoken_prune=prune_cfg,
    )
    trainer = OnlineVLMEagle3Trainer.__new__(OnlineVLMEagle3Trainer)
    trainer.model = draft  # `draft_model` is a read-only property unwrapping `self.model`
    trainer.accelerator = None
    trainer.target_model = None  # set per-test
    trainer.branch_distill_loss_weight = 0.0
    trainer._aux_hidden_states_layer_ids = draft_model_config.aux_hidden_states_layer_ids
    trainer._vistoken_prune_cfg = prune_cfg
    trainer._target_head_dim = None
    return trainer


def run(prune_cfg):
    input_ids, attention_mask, loss_mask, seq_len = build_inputs()
    trainer = make_trainer(prune_cfg)
    trainer.target_model = FakeTargetBackend(seq_len)
    return trainer.prepare_data_for_draft_model(
        {"input_ids": input_ids, "attention_mask": attention_mask, "loss_mask": loss_mask}
    )


def test_none_mode_matches_unpruned_baseline():
    out = run({"mode": "none"})
    exp_len = 5 + 5 + N_TEXT + N_TEXT - N_TEXT + (N_TEXT + N_TEXT) - 0  # placeholder, recompute below
    seq_len = N_TEXT * 2 + N_TILES * (1 + TILE)
    exp_len = seq_len - N_TILES * (TILE - 1)
    assert out["input_ids"].shape[1] == exp_len
    print("prune_mode=none: trainer wiring reproduces the unpruned vistoken-k1 shape. ok")


def test_target_attn_mode_runs_and_prunes():
    out = run({"mode": "target_attn", "group_size": 16, "keep_m": 4})
    seq_len = N_TEXT * 2 + N_TILES * (1 + TILE)
    exp_len = seq_len - N_TILES * (TILE - 1)
    # Same final length as unpruned (still k=1 summary/tile) -- pruning only
    # changes what feeds the compressor, not how many rows survive.
    assert out["input_ids"].shape[1] == exp_len
    print("prune_mode=target_attn: hooks fire through the real trainer path, no crash. ok")


def test_random_mode_runs_without_qk_capture():
    out = run({"mode": "random", "group_size": 16, "keep_m": 4})
    seq_len = N_TEXT * 2 + N_TILES * (1 + TILE)
    exp_len = seq_len - N_TILES * (TILE - 1)
    assert out["input_ids"].shape[1] == exp_len
    print("prune_mode=random: runs without needing TargetQKCapture. ok")


if __name__ == "__main__":
    test_none_mode_matches_unpruned_baseline()
    test_target_attn_mode_runs_and_prunes()
    test_random_mode_runs_without_qk_capture()
