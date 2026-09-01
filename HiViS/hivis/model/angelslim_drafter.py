"""Run an AngelSlim EAGLE-3 draft checkpoint inside HiViS's EaModel.

HiViS's EaModel drives its drafter through EAGLE-2's interface -- init_tree(),
reset_kv(), topK_genrate(hidden_states, input_ids, ...) returning
(draft_tokens, retrieve_indices, tree_mask, tree_position_ids). AngelSlim
already ships a drafter with exactly that interface in
``angelslim/compressor/speculative/inference/models/eagle3/draft``, so this
module reuses it rather than reimplementing tree decoding.

Two things stand between that class and our checkpoint:

1. ``import angelslim`` executes angelslim/__init__.py, which reaches
   qat/modules/quantizer.py and evaluates a py3.10-only ``X | None`` annotation
   at class-body time. HiViS runs on py3.9, so importing normally raises
   TypeError. The draft modules themselves are py3.9-clean, so
   ``load_angelslim_draft_module`` registers empty stand-in packages and
   imports the leaf module directly, skipping every __init__ on the way.

2. Our drafts use ``eagle_aux_injection_mode: banded_mix_fc``: the target's 9
   aux streams are first collapsed to one stream per band by a learned softmax
   (band{i}_mix_logits), and only those band streams reach the stock EAGLE-3.1
   fc_norm + nH->H fusion FC. The stock drafter sizes fc and fc_norm from
   len(aux_hidden_states_layer_ids) == 9, so it builds Linear(9H, H) and 9
   norms where the checkpoint holds Linear(3H, H) and 3 norms.

(2) is handled without touching the stock forward: the band mix runs in
topK_genrate before delegating upward, and the config handed to the base class
advertises the BAND count as its aux-stream count. The base forward already
applies fc_norm+fc only when the incoming hidden is wider than the embedding
(i.e. only on the first draft step, exactly as training does) and feeds
_next_step_hidden -- post-norm under EAGLE 3.1 ``norm_output`` -- to later
steps, so nothing downstream needs to change.
"""

import json
import os
import sys
import types

import torch
from torch import nn

_ANGELSLIM_ROOT_ENV = "ANGELSLIM_ROOT"
_DRAFT_PKG = "angelslim.compressor.speculative.inference.models.eagle3.draft"


def _default_root():
    """Walk up from this file looking for the repo that contains `angelslim/`.

    HiViS is vendored inside the AngelSlim checkout, so the repo root is a
    couple of levels up -- but not at a fixed absolute path, since this runs on
    several machines. Fall back to the cwd's repo if the layout ever changes.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    for candidate in (here, os.getcwd()):
        while True:
            if os.path.isdir(os.path.join(candidate, "angelslim")):
                return candidate
            parent = os.path.dirname(candidate)
            if parent == candidate:
                break
            candidate = parent
    return here


def load_angelslim_draft_module(angelslim_root=None):
    """Import the AngelSlim inference drafter without running angelslim/__init__."""
    import importlib

    root = angelslim_root or os.environ.get(_ANGELSLIM_ROOT_ENV) or _default_root()
    if not os.path.isdir(os.path.join(root, "angelslim")):
        raise RuntimeError(
            "Could not find an angelslim checkout at %r. Pass angelslim_root= or set %s."
            % (root, _ANGELSLIM_ROOT_ENV)
        )
    if root not in sys.path:
        sys.path.insert(0, root)

    parts = _DRAFT_PKG.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name in sys.modules:
            continue
        stub = types.ModuleType(name)
        stub.__path__ = [os.path.join(root, *parts[:i])]
        sys.modules[name] = stub
    return importlib.import_module(_DRAFT_PKG + ".llama3_eagle3")


class DrafterConfig(object):
    """Plain attribute bag for the drafter.

    Deliberately not transformers' LlamaConfig: HiViS pins transformers 4.54
    while the checkpoints were written by 5.x, and the two disagree on where
    RoPE settings live. Translating explicitly here keeps that disagreement
    from silently changing the RoPE base -- the checkpoint carries
    ``rope_parameters: {"rope_theta": 100000.0}`` (5.x), while the drafter
    reads ``config.rope_theta`` and would otherwise fall back to 10000.
    """

    def __init__(self, raw, num_aux_streams):
        self.hidden_size = raw["hidden_size"]
        self.intermediate_size = raw["intermediate_size"]
        self.num_attention_heads = raw["num_attention_heads"]
        self.num_key_value_heads = raw["num_key_value_heads"]
        self.head_dim = raw.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = raw.get("hidden_act", "silu")
        # Read by the drafter's tensor-parallel branch, which is inert at tp=1.
        self.pretraining_tp = raw.get("pretraining_tp", 1)
        self.rms_norm_eps = raw.get("rms_norm_eps", 1e-5)
        self.max_position_embeddings = raw.get("max_position_embeddings", 8192)
        self.vocab_size = raw["vocab_size"]
        self.draft_vocab_size = raw.get("draft_vocab_size", self.vocab_size)
        self.pad_token_id = raw.get("pad_token_id", 0)
        self.target_hidden_size = raw.get("target_hidden_size", self.hidden_size)

        rope = raw.get("rope_parameters") or {}
        self.rope_theta = raw.get("rope_theta", rope.get("rope_theta", 10000.0))
        scaling = raw.get("rope_scaling")
        if scaling is None:
            rope_type = rope.get("rope_type", "default")
            scaling = None if rope_type in (None, "default") else dict(rope)
        self.rope_scaling = scaling

        # EAGLE 3.1 switches.
        self.fc_norm = bool(raw.get("fc_norm", False))
        self.norm_output = bool(raw.get("norm_output", False))
        # The base class sizes fc / fc_norm off this list's LENGTH. Under
        # banded_mix_fc the streams reaching the FC are the BANDS, not the raw
        # aux layers, so advertise the band count here. The real 9-layer list
        # stays on the wrapper as `aux_hidden_states_layer_ids`.
        self.aux_hidden_states_layer_ids = list(range(num_aux_streams))


class HiViSInterfaceMixin(object):
    """Adapts AngelSlim's drafter to what HiViS's tree loop expects.

    Two adjustments, both at the topK_genrate boundary:
      * arity -- AngelSlim returns a 5th value (early_stop_signal); HiViS
        unpacks exactly four.
      * banded_mix_fc -- collapse the raw aux streams to one per band before
        the stock fc_norm + FC sees them (no-op when the checkpoint has no
        bands).
    """

    def init_banded_mix(self, bands):
        self.aux_layer_bands = tuple(tuple(b) for b in bands)
        for i, band in enumerate(self.aux_layer_bands):
            self.register_parameter(
                "band%d_mix_logits" % i,
                nn.Parameter(torch.zeros(len(band))),
            )

    def mix_aux(self, hidden_states):
        """Collapse the 9 raw aux streams to one per band via a learned softmax.

        Pass-through when the width is not the raw aux concat: later draft
        steps hand back a single H-wide hidden, which must not be re-mixed.
        """
        bands = getattr(self, "aux_layer_bands", ())
        if not bands:
            return hidden_states
        h = self.config.target_hidden_size
        n_raw = sum(len(b) for b in bands)
        if hidden_states.shape[-1] != h * n_raw:
            return hidden_states
        chunks = hidden_states.split(h, dim=-1)
        mixed, off = [], 0
        for i, band in enumerate(bands):
            stack = torch.stack(chunks[off : off + len(band)], dim=-1)
            w = torch.softmax(getattr(self, "band%d_mix_logits" % i).float(), dim=0)
            mixed.append(torch.matmul(stack, w.to(stack.dtype)))
            off += len(band)
        return torch.cat(mixed, dim=-1)

    def topK_genrate(self, hidden_states, input_ids, inputs_embeds=None, logits_processor=None):
        out = super(HiViSInterfaceMixin, self).topK_genrate(
            self.mix_aux(hidden_states), input_ids, inputs_embeds, logits_processor
        )
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = out[:4]
        if len(out) > 4 and out[4] is not None:
            raise NotImplementedError(
                "early-stop drafts are not wired into HiViS's tree loop "
                "(early_stop_signal was not None)"
            )
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def build_drafter(checkpoint_dir, total_tokens=60, depth=5, top_k=10,
                  threshold=1.0, angelslim_root=None, dtype=torch.bfloat16,
                  device=None):
    """Load an AngelSlim EAGLE-3 checkpoint as a HiViS-compatible drafter."""
    mod = load_angelslim_draft_module(angelslim_root)

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        raw = json.load(f)

    mode = raw.get("eagle_aux_injection_mode", "fused_fc")
    bands = raw.get("eagle_aux_layer_bands")
    if mode == "banded_mix_fc":
        if not bands:
            raise ValueError("banded_mix_fc checkpoint without eagle_aux_layer_bands")
        n_streams = len(bands)
    elif mode in ("fused_fc", None):
        bands = None
        n_streams = len(raw.get("aux_hidden_states_layer_ids") or [0, 1, 2])
    else:
        raise NotImplementedError(
            "eagle_aux_injection_mode=%r is not supported by this adapter yet "
            "(supported: fused_fc, banded_mix_fc)" % mode
        )

    n_layers = raw.get("num_hidden_layers", 1)
    if n_layers != 1:
        raise NotImplementedError(
            "this adapter targets the single-layer drafter (Llama3Eagle3Drafter has "
            "one `midlayer`); checkpoint has num_hidden_layers=%d" % n_layers
        )

    cfg = DrafterConfig(raw, n_streams)

    cls = type("HiViSLlama3Eagle3Drafter",
               (HiViSInterfaceMixin, mod.Llama3Eagle3Drafter), {})
    drafter = cls(cfg, load_emb=False, path=None, total_tokens=total_tokens,
                  depth=depth, top_k=top_k, threshold=threshold)
    drafter.aux_layer_bands = ()
    if bands:
        drafter.init_banded_mix(bands)

    state = _remap_train_keys(_load_state(checkpoint_dir))
    missing, unexpected = drafter.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith(("t2d", "d2t"))]
    # gist_norm is a dead weight in checkpoints trained with gist_conditioning
    # off; it has no consumer here.
    unexpected = [k for k in unexpected if "gist_norm" not in k]
    if missing:
        raise RuntimeError("drafter is missing %d checkpoint keys: %s"
                           % (len(missing), missing[:8]))

    # The real target layers whose hidden states must be captured and fed in.
    drafter.aux_hidden_states_layer_ids = list(raw["aux_hidden_states_layer_ids"])
    drafter.eval()
    for p in drafter.parameters():
        p.requires_grad_(False)
    if device is not None:
        drafter.to(device=device)
    drafter.to(dtype=dtype)
    return drafter, raw, unexpected


def _load_state(checkpoint_dir):
    from safetensors.torch import load_file

    state = {}
    shards = sorted(
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".safetensors")
    )
    for shard in shards:
        state.update(load_file(shard))
    if not state:
        bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if not os.path.isfile(bin_path):
            raise FileNotFoundError("no weights under %s" % checkpoint_dir)
        state = torch.load(bin_path, map_location="cpu")
    return state


def _remap_train_keys(state):
    """Training names the single draft block `layers.0.*`; inference calls it `midlayer.*`.

    Same module, same shapes -- only the attribute path differs between the
    training stack (an nn.ModuleList sized by num_hidden_layers) and the
    inference drafter (one fixed block).
    """
    out = {}
    for k, v in state.items():
        out["midlayer." + k[len("layers.0."):] if k.startswith("layers.0.") else k] = v
    return out
