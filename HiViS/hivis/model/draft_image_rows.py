"""Give the drafter fewer image rows than the target saw, the way it was trained.

The target ALWAYS runs its own full prefill over the full-resolution image;
nothing here touches that. What changes is only what the drafter is handed.

Two mechanisms, matching the two ways these checkpoints were trained:

``pool``   Average contiguous runs of image rows in the hidden states the
           target's full prefill already produced. No second forward.
``short``  A SECOND target forward over ``<tokens before the image> + <the same
           image at a lower resolution>``, truncated after the deepest aux
           layer, whose image rows replace the full forward's. Text rows still
           come from the full forward.

Both are imported from the training code rather than reimplemented here
(``reduce_vlm_image_rows`` and ``short_image``), because a drafter evaluated
against a slightly different reduction than it was trained on measures the
mismatch, not the reduction. That is also why ``short`` re-runs the target at
eval time instead of pooling: in bf16 the same image rows computed inside a
long sequence and inside the short one differ by ~2% RMS (attention kernels
tile their reductions differently at different sequence lengths), so the rows
have to come from the same kind of forward in both places.

POSITIONS. Both mechanisms keep ABSOLUTE position ids: the text rows after the
image were computed by the full forward and carry the positions they were
computed at, so the reduced image rows must keep theirs too. The resulting
sequence is shorter than its highest position id -- there is a gap where the
image was -- which is exactly what training fed the drafter, and why the
drafter cannot be left to infer positions from sequence length here.
"""

import copy
import importlib
import time
import os
import sys
import types

import torch

_TRAIN_PKG = "angelslim.compressor.speculative.train"


_TRAIN_ROOT_ENV = "ANGELSLIM_TRAIN_ROOT"


def _load(module, angelslim_root=None):
    """Import one AngelSlim training module without running angelslim/__init__.

    Same stand-in-package trick as angelslim_drafter.load_angelslim_draft_module,
    and for the same reason: angelslim/__init__.py evaluates py3.10-only
    annotations at import time and HiViS runs on py3.9. The leaf modules
    themselves are py3.9-clean.

    Deliberately a SEPARATE root from the one the drafter is loaded from. The
    reduction code lives on the branch the checkpoints were trained on, which
    need not be the checkout HiViS is vendored in; keeping the two independent
    means adding it here cannot quietly change which drafter implementation the
    already-measured arms ran against.
    """
    from .angelslim_drafter import _default_root, _ANGELSLIM_ROOT_ENV

    root = (angelslim_root or os.environ.get(_TRAIN_ROOT_ENV)
            or os.environ.get(_ANGELSLIM_ROOT_ENV) or _default_root())
    leaf = os.path.join(root, *module.split(".")) + ".py"
    if not os.path.isfile(leaf):
        raise RuntimeError(
            "%s not found under %r. The draft-side image reduction lives on the "
            "branch these checkpoints were trained on; point %s at that checkout "
            "(or pass --draft_image_root)." % (module, root, _TRAIN_ROOT_ENV)
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    parts = module.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        if name in sys.modules:
            continue
        stand_in = types.ModuleType(name)
        stand_in.__path__ = [os.path.join(root, *parts[:i])]
        sys.modules[name] = stand_in
    return importlib.import_module(module)


# --------------------------------------------------------------------------
# pool
# --------------------------------------------------------------------------

def pool_image_rows(input_ids, hidden_states, image_token_id, factor,
                    keep_positions=True, how="pool", angelslim_root=None):
    """Reduce the drafter's image rows by `factor`, averaging contiguous runs.

    Delegates to the collator's own reduce_vlm_image_rows, so this is the same
    grouping, the same representative row, and the same position rule the
    checkpoint was trained with.

    The tensors stay on the GPU. They used to make a round trip to host memory
    so that the collator -- written for a dataloader worker -- could do the
    averaging in CPU float32; that cost ~8 ms per prompt against 0.04 ms for the
    same arithmetic on device, and at 48 OMP threads it had a tail to 130 ms.
    reduce_vlm_image_rows now allocates on its input's device, so training (CPU
    tensors) and this call (CUDA tensors) still run the one code path.

    Returns (input_ids, hidden_states, position_ids), all batch-1.
    """
    data_utils = _load(_TRAIN_PKG + ".data.data_utils", angelslim_root)
    item = data_utils.reduce_vlm_image_rows(
        {"input_ids": input_ids, "hidden_states": hidden_states},
        image_token_id, factor, how=how, keep_positions=keep_positions,
    )
    return (
        item["input_ids"].to(input_ids.device),
        item["hidden_states"].to(hidden_states.device, hidden_states.dtype),
        item["position_ids"].to(input_ids.device),
    )


# --------------------------------------------------------------------------
# short: the target's second forward
# --------------------------------------------------------------------------

def _load_qsampler(ckpt, angelslim_root, device):
    """The frozen visual compressor and the budget it serves.

    Loaded by file path rather than by importing the package: this runs in the
    HiViS environment, whose interpreter predates some annotations the sibling
    modules under angelslim.compressor use at import time.
    """
    import importlib.util
    import json

    src = os.path.join(angelslim_root, "angelslim", "compressor", "vistoken",
                       "qsampler.py")
    spec = importlib.util.spec_from_file_location("_qsampler_standalone", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(ckpt)),
                            "qsampler_config.json")
    saved = json.load(open(cfg_path))
    fields = ("hidden_size", "num_queries", "num_blocks", "num_heads",
              "mlp_hidden", "tile_tokens", "grid")
    model = mod.QSampler(mod.QSamplerConfig(**{k: saved[k] for k in fields
                                               if k in saved}))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval().to(device=device, dtype=torch.float32)
    for prm in model.parameters():
        prm.requires_grad_(False)
    n = int(os.environ.get("VISTOKEN_QSAMPLER_N", saved["num_queries"]))
    if n > saved["num_queries"]:
        raise ValueError(
            "checkpoint serves at most %d tokens/tile, VISTOKEN_QSAMPLER_N asks "
            "for %d" % (saved["num_queries"], n))
    return model, n


_TIME_REDUCER = os.environ.get("HIVIS_TIME_REDUCER", "0") not in ("0", "", "false")
_TIMING = {"short_calls": 0, "short_s": 0.0, "outer_calls": 0, "outer_s": 0.0}

if _TIME_REDUCER:
    import atexit

    @atexit.register
    def _report_timing():
        n = _TIMING["short_calls"]
        if n:
            print(f"[reducer timing] short: {n} calls, {_TIMING['short_s']:.2f}s "
                  f"total, {1000 * _TIMING['short_s'] / n:.1f} ms/call", flush=True)
        m = _TIMING["outer_calls"]
        if m:
            print(f"[reducer timing] outer: {m} calls, {_TIMING['outer_s']:.2f}s "
                  f"total, {1000 * _TIMING['outer_s'] / m:.1f} ms/call", flush=True)


class ShortImageRows(object):
    """The second target forward, held open across a benchmark run.

    Built once (the forward hooks on the target's aux layers are permanent), then
    called per prompt.
    """

    def __init__(self, target_path, processor, aux_layer_ids, image_token_id,
                 longest_edge, device=None, dtype=None, angelslim_root=None):
        from transformers import AutoModelForImageTextToText

        self.si = _load(_TRAIN_PKG + ".short_image", angelslim_root)
        # A SEPARATE, stock copy of the target -- not the one EaModel is driving.
        # HiViS swaps that one's text_model for its own KV-cache variant, whose
        # forward does not take the arguments transformers' Idefics3Model passes
        # it. Training also ran this forward on a freshly loaded stock model
        # (offline_eagle3_trainer._short_forward), so this keeps the two
        # identical rather than merely working. ~0.5 GB for SmolVLM-256M.
        self.target = AutoModelForImageTextToText.from_pretrained(
            target_path, torch_dtype=dtype or torch.bfloat16).to(device).eval()
        for prm in self.target.parameters():
            prm.requires_grad_(False)
        # A PRIVATE copy of the processor. short_image_prefill sets
        # image_processor.size to the short edge and does not put it back, so
        # sharing the target's processor silently re-processes every prompt
        # after the first at the reduced resolution -- the target then stops
        # seeing the full image, which is the one thing this experiment must
        # never do. The copy makes that impossible rather than merely unlikely.
        self.processor = copy.deepcopy(processor)
        self.image_token_id = int(image_token_id)
        self.longest_edge = int(longest_edge)
        self.special_ids = self.si.image_special_ids(processor.tokenizer, image_token_id)
        self.layout = self.si.layout_ids(processor.tokenizer, self.special_ids)
        size = getattr(processor.image_processor, "size", None)
        self.base_size = dict(size) if isinstance(size, dict) else size
        # The trained visual compressor, if this arm has one. Training set it
        # from the same two env vars, so the two sides cannot drift: a drafter
        # trained on 16 compressed rows must be evaluated on 16 compressed rows.
        self.num_queries = None
        compressor = None
        ckpt = os.environ.get("VISTOKEN_QSAMPLER_CKPT")
        if ckpt:
            compressor, self.num_queries = _load_qsampler(
                ckpt, angelslim_root, device)
            print("draft image rows: qsampler %s -> %d rows/tile"
                  % (ckpt, self.num_queries))
        self.forward = self.si.ShortImageForward(
            self.target, aux_layer_ids, compressor=compressor,
            num_queries=self.num_queries, image_token_id=self.image_token_id)

    def __call__(self, input_ids, hidden_states, image):
        """Splice the short forward's image rows into the full forward's rows.

        With HIVIS_TIME_REDUCER=1 the wall clock of this call is accumulated and
        printed at exit, which is the only way to say how much of an arm's
        slowdown is the second target forward rather than the drafting itself.

        Returns (input_ids, hidden_states, position_ids) or None when this
        prompt has no usable image region, in which case the caller should leave
        the full rows alone.
        """
        t_enter = time.perf_counter() if _TIME_REDUCER else None
        i0, span_full, _ = self.si.image_span(
            input_ids, self.image_token_id, self.special_ids, self.layout, strict=False
        )
        if i0 is None:
            return None
        prep = self.si.short_image_prefill(
            # The processor tokenises on CPU and short_image_prefill compares the
            # re-tokenised prefix against this one, so hand it CPU ids.
            self.processor, self.base_size, image, input_ids[:, :i0].cpu(), span_full,
            self.longest_edge, self.image_token_id, self.special_ids, self.layout,
            num_queries=self.num_queries,
        )
        if prep is None:
            return None

        device = hidden_states.device
        span_short = int(prep["short_span"])
        aux = self.forward(
            input_ids=prep["short_input_ids"].to(device),
            pixel_values=prep["short_pixel_values"].to(device, self.target.dtype),
            position_ids=prep["short_position_ids"].to(device),
        )[0, i0: i0 + span_short]

        short_ids = prep["short_input_ids"][0, i0: i0 + span_short].to(device)
        full_pos = torch.arange(input_ids.shape[1], device=device)
        new_ids = self.si.splice_image_rows(
            input_ids[0], i0, span_full, None, short_ids)[None]
        new_hidden = self.si.splice_image_rows(
            hidden_states[0], i0, span_full, aux.to(hidden_states.dtype))[None]
        new_pos = self.si.splice_image_rows(
            full_pos, i0, span_full,
            prep["short_position_ids"][0, i0:].to(device))[None]
        if _TIME_REDUCER:
            torch.cuda.synchronize()
            _TIMING["short_calls"] += 1
            _TIMING["short_s"] += time.perf_counter() - t_enter
        return new_ids, new_hidden, new_pos


# --------------------------------------------------------------------------
# what initialize_tree calls
# --------------------------------------------------------------------------

class DraftImageReducer(object):
    """Callable held on the EaModel as `draft_image_reducer`.

    Returns (draft_input_ids, draft_hidden_states, absolute_position_ids), or
    None to leave the full rows in place (a prompt with no image, or one whose
    image region cannot be located).

    For ``short`` the caller must set `.image` to the prompt's PIL image before
    each generation; the second forward needs the picture, not the pixel values
    the target was given, because it re-processes it at a lower resolution.
    """

    def __init__(self, mode, image_token_id, factor=4, longest_edge=None,
                 keep_positions=True, target_path=None, processor=None,
                 aux_layer_ids=None, device=None, dtype=None, angelslim_root=None,
                 control=None, control_seed=0):
        self.mode = mode
        # A control replaces the CONTENT of the drafter's image rows while
        # leaving their count, their token ids and their absolute positions
        # exactly as the arm produced them. If tau survives one of these, the
        # drafter was not reading the image in the first place.
        if control not in (None, "zero", "shuffle", "random", "wrong", "drop"):
            raise ValueError(
                "control must be zero, shuffle, random, wrong or drop")
        self.control = control
        self._gen = torch.Generator().manual_seed(int(control_seed))
        self._bank = []
        self._swapped = 0
        self._control_calls = 0
        self.image_token_id = int(image_token_id)
        self.factor = int(factor)
        self.keep_positions = bool(keep_positions)
        self.angelslim_root = angelslim_root
        self.image = None
        self.short = None
        self._reported = False
        if mode == "short":
            if longest_edge is None:
                raise ValueError("short mode needs longest_edge")
            self.short = ShortImageRows(
                target_path, processor, aux_layer_ids, image_token_id, longest_edge,
                device=device, dtype=dtype, angelslim_root=angelslim_root,
            )
        elif mode not in ("pool", "subset"):
            raise ValueError("mode must be pool, subset or short, got %r" % mode)

    def _apply_control(self, input_ids, hidden_states, out):
        """Corrupt only the drafter's image rows; keep ids, count and positions."""
        new_ids, new_hidden, new_pos = out
        img = (new_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        if img.numel() == 0:
            return out
        self._control_calls += 1
        if self.control == "drop":
            # Not a corruption: the image rows are REMOVED. The drafter is left
            # with the text rows alone, at their original absolute positions, so
            # every arm gets a byte-identical sequence and any remaining
            # difference is the weights.
            keep = new_ids[0] != self.image_token_id
            return new_ids[:, keep], new_hidden[:, keep], new_pos[:, keep]
        rows = new_hidden[0, img]
        if self.control == "zero":
            rows = torch.zeros_like(rows)
        elif self.control == "shuffle":
            perm = torch.randperm(rows.shape[0], generator=self._gen)
            rows = rows[perm.to(rows.device)]
        elif self.control == "random":
            # Same count, drawn at random from the rows the FULL forward made.
            full_img = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
            k = min(rows.shape[0], full_img.numel())
            pick = torch.randperm(full_img.numel(), generator=self._gen)[:k].sort().values
            drawn = hidden_states[0, full_img[pick.to(full_img.device)]]
            if k < rows.shape[0]:
                drawn = torch.cat([drawn, drawn[-1:].expand(rows.shape[0] - k, -1)], 0)
            rows = drawn.to(rows.dtype)
        elif self.control == "wrong":
            # Rows this same arm computed for a DIFFERENT prompt's image.
            same = [t for t in self._bank if t.shape == rows.shape]
            if same:
                j = int(torch.randint(len(same), (1,), generator=self._gen))
                rows = same[j].to(rows.device, rows.dtype)
                self._swapped += 1
            self._bank.append(new_hidden[0, img].detach().to("cpu").clone())
            if len(self._bank) > 32:
                self._bank.pop(0)
        new_hidden = new_hidden.clone()
        new_hidden[0, img] = rows.to(new_hidden.dtype)
        return new_ids, new_hidden, new_pos

    def control_report(self):
        if self.control is None:
            return ""
        extra = ""
        if self.control == "wrong":
            extra = "  (%d/%d prompts got another prompt's rows)" % (
                self._swapped, self._control_calls)
        return "control=%s applied to %d prompts%s" % (
            self.control, self._control_calls, extra)

    def __call__(self, input_ids, hidden_states):
        if _TIME_REDUCER:
            t0 = time.perf_counter()
            try:
                return self._call(input_ids, hidden_states)
            finally:
                torch.cuda.synchronize()
                _TIMING["outer_calls"] += 1
                _TIMING["outer_s"] += time.perf_counter() - t0
        return self._call(input_ids, hidden_states)

    def _call(self, input_ids, hidden_states):
        if not bool((input_ids[0] == self.image_token_id).any()):
            return None
        if self.mode == "short":
            if self.image is None:
                raise RuntimeError(
                    "short mode: .image was not set for this prompt")
            out = self.short(input_ids, hidden_states, self.image)
        else:
            out = pool_image_rows(
                input_ids, hidden_states, self.image_token_id, self.factor,
                keep_positions=self.keep_positions, how=self.mode,
                angelslim_root=self.angelslim_root,
            )
        if out is not None and self.control is not None:
            out = self._apply_control(input_ids, hidden_states, out)
        # Say once what actually happened. A reduction that silently did nothing
        # (image token id wrong, region not found) still produces a plausible
        # tau -- of the unreduced drafter -- and nothing else would show it.
        if out is not None and not self._reported:
            self._reported = True
            n_img = int((input_ids[0] == self.image_token_id).sum())
            kept = int((out[0][0] == self.image_token_id).sum())
            print("  draft rows: %d -> %d (image %d -> %d), positions %d..%d"
                  % (input_ids.shape[1], out[0].shape[1], n_img, kept,
                     int(out[2][0, 0]), int(out[2][0, -1])))
            if self.control is not None:
                print("  draft image rows CONTROL: %s" % self.control)
        return out
