"""The target's second forward: the image, in fewer rows.

The drafter does not need the image at the target's resolution. This module
computes a SHORT version of the image's hidden states -- a second target forward
over ``<tokens before the image> + <the image at a lower resolution>`` -- so the
drafter can be fed

    text rows   from the target's FULL forward (the image is entirely in context)
    image rows  from this shorter forward

while the target itself, and everything the loss is computed against, still sees
the full image.

Two properties make the short forward cheap, and both are consequences of causal
attention rather than approximations:

* Tokens AFTER the image cannot affect the image's rows, at any layer, so they
  are simply not run. Tokens BEFORE the image can, and are included verbatim --
  which is what keeps this correct for chat templates that put text first.
* Nothing past the deepest auxiliary layer is run, and the LM head never runs,
  because only the auxiliary hidden states are wanted.

One caveat worth knowing: in bf16 the same rows computed inside a long sequence
and inside this short one differ by ~2% RMS, because the attention kernel tiles
its reductions differently at different sequence lengths and float addition is
not associative. In fp32 the two are bitwise identical. This matters only if the
image rows are taken from the long forward somewhere and from the short forward
somewhere else -- so both training and inference use the short forward.
"""
import copy

import torch

__all__ = [
    "image_special_ids",
    "layout_ids",
    "image_span",
    "short_image_prefill",
    "ShortImageForward",
    "splice_image_rows",
]


def image_special_ids(tokenizer, image_token_id):
    """Token ids the processor emits as part of the image itself.

    Idefics3/SmolVLM does not lay an image down as one run of <image>: each 512px
    tile is bracketed by <fake_token_around_image> and a <row_i_col_j> marker, so
    an 832-token image occupies an 859-token span.
    """
    ids = {int(image_token_id)}
    for tok, tid in tokenizer.get_vocab().items():
        if tok.startswith("<row_") or tok in ("<fake_token_around_image>", "<global-img>"):
            ids.add(int(tid))
    return ids


def layout_ids(tokenizer, special_ids):
    """`special_ids` plus whitespace, which separates rows of tiles."""
    ws = set()
    for tid in range(len(tokenizer)):
        try:
            if tokenizer.decode([tid]).strip() == "":
                ws.add(tid)
        except Exception:
            pass
    return set(special_ids) | ws


_ID_TENSOR_CACHE = {}


def _id_tensor(id_set, like):
    """`id_set` as a 1-D tensor on `like`'s device, built once per (set, device).

    Membership was tested one token at a time, which is a device sync per token
    when the row being scanned is the drafter's live CUDA sequence rather than a
    stored .ckpt -- 11 ms on an 832-row image. torch.isin against this costs one
    kernel instead. The set is kept alive in the cache so its id() cannot be
    recycled under the key.
    """
    key = (id(id_set), like.device)
    held = _ID_TENSOR_CACHE.get(key)
    if held is None or held[0] is not id_set:
        held = (id_set, torch.as_tensor(sorted(id_set), dtype=torch.long,
                                        device=like.device))
        _ID_TENSOR_CACHE[key] = held
    return held[1]


def image_span(ids, image_token_id, special_ids, layout, strict=True):
    """(start, span_len, n_image_tokens) of the image region, or (None, 0, 0).

    The region runs from the first <image> to the last, plus the markers that
    bracket them; everything in between belongs to the image's tiling. Real text
    in between means several images with a question between them, which this does
    not handle -- `strict` makes that a skip rather than an error.
    """
    row = ids[0] if ids.dim() == 2 else ids
    hit = (row == image_token_id).nonzero().flatten()
    if hit.numel() == 0:
        return None, 0, 0
    lo, hi = int(hit[0]), int(hit[-1])
    # Widening over the bracketing markers and checking the region for strays
    # are both membership tests. Done per token they cost one sync each; done
    # as two isin kernels the whole function takes a fixed handful of syncs.
    # The widened edge is the first position outside the run that is NOT a
    # marker, which is exactly what walking outward one token at a time found.
    outside = (~torch.isin(row, _id_tensor(special_ids, row))).nonzero().flatten()
    left, right = outside[outside < lo], outside[outside > hi]
    lo = int(left[-1]) + 1 if left.numel() else 0
    hi = int(right[0]) - 1 if right.numel() else int(row.numel()) - 1
    region = row[lo:hi + 1]
    stray_mask = ~torch.isin(region, _id_tensor(layout, row))
    if bool(stray_mask.any()):
        if strict:
            return None, 0, 0
        stray = [int(t) for t in region[stray_mask][:5]]
        raise ValueError(f"non-image tokens inside the image region: {stray[:5]}")
    return lo, hi - lo + 1, int(hit.numel())


def set_image_resolution(processor, base_size, longest_edge):
    """Cap the resolution, which is what caps the number of image tokens."""
    processor.image_processor.size = (copy.deepcopy(base_size) if longest_edge is None
                                      else {"longest_edge": int(longest_edge)})


def shrink_image_run(ids, image_token_id, num_queries):
    """Rewrite the run of <image> ids to ``num_queries``, markers untouched.

    A compressor emits fewer rows than the tile has, so the sequence handed to
    the second forward has to declare that count -- the markers around the tile
    stay exactly as the processor wrote them.
    """
    row = ids[0]
    hit = (row == image_token_id).nonzero().flatten()
    lo, hi = int(hit[0]), int(hit[-1])
    keep = torch.full((num_queries,), int(image_token_id), dtype=row.dtype)
    return torch.cat([row[:lo], keep, row[hi + 1:]]).unsqueeze(0)


def short_image_prefill(processor, base_size, image, prefix_ids, span_full, longest_edge,
                        image_token_id, special_ids, layout, num_queries=None):
    """CPU half of the short forward: build its inputs.

    Runs in a dataloader worker. Returns None when the sample has no usable image
    region. `position_ids` are ABSOLUTE: the short image rows are spread across
    the span the full image occupied, because the text rows that follow come from
    the full forward and keep the positions they were computed at.
    """
    set_image_resolution(processor, base_size, longest_edge)
    prefix_text = processor.tokenizer.decode(prefix_ids[0], skip_special_tokens=False)
    enc = processor(text=prefix_text + "<image>", images=[image], return_tensors="pt")
    ids = enc["input_ids"]
    i0, span_short, _ = image_span(ids, image_token_id, special_ids, layout, strict=False)
    if i0 is None:
        return None
    if i0 != prefix_ids.shape[1] or not torch.equal(ids[0, :i0], prefix_ids[0].to(ids.dtype)):
        raise ValueError(
            f"the prefix re-tokenised to {i0} tokens that do not match the stored "
            f"{prefix_ids.shape[1]}; the chat template used here differs from the one "
            f"the hidden states were generated with")
    if num_queries:
        # Shrink AFTER locating the span: the span is found on what the processor
        # actually emitted, then the run inside it is rewritten.
        ids = shrink_image_run(ids, image_token_id, num_queries)
        i0, span_short, _ = image_span(ids, image_token_id, special_ids, layout,
                                       strict=False)
        if i0 is None:
            return None
    img_pos = torch.linspace(0, span_full - 1, span_short).round().long() + i0
    return {
        "short_input_ids": ids,                                        # 1, i0 + span_short
        # generate_hidden_for_draft_model.py hands the target pixel_values alone,
        # so the full forward's image rows were computed without this mask. It is
        # all ones for a single image at any aspect ratio, so dropping it changes
        # nothing -- but keeping the two calls identical costs nothing either.
        "short_pixel_values": enc["pixel_values"],
        "short_position_ids": torch.cat([torch.arange(i0), img_pos]).unsqueeze(0),
        "short_img_start": torch.tensor([i0]),
        "short_span": torch.tensor([span_short]),
    }


class ShortImageForward:
    """GPU half: the target, truncated after the deepest auxiliary layer."""

    def __init__(self, model, aux_layer_ids, text_layers=None,
                 compressor=None, num_queries=None, image_token_id=None):
        self.model = model
        # When a compressor is given, the connector's output is compressed before
        # it reaches the text stack, so the text rows are computed in the presence
        # of the compressed image rather than the full one -- which is the whole
        # point: the drafter must be trained on the context it will see.
        self.compressor = compressor
        self.num_queries = num_queries
        self.image_token_id = image_token_id
        self.aux_layer_ids = list(aux_layer_ids)
        self.layers = text_layers if text_layers is not None else model.model.text_model.layers
        self.n_layers_full = len(self.layers)
        # +1: aux id L means "the output of decoder layer L" (the embed_offset in
        # _extract_auxiliary_hidden_states), so layer L itself still has to run.
        self.n_layers_keep = max(self.aux_layer_ids) + 1
        self._captured = {}
        self._handles = [self.layers[L].register_forward_hook(self._grab(L))
                         for L in self.aux_layer_ids]

    def _grab(self, layer_id):
        def hook(module, args, output):
            self._captured[layer_id] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    @torch.no_grad()
    def __call__(self, input_ids, pixel_values, position_ids, attention_mask=None):
        """Returns the aux hidden states [B, S, 3D] for the whole short sequence."""
        kw = dict(input_ids=input_ids, pixel_values=pixel_values, position_ids=position_ids,
                  attention_mask=attention_mask if attention_mask is not None
                  else torch.ones_like(input_ids))
        self._captured.clear()
        holder = self.model.model.text_model
        holder.layers = self.layers[: self.n_layers_keep]
        try:
            if self.compressor is None:
                self.model.model(**kw)      # .model, not the model: skips the LM head
            else:
                inner = self.model.model
                feats = inner.get_image_features(pixel_values=pixel_values)
                feats = getattr(feats, "pooler_output", feats)   # [tiles, 64, H]
                q = self.compressor(feats.float(), self.num_queries).to(feats.dtype)
                embeds = inner.get_input_embeddings()(input_ids)
                mask = (input_ids == self.image_token_id).unsqueeze(-1)
                embeds = embeds.masked_scatter(mask, q.to(embeds.dtype))
                holder(inputs_embeds=embeds, position_ids=kw["position_ids"],
                       attention_mask=kw["attention_mask"], use_cache=False)
        finally:
            holder.layers = self.layers
        if len(self._captured) != len(self.aux_layer_ids):
            raise RuntimeError(f"captured {sorted(self._captured)}, wanted {self.aux_layer_ids}")
        return torch.cat([self._captured[L] for L in self.aux_layer_ids], dim=-1)


def splice_image_rows(rows, i0, span_full, short_rows, short_ids=None):
    """Replace rows[i0 : i0+span_full] with `short_rows` (unbatched, dim 0 = seq).

    `rows` may be any per-position tensor. Returns the spliced tensor.
    """
    parts = [rows[:i0], short_rows if short_ids is None else short_ids, rows[i0 + span_full:]]
    return torch.cat(parts, dim=0)
