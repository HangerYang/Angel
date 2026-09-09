"""Post-encoder visual-token selection for SmolVLM (Idefics3).

The three original arms (`full` / 320 / 64) compress at the *processor*: the
image is downsized before it is tiled, so pixels are destroyed before the vision
tower ever runs. Everything here instead keeps ``longest_edge`` at its default,
lets the encoder produce all 832 rows at full resolution, and then keeps a
subset of those rows. Same sequence length, sharper features.

Layout this relies on (measured on transformers 4.57.6, SmolVLM-256M, a
900x1200 input -- rerun ``~/tmp/vek/probe_layout.py`` after any upgrade):

    pixel_values          [1, 13, 3, 512, 512]
    <image> rows          832, in 13 contiguous runs of exactly 64
    get_image_features()  -> Tensor [13, 64, 576]
    row i                 <=> tile i // 64, patch i % 64
    the LAST 64 rows      are the <global-img> tile: the whole picture at 512px

That last fact is why ``global_only`` exists: at ``longest_edge=512`` the
processor emits exactly one tile, which is also the whole picture at 512px, so
selecting only the final 64 rows of a full-resolution pass must reproduce the
64-row pixel arm. It is the end-to-end test of the surgery below.
"""

import json
import os

import torch

GLOBAL_TILE_ROWS = 64
TILE_ROWS = 64
GLOBAL_IMG_TOKEN_ID = 49152  # <global-img>, verified against the 256M tokenizer


# --------------------------------------------------------------------------
# selectors: (feats [N, H], pool [M] long, budget, query [H] or None) -> [B] long
# indices are into the full [N] row space; each returns them sorted
# --------------------------------------------------------------------------

def _sel_stride(feats, pool, budget, query, gen):
    """Evenly spaced over the pool -- the structured baseline."""
    m = pool.numel()
    if budget >= m:
        return pool
    off = torch.arange(budget, device=pool.device)
    return pool[(2 * off + 1) * m // (2 * budget)]


def _sel_random(feats, pool, budget, query, gen):
    m = pool.numel()
    if budget >= m:
        return pool
    perm = torch.randperm(m, generator=gen, device="cpu")[:budget]
    return pool[perm.to(pool.device)].sort().values


def _sel_norm(feats, pool, budget, query, gen):
    """Top-B by L2 magnitude -- cheap saliency control."""
    if budget >= pool.numel():
        return pool
    scores = feats[pool].float().norm(dim=-1)
    keep = scores.topk(budget).indices
    return pool[keep].sort().values


def _sel_divprune(feats, pool, budget, query, gen):
    """Greedy max-min cosine diversity (farthest-point sampling).

    No query, no attention: pure coverage of the feature set. This is the
    control that isolates 'does diversity alone explain the win'.
    """
    if budget >= pool.numel():
        return pool
    x = torch.nn.functional.normalize(feats[pool].float(), dim=-1)
    # Official seed (vbdi/divprune, llava_arch.py:152 DivPrune): the first pick
    # is the row whose NEAREST neighbour is furthest away -- the most isolated
    # token -- taken as the 2nd-smallest cosine distance per column, since the
    # smallest is the token's distance to itself.
    dist = 1.0 - x @ x.T
    first = int(torch.topk(dist, 2, dim=0, largest=False).values[1].argmax())
    chosen = [first]
    mind = 1.0 - x @ x[first]
    for _ in range(budget - 1):
        mind[chosen] = -1.0
        nxt = int(mind.argmax())
        chosen.append(nxt)
        mind = torch.minimum(mind, 1.0 - x @ x[nxt])
    return pool[torch.tensor(chosen, device=pool.device)].sort().values


def _sel_cdpruner(feats, pool, budget, query, gen):
    """DPP MAP over a conditional-similarity kernel (CDPruner, arXiv 2506.10967).

    L_ij = q_i q_j <f_i, f_j> with f unit-norm and q an instruction-relevance
    score, so log-det trades feature diversity against query relevance in one
    objective. Greedy MAP is the O(B*N) Cholesky-update algorithm of Chen et
    al. 2018.

    Kernel construction, the greedy loop and the relevance normalisation follow
    the official implementation (Theia-4869/CDPruner, llava_arch.py:140-186).

    DEVIATION, and it cannot be removed on this model: the official relevance is
    a cosine between CLIP *image* embeds and CLIP *text* embeds, i.e. inside the
    vision tower's contrastive space. SmolVLM ships only the SigLIP vision half
    -- ``Idefics3VisionTransformer`` has embeddings/encoder/post_layernorm and no
    projection head, and ``model.text_model`` is the SmolLM decoder, not a paired
    text tower -- so that space does not exist here. ``query`` is instead the
    mean LM input embedding of the prompt's non-image tokens, matched against the
    post-connector features, which share the LM's 576-d space. Any CDPruner
    number measured here is therefore a port, not a reproduction.
    """
    if budget >= pool.numel():
        return pool
    x = torch.nn.functional.normalize(feats[pool].float(), dim=-1)
    n = x.shape[0]

    if query is None:
        q = torch.ones(n, device=x.device)
    else:
        rel = x @ torch.nn.functional.normalize(query.float(), dim=-1)
        # The official code negates before normalising:
        #     relevance = (-relevance).mean(dim=-1)
        #     relevance = (relevance - min + 1e-6) / (max - min)
        # so a token LESS similar to the text scores higher. Reproduced here,
        # with CDPRUNER_SIGN=pos to flip it back for the ablation.
        if os.environ.get("CDPRUNER_SIGN", "neg") == "neg":
            rel = -rel
        q = (rel - rel.min() + 1e-6) / (rel.max() - rel.min() + 1e-6)

    sim = x @ x.T
    d2 = q * q  # diagonal of L; <f_i,f_i> = 1
    cis = torch.zeros(budget, n, device=x.device)
    chosen = []
    j = int(d2.argmax())
    for i in range(budget):
        chosen.append(j)
        if i == budget - 1:
            break
        num = q * q[j] * sim[j]
        if i:
            num = num - cis[:i].T @ cis[:i, j]
        ei = num / torch.sqrt(d2[j].clamp_min(1e-10))
        cis[i] = ei
        d2 = (d2 - ei * ei).clamp_min(0)
        d2[torch.tensor(chosen, device=x.device)] = -1.0
        j = int(d2.argmax())
    return pool[torch.tensor(chosen, device=pool.device)].sort().values


@torch.no_grad()
def fastv_scores(model, embeds, img_pos, layer_k):
    """Attention the last prompt token pays to each visual row at layer K.

    FastV (arXiv 2403.06764): run the decoder, take the attention of the final
    input token over the visual positions in an early layer, and keep the top-K
    rows. Unlike every other selector here the signal lives inside the LM, not
    the vision encoder, so it costs one extra prefill -- run once, before the
    real generate, on the full uncompressed sequence.
    """
    lm = model.model.text_model
    attn_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
    # SDPA never materialises the attention matrix, so output_attentions comes
    # back None. Flip to eager for this one forward and put it back after.
    prev = getattr(lm.config, "_attn_implementation", None)
    try:
        lm.config._attn_implementation = "eager"
        for mod in lm.modules():
            if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
                mod.config._attn_implementation = "eager"
        out = lm(inputs_embeds=embeds, attention_mask=attn_mask,
                 output_attentions=True, use_cache=False)
    finally:
        if prev is not None:
            lm.config._attn_implementation = prev
            for mod in lm.modules():
                if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
                    mod.config._attn_implementation = prev
    if out.attentions is None:
        raise RuntimeError(
            "output_attentions returned None even under eager -- FastV has no "
            "signal to read on this transformers version")
    # layer_k is 0-based over the returned tuple; clamp for short stacks
    k = min(layer_k, len(out.attentions) - 1)
    a = out.attentions[k]                       # [1, heads, seq, seq]
    last = a[0, :, -1, :].mean(0)               # mean over heads, last query row
    return last[img_pos]                        # [n_img]


def _pool_groups(pool, budget, tile_of):
    """Split ``pool`` into ``budget`` contiguous groups that never span a tile.

    Rows are raster order inside a tile's 8x8 grid, so a contiguous group is a
    horizontal strip of that tile -- a real spatial neighbourhood, which is what
    makes averaging them meaningful rather than arbitrary.
    """
    tiles = sorted(set(tile_of.tolist()))
    per = {t: int((tile_of == t).sum()) for t in tiles}
    # hand out the budget proportionally, at least one group per tile
    quota = {t: max(1, budget * per[t] // max(sum(per.values()), 1)) for t in tiles}
    while sum(quota.values()) > budget and len(quota) > 1:
        quota[max(quota, key=lambda t: quota[t])] -= 1
    while sum(quota.values()) < budget:
        quota[min(quota, key=lambda t: quota[t])] += 1

    groups = []
    for t in tiles:
        rows = pool[tile_of == t]
        k = quota[t]
        n = rows.numel()
        for i in range(k):
            groups.append(rows[i * n // k:(i + 1) * n // k])
    return [g for g in groups if g.numel()]


def _sel_pool(feats, pool, budget, query, gen, tile_of=None):
    """Average-pool instead of discarding: the arm the whole sweep points at.

    Every other method here keeps some rows and throws the rest away, and the
    measured ordering says coverage beats sharpness -- a complete 64-row
    thumbnail beats any 64 sharp rows picked out of 832. Pooling is the way to
    keep coverage without keeping the token count: each surviving row is the
    mean of a contiguous strip, so nothing is discarded, only blurred.

    This is also the operation AngelSlim's VisRowCompressor was built to perform
    (``out = sum_i w_i * token_i`` with no value projection) before its routing
    saturated to one-hot and turned it into a row sampler. Here w is uniform by
    construction, so it cannot collapse.

    Returns (indices, values): each kept index is its group's first row, which
    keeps a real target position and the RoPE angle computed at it.
    """
    if tile_of is None:
        tile_of = torch.zeros_like(pool)
    groups = _pool_groups(pool, budget, tile_of)
    idx = torch.stack([g[0] for g in groups])
    vals = torch.stack([feats[g].mean(0) for g in groups])
    order = idx.argsort()
    return idx[order], vals[order]


def _sel_hybrid(feats, global_rows, crop_rows, budget, split, query, gen):
    """Thumbnail + selected crops, the two halves that keep failing separately.

    Every 64-token arm so far is one of two things: the <global-img> thumbnail,
    which covers the whole picture but blurs it, or rows chosen out of the
    high-resolution crops, which are sharp but cover 8-25% of the image and lose
    the layout. The thumbnail wins, but it cannot answer anything that needs
    detail; the crops can, but only where they happen to land.

    So spend part of the budget on the thumbnail -- average-pooled down, since
    pooling keeps coverage where dropping rows does not -- and the rest on
    DivPrune over the crop rows.
    """
    n_g = max(1, int(round(budget * split)))
    n_g = min(n_g, global_rows.numel())
    n_c = budget - n_g

    idx_g, vals_g = _sel_pool(feats, global_rows, n_g, query, gen,
                              torch.zeros_like(global_rows))
    if n_c <= 0 or crop_rows.numel() == 0:
        return idx_g, vals_g
    n_c = min(n_c, crop_rows.numel())
    idx_c = _sel_divprune(feats, crop_rows, n_c, query, gen)
    vals_c = feats[idx_c]                       # crops keep their own vectors

    idx = torch.cat([idx_g, idx_c])
    vals = torch.cat([vals_g, vals_c])
    order = idx.argsort()
    return idx[order], vals[order]


SELECTORS = {
    "hybrid": None,  # needs the global/crop split, handled in RowSelector
    "fastv": None,   # handled in generate_selected: needs the LM, not just feats
    "pool": _sel_pool,
    "stride": _sel_stride,
    "random": _sel_random,
    "norm": _sel_norm,
    "divprune": _sel_divprune,
    "cdpruner": _sel_cdpruner,
}


class RowSelector:
    """Parsed ``SMOLVLM_TOKEN_SELECT`` plus the run-level RNG and stats sink."""

    def __init__(self, spec, global_policy, seed, stats_path):
        self.spec = spec
        if spec == "global_only":
            self.method, self.budget = "global_only", GLOBAL_TILE_ROWS
        else:
            method, _, budget = spec.partition(":")
            if method.startswith("hybrid"):
                method = "hybrid"
            if method.startswith("fastv"):
                # fastv[@K]:budget -- K is the decoder layer, FastV's default is 2
                self.layer_k = int(method.split("@")[1]) if "@" in method else 2
                method = "fastv"
            if method not in SELECTORS:
                raise ValueError(
                    f"unknown SMOLVLM_TOKEN_SELECT={spec!r}; "
                    f"expected global_only or one of {sorted(SELECTORS)} with :budget"
                )
            if not budget.isdigit():
                raise ValueError(f"SMOLVLM_TOKEN_SELECT={spec!r} needs an integer budget")
            self.method, self.budget = method, int(budget)
        # global_only IS the global tile, so the exclude policy cannot apply to it
        self.global_policy = "include" if self.method == "global_only" else global_policy
        self.layer_k = getattr(self, "layer_k", 2)
        # fraction of the budget spent on the thumbnail; the rest goes to crops
        self.hybrid_split = float(os.environ.get("SMOLVLM_HYBRID_SPLIT", "0.5"))
        self.gen = torch.Generator().manual_seed(seed)
        self.stats_path = stats_path
        self.drop_empty_markers = (
            os.environ.get("SMOLVLM_DROP_EMPTY_MARKERS", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        self.n_calls = 0
        self.n_from_global = 0
        self.n_selected = 0
        self.n_markers_dropped = 0
        # every rank appends its own line; nothing calls this explicitly
        import atexit

        atexit.register(self.dump)

    def __call__(self, feats, query, global_mask=None):
        """feats [N, H] -> sorted long indices to keep.

        ``global_mask`` marks the rows belonging to a <global-img> tile. A
        prompt can carry several images (47 of MMMU_DEV_VAL's 1050 rows do), so
        each image contributes its own global tile and the trailing-64
        assumption would leave every earlier thumbnail inside the candidate
        pool -- letting a selector quietly rediscover the pixel baseline on
        exactly the rows the exclude policy exists to remove.
        """
        n = feats.shape[0]
        dev = feats.device
        all_rows = torch.arange(n, device=dev)
        if global_mask is None:
            global_mask = torch.zeros(n, dtype=torch.bool, device=dev)
            if n >= GLOBAL_TILE_ROWS:
                global_mask[n - GLOBAL_TILE_ROWS:] = True

        vals = None
        if self.method == "global_only":
            idx = all_rows[global_mask]
        else:
            # "only": operate INSIDE the <global-img> thumbnail. Pixel downscaling
            # stops at 64 tokens -- the processor always emits at least one tile --
            # so this is the only route below 64, and it compresses the
            # representation that actually wins rather than the crop rows.
            if self.global_policy == "only":
                pool = all_rows[global_mask]
            elif self.global_policy == "exclude":
                pool = all_rows[~global_mask]
            else:
                pool = all_rows
            if pool.numel() == 0:
                pool = all_rows
            if self.method == "hybrid":
                idx, vals = _sel_hybrid(
                    feats, all_rows[global_mask], all_rows[~global_mask],
                    self.budget, self.hybrid_split, query, self.gen)
            elif self.method == "pool":
                tile_of = pool // TILE_ROWS
                idx, vals = _sel_pool(feats, pool, self.budget, query, self.gen, tile_of)
            else:
                idx = SELECTORS[self.method](feats, pool, self.budget, query, self.gen)

        self.n_calls += 1
        self.n_selected += int(idx.numel())
        self.n_from_global += int(global_mask[idx].sum())
        return (idx, vals) if vals is not None else idx

    def summary(self):
        return {
            "spec": self.spec,
            "method": self.method,
            "budget": self.budget,
            "global_policy": self.global_policy,
            "calls": self.n_calls,
            "rows_selected": self.n_selected,
            "rows_from_global_tile": self.n_from_global,
            "drop_empty_markers": self.drop_empty_markers,
            "marker_tokens_dropped": self.n_markers_dropped,
            "frac_from_global_tile": (
                self.n_from_global / self.n_selected if self.n_selected else None
            ),
        }

    def dump(self):
        if not self.stats_path or not self.n_calls:
            return
        rec = dict(self.summary(), rank=os.environ.get("RANK", "0"))
        with open(self.stats_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def build_selector():
    """None unless SMOLVLM_TOKEN_SELECT is set, so the pixel arms are untouched."""
    spec = os.environ.get("SMOLVLM_TOKEN_SELECT")
    if not spec:
        return None
    return RowSelector(
        spec,
        os.environ.get("SMOLVLM_GLOBAL_POLICY", "exclude"),
        int(os.environ.get("SMOLVLM_SELECT_SEED", "0")),
        os.environ.get("SMOLVLM_SELECT_STATS"),
    )


@torch.no_grad()
def generate_selected(model, processor, inputs, selector, gen_kwargs):
    """Encode at full resolution, keep a subset of visual rows, generate.

    Replaces ``model.generate(**inputs)``. Returns the decoded string.

    Three things here are easy to get wrong and silent when wrong:

    * ``generate(inputs_embeds=...)`` returns ONLY the new tokens, so the
      caller's usual ``[:, input_ids.size(1):]`` slice would delete the entire
      answer and yield "". No slice is taken below.
    * nothing rebuilds ``attention_mask`` from ``input_ids`` once embeddings are
      passed in, so it is built explicitly.
    * deleting rows re-densifies positions. That is intended (no holes), but it
      does mean this path differs from the 832-row pass by more than the row
      count -- which is exactly what ``global_only`` is there to check.
    """
    image_token_id = int(model.config.image_token_id)
    input_ids = inputs["input_ids"]
    ids = input_ids[0]

    kw = {}
    if "pixel_attention_mask" in inputs:
        kw["pixel_attention_mask"] = inputs["pixel_attention_mask"]
    inner = model.model if hasattr(model, "model") else model
    feats = inner.get_image_features(inputs["pixel_values"], **kw)
    if not isinstance(feats, torch.Tensor):
        feats = feats[0]
    feats = feats.reshape(-1, feats.shape[-1])          # [n_tiles*64, H]

    img_pos = (ids == image_token_id).nonzero().flatten()
    if img_pos.numel() != feats.shape[0]:
        raise RuntimeError(
            f"{img_pos.numel()} <image> rows but {feats.shape[0]} encoded rows; "
            "the Idefics3 merge assumption is broken on this transformers version"
        )

    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(input_ids).clone()             # [1, L, H]
    embeds[0, img_pos] = feats.to(embeds.dtype)         # same order as inputs_merger

    # query = mean embedding of the prompt's non-image tokens
    text_mask = torch.ones_like(ids, dtype=torch.bool)
    text_mask[img_pos] = False
    query = embeds[0][text_mask].mean(0) if bool(text_mask.any()) else None

    # Which encoded rows are <global-img> thumbnails: each such tile's 64-row
    # run is the one that starts right after a <global-img> token.
    global_mask = torch.zeros(img_pos.numel(), dtype=torch.bool, device=ids.device)
    g_tok = (ids == GLOBAL_IMG_TOKEN_ID).nonzero().flatten()
    if g_tok.numel():
        pos_of = {int(v): k for k, v in enumerate(img_pos.tolist())}
        for gp in g_tok.tolist():
            start = pos_of.get(gp + 1)
            if start is not None:
                global_mask[start:start + TILE_ROWS] = True
    else:  # no marker (should not happen); fall back to the trailing tile
        if img_pos.numel() >= GLOBAL_TILE_ROWS:
            global_mask[-GLOBAL_TILE_ROWS:] = True

    if selector.method == "fastv":
        scores = fastv_scores(model, embeds, img_pos, selector.layer_k)
        pool = (~global_mask).nonzero().flatten() if selector.global_policy == "exclude" \
            else torch.arange(img_pos.numel(), device=ids.device)
        if pool.numel() == 0:
            pool = torch.arange(img_pos.numel(), device=ids.device)
        b = min(selector.budget, pool.numel())
        keep_idx = pool[scores[pool].topk(b).indices].sort().values
        selector.n_calls += 1
        selector.n_selected += int(keep_idx.numel())
        selector.n_from_global += int(global_mask[keep_idx].sum())
    else:
        keep_idx = selector(feats, query, global_mask)
    pooled_vals = None
    if isinstance(keep_idx, tuple):
        keep_idx, pooled_vals = keep_idx
        # each kept row now carries its group's mean, at its own real position
        embeds[0, img_pos[keep_idx]] = pooled_vals.to(embeds.dtype)
    keep = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
    keep[img_pos] = False
    keep[img_pos[keep_idx]] = True

    if selector.drop_empty_markers:
        # Each tile is announced by exactly two tokens immediately before its
        # 64-row run -- <fake_token_around_image> then <row_R_col_C> (or
        # <global-img>). A tile that kept no rows leaves its announcement
        # dangling in front of nothing; dropping the pair makes the token stream
        # of a single-tile selection match what the processor emits natively at
        # that tile count, which is what makes global_only a tight test of the
        # embedding surgery rather than a loose one.
        tiles = img_pos.view(-1, TILE_ROWS)
        kept_mask = torch.zeros(img_pos.numel(), dtype=torch.bool, device=ids.device)
        kept_mask[keep_idx] = True
        per_tile = kept_mask.view(-1, TILE_ROWS).any(dim=1)
        for t in range(tiles.shape[0]):
            if not bool(per_tile[t]):
                start = int(tiles[t, 0])
                if start >= 2:
                    keep[start - 2:start] = False
                    selector.n_markers_dropped += 2

    inputs_embeds = embeds[:, keep]
    attention_mask = torch.ones(
        inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
    )
    out = model.generate(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs
    )
    # inputs_embeds path: `out` is already only the new tokens -- do not slice
    return processor.batch_decode(out, skip_special_tokens=True)[0]
