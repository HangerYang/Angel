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
"""Oracle gist embedding computation for gist-conditioned Eagle3 training.

Runs at data-collate time (one datapoint per step, right before it's fed to
the model) instead of as an upfront ``ds.map()`` pass over the whole dataset.
This keeps the tokenization ``ds.map()`` pass byte-for-byte identical to a
non-gist config (same pinned cache file, so it hits an already-warm
``.map_cache`` instead of rebuilding it), and it means gist encoding work is
naturally sharded across ranks by the DataLoader's distributed sampler
instead of every rank redundantly encoding the full dataset up front.

No persistence: every text is encoded live and discarded. An in-memory or
on-disk cache here would grow unboundedly across an epoch with no bound on
staleness across epochs (this is training data, re-seen every epoch), so
this deliberately does not cache.
"""

import os
from typing import List

import torch

from angelslim.utils import rank0_print


def resolve_gist_encoder_device(configured_device: str) -> str:
    """Spread each torchrun rank's gist encoder across its own GPU.

    Under torchrun (LOCAL_RANK set), every rank sees the same
    CUDA_VISIBLE_DEVICES list (not narrowed per-rank by this launch setup),
    so a single fixed device like "cuda:0" makes every rank's encoder pile
    onto the SAME physical GPU while the others sit idle. Default to
    cuda:{LOCAL_RANK % num_gpus} instead. Falls back to the configured value
    outside torchrun or if CUDA isn't available.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or not torch.cuda.is_available():
        return configured_device
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 0:
        return configured_device
    return f"cuda:{int(local_rank) % n_gpus}"


class GistEmbeddingEncoder:
    """Computes oracle gist embeddings for one example at a time.

    Two modes:

    ``remaining`` (default)
        Re-encode the suffix from token t to the end of the span, refreshed
        every ``refresh_every`` tokens. The vector shrinks as the span is
        consumed, so it carries a "what is still to come" countdown -- at the
        cost of O(L^2 / refresh_every) encoder tokens for a span of length L.

    ``whole``
        Encode the ground-truth response once and hold it constant over every
        supervised position. O(L) instead of O(L^2) -- ~100x less encoder work
        on this dataset -- but the vector no longer localizes: at token t it
        still describes text already emitted. Since the median assistant span
        here is ~29 tokens, the two modes coincide for most examples and only
        diverge on the long tail.

    Lazily loads its SentenceTransformer encoder on first use (collate_fn
    runs in the main training process here -- dataloader_num_workers=0 is
    the default and is what every online training script uses -- so there is
    no fork-after-CUDA-init hazard to guard against).
    """

    def __init__(
        self,
        tokenizer,
        model_name_or_path: str,
        refresh_every: int,
        device: str,
        batch_size: int,
        embedding_dim: int = 0,
        mode: str = "remaining",
    ):
        if mode not in ("remaining", "whole"):
            raise ValueError(
                f"gist_mode must be 'remaining' or 'whole', got {mode!r}"
            )
        self.tokenizer = tokenizer
        self.mode = mode
        self.model_name_or_path = model_name_or_path
        self.refresh_every = max(1, int(refresh_every))
        self.device = resolve_gist_encoder_device(device)
        self.batch_size = max(1, int(batch_size))
        self.embedding_dim = int(embedding_dim or 0)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            rank0_print(
                f"Loading oracle gist encoder {self.model_name_or_path} on {self.device}"
            )
            try:
                self._encoder = SentenceTransformer(
                    self.model_name_or_path,
                    device=self.device,
                    trust_remote_code=True,
                )
            except Exception as e:
                if e.__class__.__name__ == "GatedRepoError" or "GatedRepoError" in repr(e):
                    raise RuntimeError(
                        "Cannot load oracle gist encoder "
                        f"{self.model_name_or_path!r}: the Hugging Face repo is gated "
                        "for the current credentials. Accept the model license and run "
                        "`huggingface-cli login` for this user, or set HF_TOKEN before "
                        "starting oracle-gist training."
                    ) from e
                raise
            dim = int(self._encoder.get_sentence_embedding_dimension())
            if self.embedding_dim and self.embedding_dim != dim:
                raise ValueError(
                    f"gist_embedding_dim={self.embedding_dim} but encoder returns {dim}"
                )
            self.embedding_dim = dim
        return self._encoder

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        encoder = self._get_encoder()
        return (
            encoder.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            .detach()
            .cpu()
            .float()
        )

    def _spans(self, mask: torch.Tensor) -> List[tuple]:
        """Contiguous [start, end) runs of loss_mask == 1."""
        spans = []
        i = 0
        while i < mask.numel():
            if int(mask[i].item()) == 0:
                i += 1
                continue
            start = i
            while i < mask.numel() and int(mask[i].item()) == 1:
                i += 1
            spans.append((start, i))
        return spans

    def build_for_item(
        self, input_ids: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        """Returns a [1, seq_len, embedding_dim] tensor for one example."""
        ids = input_ids.detach().cpu().long().view(-1)
        mask = loss_mask.detach().cpu().long().view(-1)
        if self.embedding_dim <= 0:
            self._get_encoder()
        gist = torch.zeros((ids.numel(), self.embedding_dim), dtype=torch.float32)
        spans = self._spans(mask)
        if not spans:
            return gist.unsqueeze(0)

        if self.mode == "whole":
            # One vector for the whole ground-truth response, held constant over
            # every supervised position. O(span) encoder work per example instead
            # of the O(span^2/refresh_every) the "remaining" mode costs, at the
            # price of the countdown signal: at token t this still describes text
            # already emitted, not just what is left.
            span_ids: List[int] = []
            for start, end in spans:
                span_ids.extend(ids[start:end].tolist())
            text = self.tokenizer.decode(span_ids, skip_special_tokens=True).strip()
            vec = self._encode_texts([text])[0]
            for start, end in spans:
                gist[start:end] = vec
            return gist.unsqueeze(0)

        for start, end in spans:
            refresh_offsets = list(range(0, end - start, self.refresh_every))
            suffix_texts = []
            for offset in refresh_offsets:
                suffix_ids = ids[start + offset : end].tolist()
                text = self.tokenizer.decode(suffix_ids, skip_special_tokens=True).strip()
                suffix_texts.append(text)
            suffix_vecs = self._encode_texts(suffix_texts)
            for offset, vec in zip(refresh_offsets, suffix_vecs):
                left = start + offset
                right = min(end, left + self.refresh_every)
                gist[left:right] = vec
        return gist.unsqueeze(0)
