"""Dataset + collator for Q-Sampler distillation on SmolVLM.

The corpus is a ~15GB jsonl whose image rows embed pixels as base64 data URIs,
so it is never loaded into memory: a one-pass byte-offset index is built (and
cached next to the file) and rows are read by ``seek``. That keeps shuffling and
multi-epoch training cheap.

The collator produces BOTH branches of the distillation in one batch:

  teacher  input_ids with the full 64 ``<image>`` placeholders per tile
  student  the same sequence with each 64-run rewritten to ``num_queries``

Every non-image token is identical and in the same order in the two sequences,
which is what makes the losses alignable -- see ``text_index`` below.
"""

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _decode_image(ref: Any) -> Image.Image:
    """Accept a data URI, a local path, or a raw base64 blob -> RGB PIL image."""
    if isinstance(ref, dict):
        ref = ref.get("url") or ref.get("path") or ref.get("image")
    if not isinstance(ref, str):
        raise ValueError(f"unsupported image reference: {type(ref)}")
    if ref.startswith("data:"):
        payload = ref.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if os.path.exists(ref):
        return Image.open(ref).convert("RGB")
    # Bare base64 with no data: prefix.
    return Image.open(io.BytesIO(base64.b64decode(ref))).convert("RGB")


def _row_to_messages(row: Dict) -> (List[Dict], List[Image.Image]):
    """Convert a corpus row into chat-template messages + the images it cites."""
    messages, images = [], []
    for turn in row.get("conversations", []):
        content = turn.get("content")
        parts = []
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        else:
            for part in content or []:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif ptype in ("image", "image_url"):
                    images.append(_decode_image(part.get(ptype)))
                    parts.append({"type": "image"})
        messages.append({"role": turn.get("role", "user"), "content": parts})
    return messages, images


class JsonlOffsetDataset(Dataset):
    """Random access into a large jsonl via a cached byte-offset index."""

    def __init__(self, path: str, index_path: Optional[str] = None, limit: int = 0):
        self.path = path
        self.index_path = index_path or (path + ".offsets.npy")
        self.offsets = self._load_or_build_index()
        if limit:
            self.offsets = self.offsets[:limit]
        self._fh = None

    def _load_or_build_index(self) -> np.ndarray:
        if os.path.exists(self.index_path) and os.path.getmtime(
            self.index_path
        ) >= os.path.getmtime(self.path):
            return np.load(self.index_path)
        offsets, pos = [], 0
        with open(self.path, "rb") as f:
            for line in f:
                offsets.append(pos)
                pos += len(line)
        arr = np.asarray(offsets, dtype=np.int64)
        tmp = self.index_path + ".tmp"
        # np.save(path, ...) would append a second ".npy" to the tmp name; write
        # through a handle so the file lands exactly where os.replace expects it.
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, self.index_path)
        return arr

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, i: int) -> Dict:
        # Per-worker handle: DataLoader workers must not share a file position.
        if self._fh is None:
            self._fh = open(self.path, "rb")
        self._fh.seek(int(self.offsets[i]))
        return json.loads(self._fh.readline())


class QSamplerCollator:
    """Build the teacher/student token pair and the alignment index."""

    def __init__(
        self,
        processor,
        num_queries: int,
        tile_tokens: int = 64,
        max_length: int = 2048,
        image_token_id: Optional[int] = None,
    ):
        self.processor = processor
        self.num_queries = num_queries
        self.tile_tokens = tile_tokens
        self.max_length = max_length
        self.image_token_id = (
            image_token_id
            if image_token_id is not None
            else processor.tokenizer.convert_tokens_to_ids("<image>")
        )
        self.pad_token_id = processor.tokenizer.pad_token_id or 0

    def _shrink_image_runs(self, ids: List[int]) -> List[int]:
        """Rewrite every run of ``tile_tokens`` image ids to ``num_queries``."""
        out, i, n = [], 0, len(ids)
        while i < n:
            if ids[i] == self.image_token_id:
                j = i
                while j < n and ids[j] == self.image_token_id:
                    j += 1
                run = j - i
                if run % self.tile_tokens != 0:
                    raise ValueError(
                        f"image run of {run} is not a multiple of {self.tile_tokens}"
                    )
                tiles = run // self.tile_tokens
                out.extend([self.image_token_id] * (tiles * self.num_queries))
                i = j
            else:
                out.append(ids[i])
                i += 1
        return out

    def __call__(self, rows: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
        teacher_ids, student_ids, pixel_values = [], [], []
        for row in rows:
            try:
                messages, images = _row_to_messages(row)
                if not images:
                    continue
                text = self.processor.apply_chat_template(
                    messages, add_generation_prompt=False
                )
                enc = self.processor(text=text, images=images, return_tensors="pt")
                ids = enc["input_ids"][0].tolist()
                if len(ids) > self.max_length:
                    continue  # truncating would cut an image run in half
                teacher_ids.append(ids)
                student_ids.append(self._shrink_image_runs(ids))
                pixel_values.append(enc["pixel_values"][0])
            except Exception:
                continue  # a single corrupt row must not kill the step
        if not teacher_ids:
            return None

        def pad(seqs):
            m = max(len(s) for s in seqs)
            out = torch.full((len(seqs), m), self.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), m), dtype=torch.long)
            for k, s in enumerate(seqs):
                out[k, : len(s)] = torch.tensor(s, dtype=torch.long)
                mask[k, : len(s)] = 1
            return out, mask

        t_ids, t_mask = pad(teacher_ids)
        s_ids, s_mask = pad(student_ids)

        # Alignment: drop image positions from both sides and the remaining
        # tokens correspond 1:1, in order. Padded to a common width with a
        # validity mask so the loss can gather without a Python loop.
        t_text = [
            [p for p, v in enumerate(seq) if v != self.image_token_id]
            for seq in teacher_ids
        ]
        s_text = [
            [p for p, v in enumerate(seq) if v != self.image_token_id]
            for seq in student_ids
        ]
        for a, b in zip(t_text, s_text):
            assert len(a) == len(b), "teacher/student text token counts diverged"
        w = max(len(x) for x in t_text)
        t_idx = torch.zeros((len(t_text), w), dtype=torch.long)
        s_idx = torch.zeros((len(s_text), w), dtype=torch.long)
        v_msk = torch.zeros((len(t_text), w), dtype=torch.bool)
        for k, (a, b) in enumerate(zip(t_text, s_text)):
            t_idx[k, : len(a)] = torch.tensor(a)
            s_idx[k, : len(b)] = torch.tensor(b)
            v_msk[k, : len(a)] = True

        # Tiles are flattened across the batch before the connector, so pad the
        # per-sample tile counts to a common max (all-zero tiles are dropped
        # inside get_image_features).
        max_tiles = max(p.shape[0] for p in pixel_values)
        pv = torch.zeros(
            (len(pixel_values), max_tiles, *pixel_values[0].shape[1:]),
            dtype=pixel_values[0].dtype,
        )
        for k, p in enumerate(pixel_values):
            pv[k, : p.shape[0]] = p

        return {
            "teacher_input_ids": t_ids,
            "teacher_attention_mask": t_mask,
            "student_input_ids": s_ids,
            "student_attention_mask": s_mask,
            "pixel_values": pv,
            "teacher_text_index": t_idx,
            "student_text_index": s_idx,
            "text_valid_mask": v_msk,
        }
