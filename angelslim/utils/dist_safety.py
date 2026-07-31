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

"""Work around broken single-rank NCCL collectives under torchrun.

On some CUDA/NCCL/torch stacks, ``torchrun --nproc_per_node=1`` still inits a
process group, and HuggingFace Trainer then calls ``dist.all_gather`` during
logging / end-of-train. That path can SIGSEGV (-11) even though:

- plain ``python`` (no process group) works
- training steps themselves (incl. DDP all_reduce) may work

Semantically, world_size==1 collectives are no-ops. Short-circuiting them is
correct and avoids the bad NCCL single-rank path.

Disable with: ``ANGELSLIM_DIST_SAFETY=0``

Multi-GPU (world_size>1) is unchanged — fix NCCL/torch on the machine if those
collectives still crash.
"""

from __future__ import annotations

import os
from typing import Any, Optional

_APPLIED = False


def _world_size(group: Any = None) -> int:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return 1
    try:
        return int(dist.get_world_size(group))
    except Exception:
        return 1


def apply_single_rank_dist_safety(verbose: bool = True) -> bool:
    """Patch all_gather / barrier / HF distributed_concat for world_size<=1."""
    global _APPLIED
    if _APPLIED:
        return True
    if os.environ.get("ANGELSLIM_DIST_SAFETY", "1").lower() in ("0", "false", "no", "off"):
        return False

    import torch.distributed as dist

    # --- torch.distributed.all_gather ---
    if hasattr(dist, "all_gather"):
        _orig_all_gather = dist.all_gather

        def _safe_all_gather(tensor_list, tensor, group=None, async_op=False, **kwargs):
            if _world_size(group) <= 1:
                if not tensor_list:
                    tensor_list.append(tensor.detach().clone())
                else:
                    if tensor_list[0].shape != tensor.shape:
                        tensor_list[0].resize_as_(tensor)
                    tensor_list[0].copy_(tensor)
                if async_op:

                    class _Done:
                        def wait(self):
                            return None

                        def is_completed(self):
                            return True

                    return _Done()
                return None
            return _orig_all_gather(tensor_list, tensor, group=group, async_op=async_op, **kwargs)

        dist.all_gather = _safe_all_gather  # type: ignore[assignment]

    # --- all_gather_into_tensor (newer API) ---
    if hasattr(dist, "all_gather_into_tensor"):
        _orig_agit = dist.all_gather_into_tensor

        def _safe_agit(output_tensor, input_tensor, group=None, async_op=False, **kwargs):
            if _world_size(group) <= 1:
                output_tensor.copy_(input_tensor)
                if async_op:

                    class _Done:
                        def wait(self):
                            return None

                        def is_completed(self):
                            return True

                    return _Done()
                return None
            return _orig_agit(
                output_tensor, input_tensor, group=group, async_op=async_op, **kwargs
            )

        dist.all_gather_into_tensor = _safe_agit  # type: ignore[assignment]

    # --- barrier (end-of-train wait_for_everyone) ---
    if hasattr(dist, "barrier"):
        _orig_barrier = dist.barrier

        def _safe_barrier(group=None, async_op=False, device_ids=None, **kwargs):
            if _world_size(group) <= 1:
                if async_op:

                    class _Done:
                        def wait(self):
                            return None

                        def is_completed(self):
                            return True

                    return _Done()
                return None
            # device_ids only valid for some backends / torch versions
            try:
                return _orig_barrier(
                    group=group, async_op=async_op, device_ids=device_ids, **kwargs
                )
            except TypeError:
                return _orig_barrier(group=group, async_op=async_op, **kwargs)

        dist.barrier = _safe_barrier  # type: ignore[assignment]

    # --- HF Trainer logging path ---
    try:
        import transformers.trainer_pt_utils as tpu

        if hasattr(tpu, "distributed_concat"):
            _orig_dc = tpu.distributed_concat

            def _safe_distributed_concat(tensor, num_total_examples: Optional[int] = None):
                if _world_size() <= 1:
                    return tensor
                return _orig_dc(tensor, num_total_examples)

            tpu.distributed_concat = _safe_distributed_concat
    except Exception:
        pass

    _APPLIED = True
    if verbose:
        from .utils import rank0_print

        rank0_print(
            "[dist_safety] Enabled single-rank collective short-circuit "
            "(all_gather/barrier/HF distributed_concat when world_size<=1). "
            "Set ANGELSLIM_DIST_SAFETY=0 to disable."
        )
    return True
