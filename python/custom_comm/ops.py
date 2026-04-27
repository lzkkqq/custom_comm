# Copyright (c) 2026 custom_comm Authors. All rights reserved.
"""Public API for custom_comm operators."""

from typing import List

import torch

import custom_comm._C  # noqa: F401  (registers torch.ops + pybind11 eager path)


def allgather_batch(
    inputs: List[torch.Tensor],
    hcom: str,
    world_size: int,
) -> List[torch.Tensor]:
    """Batched AllGather for heterogeneous-dtype tensors.

    Args:
        inputs:     List of contiguous tensors with identical dim-0.
        hcom:       HCCL communicator group name.
        world_size: Number of ranks in the group.

    Returns:
        List of gathered tensors (dim-0 scaled by world_size).
    """
    # Keep the Python surface area simple and always delegate to the registered
    # Dispatcher op. This ensures eager warmup, torch.compile, and aclGraph
    # capture all share the same NPU implementation and stream semantics.
    return torch.ops.custom_comm.allgather_batch(inputs, hcom, world_size)
