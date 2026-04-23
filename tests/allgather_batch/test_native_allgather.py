# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline control: PyTorch native all_gather_into_tensor over the same
size ladder as test_ccu_ms_size_boundary, driven with async_op=True.

Structure mirrors tests/allgather_batch/test_eager.py::TestCcuPath line-for-
line; only the op call differs (dist.all_gather_into_tensor with
async_op=True + work.wait(), instead of torch.ops.custom_comm.allgather_batch).

Goal: if this baseline also flakes on the consecutive parametrized pattern,
race is in HCCL engine drain between back-to-back async launches, not in our
CCU MS kernel. If this baseline is stable while test_ccu_ms_size_boundary
still flakes, race is inside our kernel (XN live-read / LoopGroup / PostSync).

Usage:
    torchrun --nproc_per_node=4 -m pytest tests/allgather_batch/test_native_allgather.py -v --tb=short
"""

import pytest
import torch
import torch.distributed as dist


@pytest.mark.npu
@pytest.mark.ext
@pytest.mark.dist
class TestNativeAllGather:
    """Native HCCL allgather baseline with async_op=True."""

    @pytest.fixture(autouse=True)
    def _bind(self, dist_ctx):
        self.rank = dist_ctx.rank
        self.world_size = dist_ctx.world_size
        self.device = dist_ctx.device
        self.hcom = dist_ctx.hcom

    @pytest.mark.parametrize("bytes_per_desc", [
        256,
        1024,
        2048,
        4096,
        4097,
        8192,
        32768,
        49152,
        65536,
        262144,
        262145,
        1048576,
        16777216
    ])
    def test_native_allgather_size_boundary(self, bytes_per_desc):
        """Same deterministic input / reference as test_ccu_ms_size_boundary,
        but uses torch.distributed.all_gather_into_tensor with async_op=True
        + work.wait() instead of the custom_comm op.
        """
        n = bytes_per_desc
        data = (torch.arange(n, dtype=torch.int64) + self.rank).to(torch.int8).to(self.device)

        out_ms = torch.empty(n * self.world_size, dtype=torch.int8, device=self.device)
        work = dist.all_gather_into_tensor(out_ms, data, async_op=True)
        work.wait()

        expected = torch.cat([
            (torch.arange(n, dtype=torch.int64) + r).to(torch.int8)
            for r in range(self.world_size)
        ]).to(self.device)

        diff_idx = (out_ms != expected).nonzero()
        assert torch.equal(out_ms, expected), (
            f"bytes_per_desc={bytes_per_desc}: "
            f"{diff_idx.numel()} / {expected.numel()} positions differ; "
            f"first diff indices: {diff_idx[:10].flatten().tolist()}")
