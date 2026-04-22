# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager-mode tests for the allgather_batch operator.

Layered as:
  TestMetaKernel     — Meta-dispatch shape inference (no NPU)
  TestNpuFunctional  — decomposed path decomposed HCCL path (needs NPU + torchrun)
  TestCcuPath        — CCU path CCU kernel path (needs NPU + torchrun, CUSTOM_COMM_USE_CCU=1)

Graph-mode counterpart: ./test_graph.py
Performance numbers:    ./bench.py

Usage:
  pytest tests/                                         # runs meta-only (others auto-skip)
  torchrun --nproc_per_node=N -m pytest tests/          # full eager suite
"""

import os
import pytest
import torch

DTYPES = [torch.int8, torch.float16, torch.float32, torch.bfloat16]


def make_input(shape, dtype, device="meta"):
    if device == "meta":
        return torch.empty(shape, dtype=dtype, device="meta")
    return torch.randn(shape, device=device).to(dtype)


# ============================================================
# Meta kernel tests (no NPU required)
# ============================================================

@pytest.mark.ext
class TestMetaKernel:
    """Shape inference via Meta dispatch. Runs anywhere."""

    def _call(self, inputs, world_size):
        return torch.ops.custom_comm.allgather_batch(inputs, "dummy", world_size)

    @pytest.mark.parametrize("world_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_single_desc(self, world_size, dtype):
        inp = make_input((128, 64), dtype)
        [out] = self._call([inp], world_size)
        assert out.shape == (128 * world_size, 64)
        assert out.dtype == dtype

    @pytest.mark.parametrize("world_size", [2, 4, 8])
    def test_heterogeneous_dtypes(self, world_size):
        """Multiple tensors with different dtypes."""
        inp_i8 = make_input((256,), torch.int8)
        inp_f32 = make_input((4,), torch.float32)
        outs = self._call([inp_i8, inp_f32], world_size)
        assert outs[0].shape == (256 * world_size,)
        assert outs[0].dtype == torch.int8
        assert outs[1].shape == (4 * world_size,)
        assert outs[1].dtype == torch.float32

    def test_ag09_meta(self):
        """OPT-AG-04: INT8 data + FP32 scale + INT32 topk_ids (3 descs)."""
        data = torch.empty(2048, dtype=torch.int8, device="meta")
        scale = torch.empty(56, dtype=torch.float32, device="meta")
        ids = torch.empty(8, dtype=torch.int32, device="meta")
        outs = torch.ops.custom_comm.allgather_batch(
            [data, scale, ids], "dummy", 8
        )
        assert outs[0].shape == (2048 * 8,)
        assert outs[1].shape == (56 * 8,)
        assert outs[2].shape == (8 * 8,)

    def test_max_desc_count(self):
        inputs = [torch.empty(32, 16, device="meta") for _ in range(8)]
        outs = self._call(inputs, 4)
        assert len(outs) == 8
        assert all(o.shape == (128, 16) for o in outs)

    def _call(self, inputs, ws):
        return torch.ops.custom_comm.allgather_batch(inputs, "hcom", ws)

    def test_empty_dim0(self):
        [out] = self._call([torch.empty(0, 64, device="meta")], 4)
        assert out.shape == (0, 64)

    def test_preserves_dtype(self):
        for dt in [torch.int8, torch.float16, torch.float32, torch.bfloat16]:
            [out] = self._call([torch.empty(16, dtype=dt, device="meta")], 2)
            assert out.dtype == dt


# ============================================================
# NPU functional tests (require device + multi-rank)
# ============================================================

@pytest.mark.npu
@pytest.mark.ext
@pytest.mark.dist
class TestNpuFunctional:
    """decomposed path eager-mode correctness on NPU.

    Run: torchrun --nproc_per_node=N pytest tests/ -k TestNpuFunctional
    """

    @pytest.fixture(autouse=True)
    def _bind(self, dist_ctx):
        self.rank = dist_ctx.rank
        self.world_size = dist_ctx.world_size
        self.device = dist_ctx.device
        self.hcom = dist_ctx.hcom

    # ---- basic correctness ----

    @pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.float32, torch.bfloat16])
    def test_single_desc(self, dtype):
        n = 128
        # Build on CPU then transfer to avoid NPU arange dtype limitations.
        inp = (torch.arange(n) + self.rank * n).to(dtype).to(self.device).contiguous()
        outs = torch.ops.custom_comm.allgather_batch([inp], self.hcom, self.world_size)
        assert outs[0].shape == (n * self.world_size,)
        for r in range(self.world_size):
            expected = (torch.arange(n) + r * n).to(dtype).to(self.device).contiguous()
            assert torch.equal(outs[0][r * n:(r + 1) * n], expected)

    def test_heterogeneous_int8_fp32(self):
        """OPT-AG-09 core: INT8 + FP32 scale."""
        data = torch.full((2048,), self.rank + 1, dtype=torch.int8, device=self.device)
        scale = torch.full((4,), (self.rank + 1) * 0.5, dtype=torch.float32, device=self.device)
        outs = torch.ops.custom_comm.allgather_batch([data, scale], self.hcom, self.world_size)
        assert len(outs) == 2
        assert outs[0].shape == (2048 * self.world_size,)
        assert outs[1].shape == (4 * self.world_size,)

    def test_three_tensor_pack(self):
        """OPT-AG-04/09: INT8 data + FP32 scale + INT32 topk_ids."""
        x = torch.randint(0, 127, (2048,), dtype=torch.int8, device=self.device)
        s = torch.randn(56, dtype=torch.float32, device=self.device)
        ids = torch.randint(0, 1000, (8,), dtype=torch.int32, device=self.device)
        outs = torch.ops.custom_comm.allgather_batch(
            [x, s, ids], self.hcom, self.world_size
        )
        assert outs[0].shape == (2048 * self.world_size,)
        assert outs[1].shape == (56 * self.world_size,)
        assert outs[2].shape == (8 * self.world_size,)

    def test_repeated_calls(self):
        data = torch.ones(64, device=self.device, dtype=torch.float16)
        for _ in range(100):
            torch.ops.custom_comm.allgather_batch([data], self.hcom, self.world_size)


# ============================================================
# CCU path tests (CCU path, CUSTOM_COMM_USE_CCU=1)
# ============================================================

@pytest.mark.npu
@pytest.mark.ext
@pytest.mark.dist
class TestCcuPath:
    """CCU path CCU kernel tests. Run with CUSTOM_COMM_USE_CCU=1."""

    @pytest.fixture(autouse=True)
    def _bind(self, dist_ctx):
        self.rank = dist_ctx.rank
        self.world_size = dist_ctx.world_size
        self.device = dist_ctx.device
        self.hcom = dist_ctx.hcom

    def test_ccu_only(self):
        """CCU path output matches the expected all-gather of deterministic inputs."""
        import os
        data = (torch.arange(256) + self.rank).to(torch.int8).to(self.device)
        scale = (torch.arange(4, dtype=torch.float32) + self.rank * 4).to(self.device)

        os.environ["CUSTOM_COMM_USE_CCU"] = "1"
        try:
            out_data, out_scale = torch.ops.custom_comm.allgather_batch(
                [data, scale], self.hcom, self.world_size
            )
        finally:
            os.environ.pop("CUSTOM_COMM_USE_CCU", None)

        expected_data = torch.cat([
            (torch.arange(256) + r).to(torch.int8) for r in range(self.world_size)
        ]).to(self.device)
        expected_scale = torch.arange(4 * self.world_size, dtype=torch.float32).to(self.device)

        assert torch.equal(out_data, expected_data)
        assert torch.equal(out_scale, expected_scale)

    @pytest.mark.parametrize("impl", ["v1", "v2"])
    @pytest.mark.parametrize("bytes_per_desc", [
        1024,     # 1 KB -- well under single-slot limit
        2048,     # 2 KB
        4096,     # 4 KB -- exactly CCU_MS_SIZE, right at the boundary
        4097,     # 4 KB + 1 -- first size past single-slot capacity
        8192,     # 8 KB -- 2x single slot; should fail on Phase 2b-alpha
        65536,    # 64 KB -- many slots; definitely needs Phase 2b-gamma
    ])
    def test_ccu_ms_size_boundary(self, bytes_per_desc, impl):
        """Probe CalGoSize boundaries: small -> 4 KB -> multi-slot.

        For each (payload size, MS impl) combination, run MS path and diff
        against SCHED path. impl=v1 is the hand-rolled kernel; impl=v2 uses
        GroupBroadcastBatch via CcuKernelAlgBase. Both must byte-exactly
        match SCHED, which is the golden reference.
        """
        import os
        # Use deterministic per-rank payload
        n_elts = bytes_per_desc
        data = (torch.arange(n_elts, dtype=torch.int8) + self.rank).to(self.device)

        # Run MS path (v1 or v2 per parametrization)
        os.environ["CUSTOM_COMM_USE_CCU"] = "1"
        os.environ["CUSTOM_COMM_CCU_MODE"] = "ms"
        os.environ["CUSTOM_COMM_CCU_MS_IMPL"] = impl
        os.environ["CUSTOM_COMM_CCU_MS_DIAG"] = "1"
        try:
            [out_ms] = torch.ops.custom_comm.allgather_batch(
                [data], self.hcom, self.world_size)
        finally:
            pass

        # Run SCHED path (reference truth)
        os.environ["CUSTOM_COMM_CCU_MODE"] = "sched"
        os.environ.pop("CUSTOM_COMM_CCU_MS_IMPL", None)
        [out_sched] = torch.ops.custom_comm.allgather_batch(
            [data], self.hcom, self.world_size)

        os.environ.pop("CUSTOM_COMM_USE_CCU", None)
        os.environ.pop("CUSTOM_COMM_CCU_MODE", None)
        os.environ.pop("CUSTOM_COMM_CCU_MS_DIAG", None)

        assert torch.equal(out_ms, out_sched), (
            f"MS({impl}) vs SCHED mismatch at bytes_per_desc={bytes_per_desc}; "
            f"diff at {(out_ms != out_sched).nonzero()}")

    def test_ccu_ms_v2_only(self):
        """Smoke test: V2 kernel (GroupBroadcastBatch) produces the expected
        all-gather of deterministic inputs. Mirrors test_ccu_only but forces
        CCU_MODE=ms and CCU_MS_IMPL=v2 so the V2 code path is exercised
        without relying on the v1-vs-SCHED diff from size_boundary.
        """
        import os
        data = (torch.arange(256) + self.rank).to(torch.int8).to(self.device)
        scale = (torch.arange(4, dtype=torch.float32) + self.rank * 4).to(self.device)

        os.environ["CUSTOM_COMM_USE_CCU"]     = "1"
        os.environ["CUSTOM_COMM_CCU_MODE"]    = "ms"
        os.environ["CUSTOM_COMM_CCU_MS_IMPL"] = "v2"
        try:
            out_data, out_scale = torch.ops.custom_comm.allgather_batch(
                [data, scale], self.hcom, self.world_size
            )
        finally:
            os.environ.pop("CUSTOM_COMM_USE_CCU", None)
            os.environ.pop("CUSTOM_COMM_CCU_MODE", None)
            os.environ.pop("CUSTOM_COMM_CCU_MS_IMPL", None)

        expected_data = torch.cat([
            (torch.arange(256) + r).to(torch.int8) for r in range(self.world_size)
        ]).to(self.device)
        expected_scale = torch.arange(4 * self.world_size, dtype=torch.float32).to(self.device)

        assert torch.equal(out_data, expected_data)
        assert torch.equal(out_scale, expected_scale)
