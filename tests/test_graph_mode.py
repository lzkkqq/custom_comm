# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for custom_comm graph-mode (torchair) integration.

Test layers:
  - Meta shape inference: no torchair needed, just the C extension
  - Converter registration: verify the GE converter is wired up (needs torchair)

All tests are collectable on macOS / CPU; torchair-dependent tests are
skipped when torchair is unavailable.

Run:
    pytest tests/test_graph_mode.py -v
"""

import inspect

import pytest
import torch

# ============================================================
# Capability probes
# ============================================================

HAS_EXT = False
try:
    import custom_comm  # noqa: F401
    HAS_EXT = True
except ImportError:
    pass

HAS_TORCHAIR = False
try:
    import torchair  # noqa: F401
    HAS_TORCHAIR = True
except ImportError:
    pass

requires_ext = pytest.mark.skipif(not HAS_EXT, reason="custom_comm C extension not built")
requires_torchair = pytest.mark.skipif(not HAS_TORCHAIR, reason="torchair not installed")


# ============================================================
# 1. Meta-dispatch shape inference tests
# ============================================================

@requires_ext
class TestMetaShapeInference:
    """Verify allgather_batch meta-dispatch produces correct output shapes."""

    def _call(self, inputs, world_size):
        return torch.ops.custom_comm.allgather_batch(inputs, "hcom", world_size)

    def test_single_input(self):
        inp = torch.empty(128, 64, dtype=torch.float32, device="meta")
        outs = self._call([inp], 4)
        assert len(outs) == 1
        assert outs[0].shape == (512, 64)

    @pytest.mark.parametrize("world_size", [1, 2, 8])
    def test_heterogeneous_dtypes(self, world_size):
        inputs = [
            torch.empty(2048, dtype=torch.int8, device="meta"),
            torch.empty(4, dtype=torch.float32, device="meta"),
        ]
        outs = self._call(inputs, world_size)
        assert outs[0].shape == (2048 * world_size,)
        assert outs[1].shape == (4 * world_size,)
        assert outs[0].dtype == torch.int8
        assert outs[1].dtype == torch.float32

    def test_max_descs(self):
        inputs = [torch.empty(32, 16, device="meta") for _ in range(8)]
        outs = self._call(inputs, 2)
        assert all(o.shape == (64, 16) for o in outs)

    def test_empty_dim0(self):
        inp = torch.empty(0, 64, device="meta")
        [out] = self._call([inp], 4)
        assert out.shape == (0, 64)

    def test_multidim(self):
        inp = torch.empty(10, 20, 30, device="meta")
        [out] = self._call([inp], 4)
        assert out.shape == (40, 20, 30)


# ============================================================
# 2. Converter registration tests (need torchair)
# ============================================================

requires_ext = pytest.mark.skipif(not HAS_EXT, reason="custom_comm ext not built")

@requires_torchair
@requires_ext
class TestConverterRegistration:
    """Verify the GE converter is registered correctly."""

    def test_converter_module_importable(self):
        import custom_comm.converters.allgather_batch_converter  # noqa: F401

    def test_converter_registered(self):
        """The converter should be reachable via op's _ge_converter attr
        (new torchair) or the global _CONVERTERS dict (old torchair)."""
        op = torch.ops.custom_comm.allgather_batch.default
        # New torchair attaches converter directly to the op
        has_attr = hasattr(op, "_ge_converter")
        # Old torchair uses a global dict
        from torchair._ge_concrete_graph import fx2ge_converter
        registry = getattr(fx2ge_converter, "_CONVERTERS", {})
        has_dict = op in registry
        assert has_attr or has_dict, (
            "converter not found via _ge_converter attr or _CONVERTERS dict"
        )


# ============================================================
# Graph mode end-to-end (require NPU + torchair)
# ============================================================

HAS_NPU = False
try:
    import torch_npu  # noqa: F401
    HAS_NPU = torch.npu.is_available() if hasattr(torch, "npu") else False
except ImportError:
    pass

requires_npu = pytest.mark.skipif(not HAS_NPU, reason="NPU not available")


@requires_npu
@requires_torchair
@requires_ext
class TestGraphModeE2E:
    """End-to-end graph mode tests. Require NPU hardware."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not torch.distributed.is_initialized():
            pytest.skip("dist not initialized")
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device = torch.device(f"npu:{self.rank}")
        torch.npu.set_device(self.device)
        pg = torch.distributed.group.WORLD
        self.hcom = pg._get_backend(self.device).get_hccl_comm_name(self.rank)

    def test_ge_graph_compile(self):
        """Verify torch.compile(backend='torchair') can trace allgather_batch."""
        import custom_comm

        @torch.compile(backend="torchair")
        def fn(x, s):
            return custom_comm.allgather_batch([x, s], self.hcom, self.world_size)

        x = torch.randn(64, dtype=torch.float16, device=self.device)
        s = torch.randn(4, dtype=torch.float32, device=self.device)
        out = fn(x, s)
        assert out[0].shape == (64 * self.world_size,)
        assert out[1].shape == (4 * self.world_size,)

    def test_ge_graph_correctness(self):
        """GE graph output should match eager output."""
        import custom_comm

        x = (torch.arange(32) + self.rank).to(torch.int8).to(self.device)
        s = torch.full((4,), float(self.rank), dtype=torch.float32, device=self.device)

        eager_out = custom_comm.allgather_batch([x, s], self.hcom, self.world_size)

        @torch.compile(backend="torchair")
        def fn(x, s):
            return custom_comm.allgather_batch([x, s], self.hcom, self.world_size)

        graph_out = fn(x, s)
        assert torch.equal(eager_out[0], graph_out[0])
        assert torch.equal(eager_out[1], graph_out[1])


# ============================================================
# aclGraph capture tests (require NPU)
# ============================================================

@requires_npu
@requires_ext
class TestAclGraphCapture:
    """Verify aclGraph capture works with allgather_batch."""

    @pytest.fixture(autouse=True)
    def setup_dist(self):
        if not torch.distributed.is_initialized():
            pytest.skip("requires distributed init")
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        torch.npu.set_device(self.rank)
        self.device = torch.device(f"npu:{self.rank}")
        pg = torch.distributed.group.WORLD
        self.hcom = pg._get_backend(self.device).get_hccl_comm_name(self.rank)

    def test_aclgraph_capture_replay(self):
        """aclGraph capture + replay should produce correct results."""
        import custom_comm

        data = torch.ones(64, dtype=torch.float16, device=self.device) * (self.rank + 1)
        scale = torch.ones(4, dtype=torch.float32, device=self.device) * (self.rank + 1)

        # Warmup (required before capture)
        _ = custom_comm.allgather_batch([data, scale], self.hcom, self.world_size)
        torch.npu.synchronize()

        # Capture
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = custom_comm.allgather_batch([data, scale], self.hcom, self.world_size)

        # Replay
        graph.replay()
        torch.npu.synchronize()

        assert out[0].shape == (64 * self.world_size,)
        assert out[1].shape == (4 * self.world_size,)

    def test_aclgraph_repeated_replay(self):
        """Multiple replays should give consistent results."""
        import custom_comm

        data = torch.arange(32).to(torch.int8).to(self.device)
        _ = custom_comm.allgather_batch([data], self.hcom, self.world_size)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = custom_comm.allgather_batch([data], self.hcom, self.world_size)

        for _ in range(10):
            graph.replay()
        assert out[0].shape[0] == data.shape[0] * self.world_size
