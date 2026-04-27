# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-mode (torchair / torch.compile / aclGraph) tests for allgather_batch.

Layers:
  - TestMetaShapeInference : shape-only, meta-dispatch
  - TestConverterRegistration : torchair GE converter registration
  - TestGraphModeE2E : torch.compile(backend=torchair) end-to-end on NPU
  - TestAclGraphCapture : NPU aclGraph capture + replay

Eager-mode counterparts live in ./test_eager.py.

Run:
    pytest tests/allgather_batch/test_graph.py -v
"""

import pytest
import torch
import torch.distributed as dist


# ============================================================
# 1. Meta-dispatch shape inference tests
# ============================================================

@pytest.mark.ext
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

@pytest.mark.torchair
@pytest.mark.ext
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

@pytest.mark.npu
@pytest.mark.ext
@pytest.mark.torchair
@pytest.mark.dist
class TestGraphModeE2E:
    """End-to-end graph mode tests. Require NPU hardware."""

    @pytest.fixture(autouse=True)
    def _bind(self, dist_ctx):
        self.rank = dist_ctx.rank
        self.world_size = dist_ctx.world_size
        self.device = dist_ctx.device
        self.hcom = dist_ctx.hcom

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

@pytest.mark.npu
@pytest.mark.ext
@pytest.mark.dist
class TestAclGraphCapture:
    """aclGraph capture/replay correctness."""

    @pytest.fixture(autouse=True)
    def _setup(self, dist_ctx):
        self.rank = dist_ctx.rank
        self.world_size = dist_ctx.world_size
        self.device = dist_ctx.device
        self.hcom = dist_ctx.hcom

    def _gather_expected(self, values_by_rank, dtype):
        return torch.cat([
            torch.as_tensor(v, dtype=dtype, device=self.device)
            for v in values_by_rank
        ])

    def _capture_graph(self, fn):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = fn()
        return graph, outputs

    def _manual_capture_graph(self, fn, stream_=None):
        import torch_npu  # noqa: F401

        if stream_ is None:
            stream_ = torch_npu.npu.Stream()
        with torch_npu.npu.stream(stream_):
            graph = torch_npu.npu.NPUGraph()
            graph.capture_begin()
            outputs = fn()
            graph.capture_end()
        return stream_, graph, outputs

    def _manual_replay(self, stream_, graph):
        import torch_npu  # noqa: F401

        with torch_npu.npu.stream(stream_):
            graph.replay()
        torch.npu.synchronize()


    def test_aclgraph_capture_replay(self):
        """aclGraph capture + replay should produce correct results."""
        import custom_comm

        # Capture uses static tensors; replay should consume updated contents
        # after in-place copies, mirroring the canonical aclGraph testing style.
        static_data = torch.ones(64, dtype=torch.float16, device=self.device) * (self.rank + 1)
        static_scale = torch.ones(4, dtype=torch.float32, device=self.device) * (self.rank + 1)

        # Warmup (required before capture)
        _ = custom_comm.allgather_batch([static_data, static_scale], self.hcom, self.world_size)
        torch.npu.synchronize()

        stream_, graph, out = self._manual_capture_graph(
            lambda: custom_comm.allgather_batch([static_data, static_scale], self.hcom, self.world_size)
        )

        real_data = torch.full(
            static_data.shape,
            self.rank + 11,
            dtype=static_data.dtype,
            device=self.device,
        )
        real_scale = torch.arange(4, dtype=torch.float32, device=self.device) + self.rank * 10

        static_data.copy_(real_data)
        static_scale.copy_(real_scale)

        self._manual_replay(stream_, graph)

        expected_data = self._gather_expected(
            [torch.ones(64, dtype=torch.float16) * (r + 11) for r in range(self.world_size)],
            torch.float16,
        )
        expected_scale = self._gather_expected(
            [torch.arange(4, dtype=torch.float32) + r * 10 for r in range(self.world_size)],
            torch.float32,
        )

        assert out[0].shape == (64 * self.world_size,)
        assert out[1].shape == (4 * self.world_size,)
        assert torch.equal(out[0], expected_data)
        assert torch.equal(out[1], expected_scale)

    def test_aclgraph_multiple_graph_instances(self):
        """Two captured graphs with different ops should coexist and replay independently."""
        import custom_comm

        static_b = torch.arange(16, dtype=torch.float32, device=self.device) + self.rank * 10

        dist.broadcast(static_b, src=0)
        torch.npu.synchronize()
        stream_, graph_b, _ = self._manual_capture_graph(
            lambda: (dist.broadcast(static_b, src=0), static_b)[1]
        )

        real_b = torch.arange(16, dtype=torch.float32, device=self.device) + self.rank * 10 + 5
        static_b.copy_(real_b)
        self._manual_replay(stream_, graph_b)
        expected_b = torch.arange(16, dtype=torch.float32, device=self.device) + 5
        assert torch.equal(static_b, expected_b)

        static_a = (
            torch.arange(64, dtype=torch.int64) + self.rank * 64
        ).to(torch.int8).to(self.device)
        _ = custom_comm.allgather_batch([static_a], self.hcom, self.world_size)
        torch.npu.synchronize()
        stream_a, graph_a, out_a = self._manual_capture_graph(
            lambda: custom_comm.allgather_batch([static_a], self.hcom, self.world_size),
            stream_=stream_,
        )

        real_a = (
            torch.arange(64, dtype=torch.int64) + self.rank * 64 + 9
        ).to(torch.int8).to(self.device)
        static_a.copy_(real_a)
        self._manual_replay(stream_a, graph_a)
        expected_a = self._gather_expected(
            [torch.arange(64, dtype=torch.int64) + r * 64 + 9 for r in range(self.world_size)],
            torch.int8,
        )
        assert torch.equal(out_a[0], expected_a)

        real_a_2 = (
            torch.arange(64, dtype=torch.int64) + self.rank * 64 + 15
        ).to(torch.int8).to(self.device)
        static_a.copy_(real_a_2)
        self._manual_replay(stream_a, graph_a)
        expected_a_2 = self._gather_expected(
            [torch.arange(64, dtype=torch.int64) + r * 64 + 15 for r in range(self.world_size)],
            torch.int8,
        )
        assert torch.equal(out_a[0], expected_a_2)

        real_b_2 = torch.arange(16, dtype=torch.float32, device=self.device) + self.rank * 10 + 7
        static_b.copy_(real_b_2)
        self._manual_replay(stream_, graph_b)
        expected_b_2 = torch.arange(16, dtype=torch.float32, device=self.device) + 7
        assert torch.equal(static_b, expected_b_2)

    def test_aclgraph_repeated_replay(self):
        """Multiple replays should give consistent results."""
        import custom_comm

        static_data = (torch.arange(32) + self.rank * 32).to(torch.int8).to(self.device)
        _ = custom_comm.allgather_batch([static_data], self.hcom, self.world_size)
        torch.npu.synchronize()

        stream_, graph, out = self._manual_capture_graph(
            lambda: custom_comm.allgather_batch([static_data], self.hcom, self.world_size)
        )

        for step in range(10):
            real_data = (torch.arange(32) + self.rank * 32 + step).to(torch.int8).to(self.device)
            static_data.copy_(real_data)
            self._manual_replay(stream_, graph)

            expected = self._gather_expected(
                [torch.arange(32) + r * 32 + step for r in range(self.world_size)],
                torch.int8,
            )
            assert torch.equal(out[0], expected)

        assert out[0].shape[0] == static_data.shape[0] * self.world_size

    def test_aclgraph_reset_then_recapture(self):
        """reset() should release the old graph so a different-op recapture can proceed."""
        import custom_comm
        static_old = torch.arange(48, dtype=torch.float32, device=self.device) + self.rank * 3
        dist.broadcast(static_old, src=0)
        torch.npu.synchronize()

        stream_, graph_old, _ = self._manual_capture_graph(
            lambda: (dist.broadcast(static_old, src=0), static_old)[1]
        )
        graph_old.reset()
        torch.npu.synchronize()

        static_new = (
            torch.arange(96, dtype=torch.int64) + self.rank * 2
        ).to(torch.int8).to(self.device)
        _ = custom_comm.allgather_batch([static_new], self.hcom, self.world_size)
        torch.npu.synchronize()

        stream_new, graph_new, out_new = self._manual_capture_graph(
            lambda: custom_comm.allgather_batch([static_new], self.hcom, self.world_size),
            stream_=stream_,
        )

        real_new = (
            torch.arange(96, dtype=torch.int64) + self.rank * 2 + 21
        ).to(torch.int8).to(self.device)
        static_new.copy_(real_new)
        self._manual_replay(stream_new, graph_new)

        expected_new = self._gather_expected(
            [torch.arange(96, dtype=torch.int64) + r * 2 + 21 for r in range(self.world_size)],
            torch.int8,
        )
        assert torch.equal(out_new[0], expected_new)

    def test_aclgraph_mixed_ops_single_graph(self):
        """One graph may contain custom_comm plus a different built-in HCCL op in sequence."""
        import custom_comm

        static_a = (
            torch.arange(64, dtype=torch.int64) + self.rank * 64
        ).to(torch.int8).to(self.device)
        static_b = torch.arange(32, dtype=torch.float32, device=self.device) + self.rank * 20

        _ = custom_comm.allgather_batch([static_a], self.hcom, self.world_size)
        dist.broadcast(static_b, src=0)
        torch.npu.synchronize()

        def capture_body():
            out_a = custom_comm.allgather_batch([static_a], self.hcom, self.world_size)
            dist.broadcast(static_b, src=0)
            return out_a

        stream_, graph, out_a = self._manual_capture_graph(capture_body)

        real_a = (
            torch.arange(64, dtype=torch.int64) + self.rank * 64 + 31
        ).to(torch.int8).to(self.device)
        real_b = torch.arange(32, dtype=torch.float32, device=self.device) + self.rank * 20 + 5

        static_a.copy_(real_a)
        static_b.copy_(real_b)

        self._manual_replay(stream_, graph)

        expected_a = self._gather_expected(
            [torch.arange(64, dtype=torch.int64) + r * 64 + 31 for r in range(self.world_size)],
            torch.int8,
        )
        expected_b = torch.arange(32, dtype=torch.float32, device=self.device) + 5

        assert torch.equal(out_a[0], expected_a)
        assert torch.equal(static_b, expected_b)


