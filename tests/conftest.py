# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures and capability gating for custom_comm tests.

Markers declared here (auto-skipped when the capability is missing):
    npu       Ascend NPU runtime (torch_npu + torch.npu.is_available())
    ext       compiled custom_comm C extension importable
    torchair  torchair graph-mode backend installed
    dist      launched under torchrun (RANK / WORLD_SIZE set)

Run locally (meta only):
    pytest tests/

Run on a multi-NPU host:
    torchrun --nproc_per_node=N -m pytest tests/
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest


def _has(mod: str) -> bool:
    # Use broad Exception (not just ImportError): some packages — notably
    # torch_npu.dynamo.torchair — raise custom error types during import when
    # their runtime deps are missing, and those don't subclass ImportError.
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def _npu_ready() -> bool:
    if not _has("torch_npu"):
        return False
    try:
        import torch
        return hasattr(torch, "npu") and bool(torch.npu.is_available())
    except Exception:
        return False


CAPS = SimpleNamespace(
    npu=_npu_ready(),
    ext=_has("custom_comm"),
    torchair=_has("torchair"),
    dist="RANK" in os.environ and "WORLD_SIZE" in os.environ,
)

_MARKER_REASONS = {
    "npu":      "NPU unavailable (torch_npu missing or no device)",
    "ext":      "custom_comm extension not built / not importable",
    "torchair": "torchair not installed",
    "dist":     "not launched via torchrun (RANK / WORLD_SIZE unset)",
}


def pytest_configure(config):
    for name, reason in _MARKER_REASONS.items():
        config.addinivalue_line("markers", f"{name}: {reason}")


def pytest_collection_modifyitems(config, items):
    for item in items:
        for name, reason in _MARKER_REASONS.items():
            if name in item.keywords and not getattr(CAPS, name):
                item.add_marker(pytest.mark.skip(reason=reason))
                break


def _ccu_ms_pg_options():
    """Build ProcessGroupHCCL.Options forcing op-expansion-mode=CCU_MS (5).

    Without this, HCCL defaults the per-comm expansion mode to SCHED, which
    eagerly pre-allocates die-local block resources (blockMsReq / blockCke /
    blockLoopEngine). Later, when our V2 kernel calls HcclCcuKernelRegister,
    the CCU resource manager sees die1 already saturated and returns
    HCCL error 7 (CheckResIfAvailable). Declaring CCU_MS up front reserves
    the right pool shape from the start and matches ccu_ms_bench.py.
    """
    try:
        import torch_npu  # noqa: F401
        opt = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        opt.hccl_config = {"hccl_op_expansion_mode": 5}  # CCU_MS
        return opt
    except Exception:
        # torch_npu missing or API shape changed: fall back to default init.
        return None


@pytest.fixture(scope="session")
def dist_ctx():
    """Session-scoped HCCL context: rank, world_size, device, hcom."""
    if not (CAPS.npu and CAPS.dist):
        pytest.skip("dist_ctx requires NPU + torchrun")

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    if not dist.is_initialized():
        dist.init_process_group(backend="hccl", pg_options=_ccu_ms_pg_options())

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    pg = dist.distributed_c10d._get_default_group()
    backend = pg._get_backend(device)
    # Prefer the raw HcclComm handle (int64) exposed via get_hccl_comm() when
    # available; fall back to the legacy name-based API otherwise.
    if hasattr(backend, "get_hccl_comm"):
        hcom = str(backend.get_hccl_comm(rank))
    else:
        hcom = backend.get_hccl_comm_name(rank)

    # Diagnostic: dump what torch_npu hands us so we can compare against
    # what HcomGetCommHandleByGroup expects.
    import sys
    print(
        f"[dist_ctx rank={rank}] hcom={hcom!r} type={type(hcom).__name__} "
        f"backend={type(backend).__name__} "
        f"get_hccl_comm={getattr(backend, 'get_hccl_comm', None)}",
        file=sys.stderr, flush=True,
    )

    try:
        yield SimpleNamespace(
            rank=rank, world_size=world_size, device=device, hcom=hcom,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
