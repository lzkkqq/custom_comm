#!/usr/bin/env python3
"""Smoke test: verify NPU + HCCL environment is functional.

AllGather runs on a dedicated sub-group carrying --expansion-mode (the
custom_comm target path, default CCU_MS).  AllReduce / Broadcast run on the
default group carrying --fallback-mode (default AICPU_TS), because CCU_MS on
2-card mis-reduces small tensors (HCCL chunk-granularity quirk — see
benchmark_ultimate.py:445-459 for the AllGather-side analogue).

Usage:
    torchrun --nproc_per_node=2 tests/smoke_test.py
    torchrun --nproc_per_node=2 tests/smoke_test.py --expansion-mode CCU_MS
    torchrun --nproc_per_node=2 tests/smoke_test.py \\
        --expansion-mode CCU_MS --fallback-mode AICPU_TS
"""
import argparse
import os

import torch
import torch.distributed as dist
import torch_npu  # registers torch.npu; also exposes ProcessGroupHCCL.Options

from _hccl_modes import EXPANSION_MODE as _EXPANSION_MODE_MAP, build_hccl_options


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expansion-mode", type=str, default="CCU_MS",
                    choices=list(_EXPANSION_MODE_MAP.keys()) + ["NONE"],
                    help="Mode for the AllGather sub-group — the path "
                         "custom_comm actually cares about.")
    ap.add_argument("--fallback-mode", type=str, default="AICPU_TS",
                    choices=list(_EXPANSION_MODE_MAP.keys()) + ["NONE"],
                    help="Mode for the default group; AllReduce / Broadcast "
                         "run here. Kept off CCU_MS because small-tensor "
                         "AllReduce under CCU_MS + 2-card mis-reduces the "
                         "tail chunk.")
    args = ap.parse_args()

    # The env var is ignored by torch_npu anyway — pop it so nothing
    # downstream mistakes it for the active mode.
    os.environ.pop("HCCL_OP_EXPANSION_MODE", None)

    # Default group carries --fallback-mode. Keeping it off CCU_MS matters for
    # two reasons: (1) only one CCU_MS comm can exist at a time on 2-card
    # configs (see benchmark_ultimate.py:420-422); (2) small-tensor AllReduce
    # on CCU_MS mis-reduces the tail chunk.
    dist.init_process_group(
        "hccl", pg_options=build_hccl_options(args.fallback_mode)
    )
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.npu.set_device(rank)

    # AllGather sub-group — carries the user's target mode (e.g. CCU_MS).
    # HCCL requires every rank to call new_group(), even when NONE.
    if args.expansion_mode == "NONE":
        ag_pg = None
    else:
        ag_pg = dist.new_group(
            ranks=list(range(ws)),
            pg_options=build_hccl_options(args.expansion_mode),
        )

    # 1. AllGather on the sub-group — use all_gather_into_tensor (the
    # list-based dist.all_gather goes through a flat-buffer -> scatter step
    # that races with the CCU stream and silently returns pre-op zeros
    # under CCU_MS; sync after the call so the readback isn't racy).
    x = torch.tensor([rank], dtype=torch.float32, device=f"npu:{rank}")
    out_flat = torch.zeros(ws, dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out_flat, x, group=ag_pg)
    torch.npu.synchronize()
    actual = [int(v) for v in out_flat.tolist()]
    expected = list(range(ws))
    assert actual == expected, f"AllGather failed: {actual} != {expected}"

    # 2. AllReduce
    y = torch.ones(4, device=f"npu:{rank}") * (rank + 1)
    dist.all_reduce(y)
    assert y.eq(sum(range(1, ws + 1))).all(), f"AllReduce failed: {y}"

    # 3. Broadcast
    z = torch.tensor([42.0], device=f"npu:{rank}") if rank == 0 \
        else torch.zeros(1, device=f"npu:{rank}")
    dist.broadcast(z, src=0)
    assert z.item() == 42.0, f"Broadcast failed: {z}"

    # Flush pending CCU work before the sub-group's comm is destroyed;
    # CCU_MS otherwise leaks MS / CKE / LoopEngine handles.
    if ag_pg is not None:
        torch.npu.synchronize()
        dist.destroy_process_group(ag_pg)

    if rank == 0:
        print(f"PASS  W={ws}  "
              f"allgather={args.expansion_mode}  "
              f"allreduce/broadcast={args.fallback_mode}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
