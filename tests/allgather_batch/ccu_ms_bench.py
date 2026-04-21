#!/usr/bin/env python3
# Copyright (c) 2026 custom_comm authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: compare AllGather strategies for allgather_batch.

All methods run under HCCL op-expansion-mode=CCU_MS (mode 5 on Atlas A5).
CCU hardware resources can only host one active comm at a time, so the
process group is torn down between methods and re-created for the next.

    torchrun --nproc_per_node=8 tests/allgather_batch/bench.py
    torchrun --nproc_per_node=8 tests/allgather_batch/bench.py --expansion-mode NONE
"""
import argparse
import os

import torch
import torch.distributed as dist

HAS_NPU = False
try:
    import torch_npu, custom_comm, custom_comm._C as _C  # noqa: F401,E401
    HAS_NPU = True
except ImportError:
    pass

WARMUP, ITERS = 50, 200

# Mirrors smoke_test.py — HCCL_OP_EXPANSION_MODE env var is ignored by
# torch_npu's backend; the only reliable knob is a per-comm HcclCommConfig,
# set via ProcessGroupHCCL.Options().hccl_config.
_EXPANSION_MODE_MAP = {
    "DEFAULT":    0,
    "HOSTCPU_TS": 1,
    "AICPU_TS":   2,
    "AIV":        3,
    "AIV_ONLY":   4,
    "CCU_MS":     5,
    "CCU_SCHED":  6,
    "AICPU_UB":   7,
}


def build_hccl_options(expansion_mode_name):
    if expansion_mode_name is None or expansion_mode_name == "NONE":
        return None
    opt = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    opt.hccl_config = {
        "hccl_op_expansion_mode": int(_EXPANSION_MODE_MAP[expansion_mode_name])
    }
    return opt


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    t0 = torch.npu.Event(enable_timing=True)
    t1 = torch.npu.Event(enable_timing=True)
    t0.record()
    for _ in range(ITERS):
        fn()
    t1.record()
    torch.npu.synchronize()
    return t0.elapsed_time(t1) * 1000 / ITERS


def init_pg(expansion_mode):
    os.environ.pop("HCCL_OP_EXPANSION_MODE", None)
    dist.init_process_group(
        backend="hccl", pg_options=build_hccl_options(expansion_mode)
    )
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.npu.set_device(rank)
    dev = torch.device(f"npu:{rank}")
    # Prefer the raw HcclComm handle (int64) exposed via get_hccl_comm():
    # stringified, our C++ side parses it as a pointer value and bypasses
    # HcomGetCommHandleByGroup, which on this torch_npu version returns a
    # comm whose internal collComm is nullptr (→ HcclGetRankSize HCCL_E_PTR).
    backend = dist.distributed_c10d._get_default_group()._get_backend(dev)
    if hasattr(backend, "get_hccl_comm"):
        hcom = str(backend.get_hccl_comm(rank))
    else:
        hcom = backend.get_hccl_comm_name(rank)
    return rank, ws, dev, hcom


def destroy_pg():
    dist.destroy_process_group()
    # Flush any pending NPU work so the CCU comm is fully released
    # before the next method initialises a fresh one on the same resource.
    torch.npu.synchronize()


def bench_method(build_fn, expansion_mode):
    """init PG → build callable → warmup+time → destroy PG."""
    rank, ws, dev, hcom = init_pg(expansion_mode)
    fn = build_fn(rank, ws, dev, hcom)
    us = timed(fn)
    destroy_pg()
    return rank, ws, us


def make_inputs(dev):
    N, H, K = 32, 7168, 8
    x   = torch.randint(0, 127, (N, H), dtype=torch.int8,    device=dev)
    s   = torch.randn(N,                 dtype=torch.float32, device=dev)
    ids = torch.randint(0, 8,   (N, K),  dtype=torch.int32,   device=dev)
    return [x, s, ids], N


def main():
    if not HAS_NPU:
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--expansion-mode", type=str, default="CCU_MS",
                    choices=list(_EXPANSION_MODE_MAP.keys()) + ["NONE"],
                    help="HCCL op expansion mode applied to every method.")
    args = ap.parse_args()
    mode = args.expansion_mode

    def build_a(rank, ws, dev, hcom):
        tensors, _ = make_inputs(dev)
        def fn():
            for t in tensors:
                dist.all_gather([torch.empty_like(t) for _ in range(ws)], t)
        return fn

    def build_b(rank, ws, dev, hcom):
        tensors, N = make_inputs(dev)
        def fn():
            for t in tensors:
                dist.all_gather_into_tensor(
                    torch.empty(N * ws, *t.shape[1:], dtype=t.dtype, device=dev), t)
        return fn

    def build_c(rank, ws, dev, hcom):
        tensors, N = make_inputs(dev)
        def fn():
            u8 = [t.reshape(N, -1).contiguous().view(torch.uint8) for t in tensors]
            packed = torch.cat(u8, dim=1)
            out = torch.empty(N * ws, packed.shape[1], dtype=torch.uint8, device=dev)
            dist.all_gather_into_tensor(out, packed)
            col = 0
            for t in tensors:
                bw = t.nbytes // N
                out[:, col:col + bw].contiguous().view(t.dtype).reshape(
                    [N * ws] + list(t.shape[1:]))
                col += bw
        return fn

    def build_d(rank, ws, dev, hcom):
        tensors, _ = make_inputs(dev)
        def fn():
            torch.ops.custom_comm.allgather_batch(tensors, hcom, ws)
        return fn

    def build_e(rank, ws, dev, hcom):
        tensors, _ = make_inputs(dev)
        def fn():
            _C.allgather_batch_eager(tensors, hcom, ws)
        return fn

    def build_f(rank, ws, dev, hcom):
        tensors, N = make_inputs(dev)
        outs = [torch.empty(N * ws, *t.shape[1:], dtype=t.dtype, device=dev)
                for t in tensors]
        def fn():
            _C.allgather_batch_inplace(tensors, outs, hcom, ws)
        return fn

    methods = [
        ("A) 3 all_gather(list),ds3",       build_a),
        ("B) 3 all_gather_into_tensor,pg2", build_b),
        ("C) 1 ag_packed,pure py",          build_c),
        ("D) 1 agb,torch.ops(Dispatcher)",  build_d),
        ("E) 1 agb,pybind11(eager)",        build_e),
        ("F) 1 agb,pybind11(in-place)",     build_f),
    ]

    rank = ws = None
    results = []
    for label, build in methods:
        rank, ws, us = bench_method(build, mode)
        results.append((label, us))

    if rank == 0:
        W = 36
        mx = max(v for _, v in results)
        print(f"\nOPT-AG-04 Benchmark  W={ws}  N=32  mode={mode}")
        print("-" * 60)
        for label, us in results:
            bar = "\u2588" * int(us / mx * 30)
            print(f"  {label:<{W}} {us:8.1f} us  {bar}")
        print()
        ta, tb, _tc, td, te, tf = [v for _, v in results]
        print(f"  me vs ds:  {ta/tf:.2f}x  (saved {ta - tf:.0f} us)")
        print(f"  me vs pg:  {tb/tf:.2f}x  (saved {tb - tf:.0f} us)")
        print(f"  Dispatcher:   {td - te:+.1f} us")
        print(f"  ReturnValue:  {te - tf:+.1f} us")


if __name__ == "__main__":
    main()
