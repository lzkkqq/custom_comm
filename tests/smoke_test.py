#!/usr/bin/env python3
"""Smoke test: verify NPU + HCCL environment is functional.

Usage:
    torchrun --nproc_per_node=2 tests/smoke_test.py
"""
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.npu.set_device(rank)

    # 1. AllGather
    x = torch.tensor([rank], dtype=torch.float32, device=f"npu:{rank}")
    out = [torch.zeros(1, device=x.device) for _ in range(ws)]
    dist.all_gather(out, x)
    expected = list(range(ws))
    actual = [int(t.item()) for t in out]
    assert actual == expected, f"AllGather failed: {actual} != {expected}"

    # 2. AllReduce
    y = torch.ones(4, device=f"npu:{rank}") * (rank + 1)
    dist.all_reduce(y)
    assert y.eq(sum(range(1, ws + 1))).all(), f"AllReduce failed: {y}"

    # 3. Broadcast
    z = torch.tensor([42.0], device=f"npu:{rank}") if rank == 0 else torch.zeros(1, device=f"npu:{rank}")
    dist.broadcast(z, src=0)
    assert z.item() == 42.0, f"Broadcast failed: {z}"

    if rank == 0:
        print(f"PASS  W={ws}  allgather/allreduce/broadcast")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
