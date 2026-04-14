# custom_comm

custom_comm is a library of high-performance custom communication operators for
Ascend NPUs. It provides fused collective primitives on top of HCCL/HComm,
registered as PyTorch custom ops via `torch_npu`, with support for eager mode,
`torch.compile` (torchair GE graph mode), and aclGraph capture.

The first operator is `allgather_batch` -- a batched AllGather that gathers up
to 8 heterogeneous-dtype tensors in a single collective call, eliminating
per-tensor kernel launch overhead. The primary use case is MoE quantized
inference, where INT8 activations, FP32 scales, and INT32 routing indices need
to be gathered together.

## Prerequisites

custom_comm requires the following:

- Ascend NPU with CANN 9.0+ toolkit (Atlas A5 / Ascend 950)
- Python 3.10+
- PyTorch >= 2.6
- torch_npu >= 2.6

## Installation

### From source (recommended)

```bash
# Ensure CANN environment is set up
source ~/Ascend/set_env.sh

# Install in development mode
pip install -e .
```

### Build C++ library only

For integration without Python bindings:

```bash
cmake -B build
cmake --build build
```

## Quick Start

```python
import torch
import torch_npu
import custom_comm

torch.distributed.init_process_group(backend="hccl")
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.npu.set_device(rank)

# Obtain HCCL communicator handle
pg = torch.distributed.group.WORLD
hcom = pg._get_backend(torch.device(f"npu:{rank}")).get_hccl_comm_name(rank)

# Batched AllGather: INT8 data + FP32 scales + INT32 topk_ids in one call
data = torch.randint(0, 127, (2048, 4096), dtype=torch.int8, device="npu")
scales = torch.randn(2048, dtype=torch.float32, device="npu")
ids = torch.randint(0, 8, (2048,), dtype=torch.int32, device="npu")

results = custom_comm.allgather_batch([data, scales, ids], hcom, world_size)
# results[0]: (2048 * world_size, 4096) int8
# results[1]: (2048 * world_size,)      float32
# results[2]: (2048 * world_size,)      int32
```

### Running

```bash
torchrun --nproc_per_node=8 your_script.py
```

## Execution Modes

custom_comm supports two runtime strategies for the batched AllGather operation,
selected via the `CUSTOM_COMM_USE_CCU` environment variable:

### Phase 1: Decomposed (default)

Packs all input tensors into a single contiguous buffer (byte-level view), performs
one `HcclAllGather` call, then unpacks the results. This path works on all CANN
versions and does not require CCU hardware scheduling.

### Phase 2: CCU Batched

Registers a custom CCU kernel that performs zero-copy RDMA gathers directly between
each descriptor's send/recv buffers. Eliminates pack/unpack overhead and reduces
HBM traffic. Requires CANN 9.0+ with HComm CCU support.

```bash
# Enable CCU path
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 your_script.py
```

## Graph Mode

### GE Graph (torchair)

custom_comm registers a torchair GE converter that decomposes `allgather_batch`
into multiple `HcomAllGather` GE nodes. This preserves graph-level optimizations
while providing correct semantics:

```python
@torch.compile(backend="npu")
def step(x, s, ids):
    return custom_comm.allgather_batch([x, s, ids], hcom, world_size)
```

### aclGraph Capture

When the main stream is in aclGraph capture mode, `allgather_batch` automatically
detects the capture state and registers the CCU slave stream into the graph model,
so that Phase 2 CCU kernel operations are captured correctly.

## Testing

```bash
# Shape inference tests (no NPU required)
pytest tests/ -k "meta"

# NPU functional tests
torchrun --nproc_per_node=8 pytest tests/test_allgather_batch.py

# Performance benchmarks
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py --ag09
```

## License

Apache-2.0
