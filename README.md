# custom_comm

High-performance custom collective communication operators for Ascend NPUs.

custom_comm extends HCCL with fused communication primitives that are not
available in the standard library. It exposes both a C API (for integration
into runtimes) and a PyTorch custom-op interface (for direct use in Python
training/inference scripts), with support for eager mode, `torch.compile`,
and graph mode via torchair.

## Operators

| Operator | Description |
|---|---|
| `allgather_batch` | Gather up to 8 heterogeneous-dtype tensors in a single operation, avoiding per-tensor launch overhead |

## Prerequisites

- CANN 9.0.0 SDK (Atlas 800I A2 / Ascend 910B)
- Python 3.8+
- PyTorch 2.1+ with torch_npu

## Installation

```bash
# Ensure CANN environment is sourced
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Install from source
pip install -e .
```

For C++ library only (no Python bindings):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Quick Start

```python
import torch
import torch_npu
import custom_comm

# Standard distributed setup
torch.distributed.init_process_group(backend="hccl")
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.npu.set_device(rank)

# Obtain HCCL communicator name
pg = torch.distributed.group.WORLD
backend = pg._get_backend(torch.device(f"npu:{rank}"))
hcom = backend.get_hccl_comm_name(rank)

# ---- Core use case: batched quantized AllGather ----
# Gather INT8 activations and FP32 scales in a single operation,
# instead of issuing two separate AllGather calls.
activations = torch.randint(0, 127, (2048, 4096), dtype=torch.int8, device="npu")
scales = torch.randn(2048, dtype=torch.float32, device="npu")

gathered = custom_comm.allgather_batch(
    [activations, scales], hcom, world_size
)
# gathered[0]: (2048 * world_size, 4096), int8
# gathered[1]: (2048 * world_size,),       float32
```

### Running

```bash
torchrun --nproc_per_node=8 your_script.py
```

## Execution Paths

custom_comm supports two runtime strategies, selected via environment variable:

| `CUSTOM_COMM_USE_CCU` | Strategy | Description |
|:---:|---|---|
| unset / `0` | Phase 1 (Decomposed) | Packs heterogeneous tensors into a flat byte buffer, calls `HcclAllGather` once, then unpacks. Works on all CANN versions. |
| `1` | Phase 2 (CCU Kernel) | Registers a single CCU kernel that performs multi-descriptor RDMA gather directly. Zero-copy, lower latency. Requires CANN 9.0+ with HComm CCU support. |

## Graph Mode (torchair)

custom_comm registers a GE converter so `allgather_batch` works inside
`torch.compile` / torchair traced graphs:

```python
import torch
import custom_comm

@torch.compile(backend="npu")
def step(x, s, hcom, world_size):
    return custom_comm.allgather_batch([x, s], hcom, world_size)
```

The converter decomposes the batched op into N individual `HcomAllGather` GE
nodes (GE does not yet have a native batched gather op). This preserves other
graph-level optimizations while maintaining functional correctness.

## Benchmarking

```bash
# Run on 8 NPUs, compare Phase 1 vs Phase 2
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
```

## Testing

```bash
# Shape-inference tests (no NPU needed, runs on macOS/Linux)
pytest tests/ -k "meta"

# Full functional tests (requires NPU + HCCL)
torchrun --nproc_per_node=2 -m pytest tests/test_allgather_batch.py -v

# Graph-mode tests
pytest tests/test_graph_mode.py -v
```

## Project Layout

```
CMakeLists.txt              CMake build (C++ library)
setup.py                    Python package (torch extension)
cmake/FindCANN.cmake        CANN SDK discovery
ops/
  allgather_batch/
    inc/                    C/C++ headers (public C API + internals)
    src/                    Implementation (dispatch, decomposed, CCU kernel)
python/
  custom_comm/
    __init__.py             Package init, loads C extension
    ops.py                  torch.ops wrapper
    converters/             torchair GE graph-mode converters
tests/                      Unit tests + benchmarks
docs/
  design/                   Architecture diagrams (PlantUML, d2)
  raw/                      Design documents and analysis
```

## License

Apache-2.0
