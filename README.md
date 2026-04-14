# custom_comm

High-performance custom communication operators for Ascend NPUs, built on top of
HCCL and the CANN software stack.

custom_comm provides fused, batched collective operations that are not available
in the standard HCCL library. It integrates with PyTorch via `torch_npu` custom
ops, supports both eager-mode and graph-mode (via torchair/aclgraph), and
exposes a C API for direct integration with HCCL-based applications.

## Operators

| Operator | Description |
|:---|:---|
| `allgather_batch` | Batched AllGather that gathers up to 8 heterogeneous-dtype tensors in a single call, avoiding per-tensor launch overhead |

## Prerequisites

- Python 3.8+
- CANN 9.0.0 SDK
- PyTorch 2.1+ with torch_npu
- Ascend NPU (Atlas 800T A2 or equivalent)
- CMake 3.14+

## Installation

### From source (on Ascend host)

```bash
# Ensure CANN environment is sourced
source /usr/local/Ascend/ascendc-toolkit/latest/set_env.sh

pip install -e .
```

### Build C++ library only (for integration without Python)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Quick start

```python
import torch
import torch_npu
import custom_comm

# Initialize distributed environment
torch.distributed.init_process_group(backend="hccl")

# Quantized activations (INT8) + scales (FP32) -- a common pattern in
# quantized AllGather where data and metadata have different dtypes.
activations = torch.randn(1024, 4096, dtype=torch.int8, device="npu")
scales = torch.randn(1024, dtype=torch.float32, device="npu")

# Single batched AllGather -- one kernel launch for both tensors
results = custom_comm.allgather_batch(
    [activations, scales],
    hcom=hcom_group_name,
    world_size=world_size,
)
# results[0].shape == (1024 * world_size, 4096), dtype=int8
# results[1].shape == (1024 * world_size,),      dtype=float32
```

## Execution modes

| Mode | Env var | Description |
|------|---------|-------------|
| Phase 1 (default) | -- | Decomposed: packs tensors into a flat buffer, calls HcclAllGather once, unpacks |
| Phase 2 (CCU) | `CUSTOM_COMM_USE_CCU=1` | Direct CCU kernel: zero-copy RDMA gather per descriptor, no pack/unpack |

## Graph mode (torchair / aclgraph)

`custom_comm` registers a GE converter so the operator can run inside
`torch.compile(backend="npu")` graphs.  The converter decomposes the batched
call into N individual `HcomAllGather` nodes (GE has no native batched op).

```python
import torch
import custom_comm

@torch.compile(backend="npu")
def my_fn(x, s):
    return custom_comm.allgather_batch([x, s], "hcom_group", world_size=8)
```

## Testing

```bash
# Meta-dispatch tests (runs anywhere, no NPU required)
pytest tests/ -k "not npu"

# Full NPU tests (requires Ascend device + HCCL init)
pytest tests/
```

## Project Structure

```
custom_comm/
  CMakeLists.txt                 # Build system
  setup.py                       # pip install entry
  cmake/FindCANN.cmake           # CANN SDK discovery
  ops/
    allgather_batch/
      inc/                       # C headers (C API, engine, common defs)
      src/                       # C++ implementation
  python/custom_comm/            # Python package
    __init__.py                  # torch custom op loading
    converters/                  # GE graph-mode converters
  tests/                         # pytest suite
  docs/design/                   # Architecture diagrams (PlantUML, D2)
```

## License

Apache-2.0
