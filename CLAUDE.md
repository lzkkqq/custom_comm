# CLAUDE.md

This file provides guidance for Claude Code when working with this repository.

## Project Overview

custom_comm is a custom communication operator library for Ascend NPU (CANN ecosystem).
It delivers high-performance batched collective operations as PyTorch custom ops, with
both eager-mode and graph-mode (torchair/aclgraph) support on Atlas 800T A2 (Ascend 910B).

The first operator is `allgather_batch` -- a fused, batched AllGather that gathers
up to 8 heterogeneous tensors in a single collective call, avoiding per-tensor launch
overhead.  Two execution paths exist:

- Phase 1 (decomposed): packs tensors into a contiguous buffer, calls HcclAllGather once, unpacks.
- Phase 2 (CCU): programs the CCU directly via HComm primitives for zero-copy RDMA gather.

## Repository Layout

    CMakeLists.txt              # C++ build (syntax-check on macOS, full build on NPU host)
    cmake/FindCANN.cmake        # SDK discovery (hccl, acl, hcomm headers & libs)
    setup.py                    # pip install -e . (calls torch.utils.cpp_extension)
    ops/
      allgather_batch/
        inc/                    # C/C++ headers (.h)
        src/                    # C/C++ sources (.cc)
    python/
      custom_comm/
        __init__.py             # torch.ops.load_library + Python API
        ops.py                  # torch.autograd.Function wrappers
        converters/             # torchair GE graph-mode converters
    tests/
      test_allgather_batch.py   # unit tests (meta + NPU)
      test_graph_mode.py        # torchair graph-mode tests
    docs/
      design/                   # Architecture diagrams (PlantUML, d2)
      raw/                      # Design docs, analysis, reviews

## Build & Install

### Prerequisites

- CANN 9.0.0 SDK (`ASCEND_CANN_PACKAGE_PATH` or default `/usr/local/Ascend/ascend-toolkit/latest`)
- PyTorch 2.1+ with torch_npu
- Python 3.8+

### Source install (on NPU host)

    pip install -e .

### Syntax-check only (macOS, no CANN runtime)

    cmake -B build && cmake --build build

## Testing

    # Meta-device tests (no NPU needed):
    pytest tests/ -k "Meta"

    # NPU functional tests (single node, N devices):
    torchrun --nproc_per_node=N pytest tests/test_allgather_batch.py

## Architecture

Two execution phases, selected at runtime via `CUSTOM_COMM_USE_CCU` env var:

- Phase 1 (default): Decomposed strategy -- packs heterogeneous descriptors into a
  contiguous buffer, calls `HcclAllGather` once, then unpacks.  Works on any CANN version.
- Phase 2 (CCU):  Registers a CCU kernel via `HcclRegisterCustomKernel`, launches a
  single CCU program that performs direct RDMA gather per descriptor.  Requires HComm
  CCU API (CANN 9.0+).

### Key interfaces

- C API: `ops/allgather_batch/inc/hccl_custom_allgather_batch.h` -- `HcclAllGatherBatch()`
- Python: `custom_comm.allgather_batch(inputs, hcom, world_size)` -- torch custom op
- Graph mode: `python/custom_comm/converters/` -- torchair FX-to-GE converter

## Code Style

- C++17, formatted by `.clang-format` (4-space indent, ~100 col limit)
- Python: standard PEP 8, no additional formatter enforced
- License header: `Copyright (c) 2026 custom_comm Authors. SPDX-License-Identifier: Apache-2.0`
