# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""custom_comm: Custom communication operators for Ascend NPU."""

import ctypes
import os

import torch

# Preload the ABI=0 shim (libcustom_comm_impl.so) before loading the torch
# extension. _C.so is linked with RPATH=$ORIGIN so this is belt-and-suspenders.
# It also surfaces any shim load errors clearly rather than hiding them behind
# a lazy dlopen during a later custom op call.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_shim_path = os.path.join(_pkg_dir, "libcustom_comm_impl.so")
if os.path.exists(_shim_path):
    ctypes.CDLL(_shim_path, mode=ctypes.RTLD_GLOBAL)

# Load the C++ extension (ABI=1, torch-matched) that registers
# torch.ops.custom_comm.*.
try:
    import custom_comm._C  # noqa: F401
except ImportError:
    pass  # Allow metadata-only install without torch_npu.

from custom_comm.ops import allgather_batch  # noqa: E402

# Graph mode converter registration (torchair is an optional dep).
try:
    import custom_comm.converters  # noqa: F401
except ImportError:
    pass

__all__ = ["allgather_batch"]
