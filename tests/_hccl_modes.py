# Copyright (c) 2026 custom_comm authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared HCCL op-expansion-mode table and ProcessGroupHCCL.Options factory.

torch_npu ignores the `HCCL_OP_EXPANSION_MODE` env var when building its
per-PG HcclCommConfig; the only reliable knob is the `hccl_op_expansion_mode`
key under `ProcessGroupHCCL.Options.hccl_config`.  This module keeps the
name-to-int table and the options builder in one place so conftest, the
smoke test, and ccu_ms_bench don't each hard-code the value.

Values 0..7 mirror HCCL's internal `HcclAccelerator` enum, defined at
    hcomm/src/legacy/common/types/types.h:30
        MAKE_ENUM(HcclAccelerator, DEFAULT, HOSTCPU_TS, AICPU_TS, AIV,
                                   AIV_ONLY, CCU_MS, CCU_SCHED, AICPU)
HCCL's env-var parser (`CastHcclAccelerator` at
hcomm/src/legacy/framework/env_config/env_func.cc:616) only accepts a subset
of these as strings: AI_CPU / AIV / AIV_ONLY / HOST / HOST_TS / CCU_MS /
CCU_SCHED. We skip that string layer — `hccl_config["hccl_op_expansion_mode"]`
takes the raw integer directly.
"""
from __future__ import annotations

import os

ENV_VAR = "HCCL_OP_EXPANSION_MODE"

EXPANSION_MODE = {
    "DEFAULT":    0,
    "HOSTCPU_TS": 1,
    "AICPU_TS":   2,
    "AIV":        3,
    "AIV_ONLY":   4,
    "CCU_MS":     5,
    "CCU_SCHED":  6,
    "AICPU":      7,
}


def _resolve(mode):
    """Return integer op_expansion_mode, or None to skip the override."""
    if mode is None or mode == "" or mode == "NONE":
        return None
    if isinstance(mode, str):
        return int(mode) if mode.lstrip("-").isdigit() else EXPANSION_MODE[mode]
    return int(mode)


def build_hccl_options(mode):
    """Return ProcessGroupHCCL.Options with hccl_op_expansion_mode set.

    `mode` accepts a name ("CCU_MS"), an int (5), a digit string ("5"), or
    None/"NONE" to skip the override.  Returns None when no override should
    be applied or when torch_npu isn't importable (lets conftest load in
    pure-host environments).
    """
    value = _resolve(mode)
    if value is None:
        return None
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return None
    opt = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    opt.hccl_config = {"hccl_op_expansion_mode": value}
    return opt


def build_hccl_options_from_env(default="CCU_MS"):
    """Like `build_hccl_options`, but mode comes from the HCCL_OP_EXPANSION_MODE env var.

    If the env var is unset or empty, fall back to `default`.  The env var
    can be a mode name (e.g. "CCU_MS") or an integer literal (e.g. "5").
    """
    return build_hccl_options(os.environ.get(ENV_VAR) or default)
