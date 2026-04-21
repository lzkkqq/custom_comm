// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Engine context for the CCU MS backend (skeleton).
//
// Parallels ccu_sched/engine_ctx.cc: registers MS kernel on first use,
// and dispatches launches.  Full implementation will arrive with the MS
// kernel.  For now everything returns HCCL_E_NOT_SUPPORT so downstream
// callers fall back cleanly.

#include "ccu_ms/engine_ctx_ms.h"

#include "common.h"
#include "log_util.h"

#include <hccl/hccl_types.h>

namespace custom_comm {
namespace ms {

HcclResult InitCcuContext(HcclComm comm) {
    (void)comm;
    CC_LOG_ERROR("ccu_ms::InitCcuContext: not implemented yet");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult LaunchCcuKernel(HcclComm comm, const void *taskArg) {
    (void)comm; (void)taskArg;
    CC_LOG_ERROR("ccu_ms::LaunchCcuKernel: not implemented yet");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult GetCcuThreadHandle(HcclComm comm, uint64_t *threadHandle) {
    (void)comm; (void)threadHandle;
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ms
}  // namespace custom_comm
