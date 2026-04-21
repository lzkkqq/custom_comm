// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Engine context for the MS (LoopGroup MultiSlot) CCU backend.
//
// Mirror of ccu_sched/engine_ctx.cc: manages the per-comm channel/kernel
// cache that InitCcuContext fills and LaunchCcuKernel consumes. Uses a
// distinct CTX tag so both CCU_MODEs may co-exist on the same comm.
//
// Present impl is a placeholder — returns HCCL_E_NOT_SUPPORT from every
// entry point until the MS kernel body (Algorithm) is implemented. The
// dispatcher in ccu_dispatch.cc routes to these stubs only when the user
// sets CUSTOM_COMM_CCU_MODE=ms; SCHED remains the default path.

#include "ccu_ms/engine_ctx_ms.h"

#include "common.h"
#include "log_util.h"

#include <hccl/hccl_types.h>

namespace custom_comm {
namespace ms {

HcclResult InitCcuContext(HcclComm comm) {
    (void)comm;
    CC_LOG_ERROR("ccu_ms::InitCcuContext: MS backend not yet implemented");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult LaunchCcuKernel(HcclComm comm, const void *taskArg) {
    (void)comm; (void)taskArg;
    CC_LOG_ERROR("ccu_ms::LaunchCcuKernel: MS backend not yet implemented");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult GetCcuThreadHandle(HcclComm comm, uint64_t *threadHandle) {
    (void)comm;
    if (threadHandle != nullptr) *threadHandle = 0;
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ms
}  // namespace custom_comm
