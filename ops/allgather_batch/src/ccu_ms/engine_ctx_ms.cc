// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CCU MS backend engine context.
// Mirrors ccu_sched/engine_ctx.cc, but registers the MS kernel template
// (CcuKernelAgBatchMesh1DMs) instead of the SCHED one. Selected via
// CUSTOM_COMM_CCU_MODE=ms.
//
// For the moment this is a skeleton: Init() and Launch() return
// HCCL_E_NOT_SUPPORT until the kernel is wired up in commit 4.

#include "ccu_ms/engine_ctx_ms.h"
#include "common.h"
#include "log_util.h"

#include <hccl/hccl_types.h>

namespace custom_comm {
namespace ms {

HcclResult InitCcuContext(HcclComm /*comm*/) {
    CC_LOG_ERROR("ccu_ms::InitCcuContext: MS backend not yet implemented");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult LaunchCcuKernel(HcclComm /*comm*/, const void * /*taskArg*/) {
    CC_LOG_ERROR("ccu_ms::LaunchCcuKernel: MS backend not yet implemented");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult GetCcuThreadHandle(HcclComm /*comm*/, uint64_t * /*threadHandle*/) {
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ms
}  // namespace custom_comm
