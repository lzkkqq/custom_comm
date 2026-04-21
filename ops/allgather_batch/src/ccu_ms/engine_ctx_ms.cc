// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CCU MS backend engine context.
//
// Mirrors ccu_sched/engine_ctx.cc but:
//   * Uses the MS kernel (RegisterBatchedAGKernelMs from ccu_ms/) instead of
//     the SCHED variant.
//   * Distinct ctx tag so SCHED and MS can coexist on the same comm.
//
// The MS kernel itself is still a skeleton (Algorithm() returns HCCL_E_NOT_SUPPORT),
// so calls that reach here will propagate that error to the caller.

#include "ccu_ms/engine_ctx_ms.h"
#include "ccu_ms/ccu_kernel_ag_batch_mesh1d_ms.h"
#include "common.h"
#include "log_util.h"

#include <hccl/hccl_comm.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res.h>
#include <hcomm/ccu/ccu_kernel.h>
#include <hcomm/ccu/hccl_ccu_res.h>

#include <acl/acl.h>

#include <cstdint>
#include <vector>

namespace custom_comm {
namespace ms {

static constexpr const char *CTX_TAG_MS = "custom_comm_ag_batch_ms";

// XN slots required: token(1) + MAX_DESC_COUNT recv addrs + post-sync(1).
// Matches SCHED so the two paths can co-exist.
static constexpr uint32_t NOTIFY_COUNT = 1 + MAX_DESC_COUNT + 1;

namespace {
struct CcuContextMs {
    CcuKernelHandle kernelHandle{};
    ThreadHandle    threadHandle{};
    bool            initialized{false};
};

CcuContextMs *LookupCtx(HcclComm comm) {
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, CTX_TAG_MS, COMM_ENGINE_CCU, &ctx, &ctxSize) == HCCL_SUCCESS) {
        return static_cast<CcuContextMs *>(ctx);
    }
    return nullptr;
}

}  // namespace

HcclResult InitCcuContext(HcclComm comm) {
    (void)comm;
    CC_LOG_ERROR("ccu_ms::InitCcuContext: not yet implemented");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult LaunchCcuKernel(HcclComm comm, const void *taskArg) {
    (void)comm; (void)taskArg;
    return HCCL_E_NOT_SUPPORT;
}

HcclResult GetCcuThreadHandle(HcclComm comm, uint64_t *threadHandle) {
    (void)comm;
    if (threadHandle) *threadHandle = 0;
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ms
}  // namespace custom_comm
