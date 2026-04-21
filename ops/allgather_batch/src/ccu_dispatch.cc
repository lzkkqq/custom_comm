// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CCU backend dispatcher. Reads CUSTOM_COMM_CCU_MODE once per process
// and routes InitCcuContext / LaunchCcuKernel / GetCcuThreadHandle to either
// the SCHED implementation (ccu_sched/engine_ctx.cc) or the MS
// implementation (ccu_ms/engine_ctx_ms.cc).
//
// Valid values:
//   unset, "", "sched" -> SCHED path (byte-for-byte unchanged)
//   "ms"               -> MS path
//
// Unknown values fall back to SCHED with a log line for diagnosability.

#include "ccu_dispatch.h"

#include <cstdlib>
#include <cstring>

#include "ccu_ms/engine_ctx_ms.h"
#include "ccu_sched/engine_ctx.h"
#include "log_util.h"

namespace custom_comm {

namespace {

CcuMode ParseCcuMode() {
    const char *env = std::getenv("CUSTOM_COMM_CCU_MODE");
    if (env == nullptr || env[0] == '\0') {
        return CcuMode::kSched;
    }
    if (std::strcmp(env, "ms") == 0 || std::strcmp(env, "MS") == 0) {
        CC_LOG_INFO("CUSTOM_COMM_CCU_MODE=%s -> selecting MS backend", env);
        return CcuMode::kMs;
    }
    if (std::strcmp(env, "sched") == 0 || std::strcmp(env, "SCHED") == 0) {
        return CcuMode::kSched;
    }
    CC_LOG_ERROR("CUSTOM_COMM_CCU_MODE=%s is not recognized; falling back to SCHED", env);
    return CcuMode::kSched;
}

}  // namespace

CcuMode GetCcuMode() {
    // One-time resolve -- safe across threads (static initialization is thread safe since C++11).
    static const CcuMode cached = ParseCcuMode();
    return cached;
}

HcclResult DispatchInitCcuContext(HcclComm comm) {
    if (GetCcuMode() == CcuMode::kMs) {
        return ms::InitCcuContext(comm);
    }
    return InitCcuContext(comm);
}

HcclResult DispatchLaunchCcuKernel(HcclComm comm, const void *taskArg) {
    if (GetCcuMode() == CcuMode::kMs) {
        return ms::LaunchCcuKernel(comm, taskArg);
    }
    return LaunchCcuKernel(comm, taskArg);
}

HcclResult DispatchGetCcuThreadHandle(HcclComm comm, uint64_t *threadHandle) {
    if (GetCcuMode() == CcuMode::kMs) {
        return ms::GetCcuThreadHandle(comm, threadHandle);
    }
    return GetCcuThreadHandle(comm, threadHandle);
}

}  // namespace custom_comm
