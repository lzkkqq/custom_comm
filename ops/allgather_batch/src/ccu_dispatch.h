// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CCU backend dispatcher: routes InitCcuContext / LaunchCcuKernel /
// GetCcuThreadHandle to either the SCHED or the MS implementation based on
// the CUSTOM_COMM_CCU_MODE env var (values: "sched" default, or "ms").

#ifndef CUSTOM_COMM_CCU_DISPATCH_H
#define CUSTOM_COMM_CCU_DISPATCH_H

#include <cstdint>

#include <hccl/hccl_types.h>

namespace custom_comm {

enum class CcuMode : uint8_t {
    kSched = 0,
    kMs    = 1,
};

// Reads env var on first call and caches the result for the process.
CcuMode GetCcuMode();

// Dispatches to the selected backend. Shared C ABI with both SCHED and MS.
HcclResult DispatchInitCcuContext(HcclComm comm);
HcclResult DispatchLaunchCcuKernel(HcclComm comm, const void* taskArg);
HcclResult DispatchGetCcuThreadHandle(HcclComm comm, uint64_t* threadHandle);

}  // namespace custom_comm

#endif  // CUSTOM_COMM_CCU_DISPATCH_H
