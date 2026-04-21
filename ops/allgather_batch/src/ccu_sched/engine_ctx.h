// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// FROZEN CONTRACT -- do not change after decomposed path ships.

#ifndef CUSTOM_COMM_ENGINE_CTX_H_
#define CUSTOM_COMM_ENGINE_CTX_H_

#include <hccl/hccl_types.h>

// ============================================================
// EngineCtx lifecycle (CCU path implementation)
// ============================================================

// CcuContext holds registered CCU kernel handle and thread handle,
// cached per (comm, ctxTag) via HcclEngineCtxCreate/Get.
// decomposed path: these functions are stubs.
// CCU path: full CCU resource allocation + kernel registration.

namespace custom_comm {

// Initialize CCU context for the given communicator.
// First call: HcclEngineCtxCreate + channel acquire + kernel register.
// Subsequent calls: HcclEngineCtxGet (cached).
// Returns HCCL_E_NOT_SUPPORT in decomposed path.
HcclResult InitCcuContext(HcclComm comm);

// Launch the CCU kernel with given task arguments.
// Returns HCCL_E_NOT_SUPPORT in decomposed path.
HcclResult LaunchCcuKernel(HcclComm comm, const void *taskArg);

// Retrieve the CCU thread handle from the cached context.
// Used by aclGraph capture to get the slave stream via HcclThreadResGetInfo.
HcclResult GetCcuThreadHandle(HcclComm comm, uint64_t *threadHandle);

}  // namespace custom_comm

#endif  // CUSTOM_COMM_ENGINE_CTX_H_
