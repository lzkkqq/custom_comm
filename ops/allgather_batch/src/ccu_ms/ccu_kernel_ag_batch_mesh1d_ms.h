// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CCU MS kernel for batched AllGather on mesh1d.
//
// Declares the registration and launch entry points that the MS engine_ctx
// uses to install the kernel and issue it per invocation. Implementation
// lives in ccu_kernel_ag_batch_mesh1d_ms.cc.

#ifndef CUSTOM_COMM_OPS_ALLGATHER_BATCH_CCU_MS_CCU_KERNEL_AG_BATCH_MESH1D_MS_H_
#define CUSTOM_COMM_OPS_ALLGATHER_BATCH_CCU_MS_CCU_KERNEL_AG_BATCH_MESH1D_MS_H_

#include "common.h"

#include <cstdint>
#include <vector>

#include <hccl/hccl_types.h>
#include <hcomm/ccu/ccu_kernel.h>

namespace custom_comm {
namespace ms {

// Registers the MS kernel with the HCCL engine.
// `handle` out-parameter is the CcuKernelHandle that LaunchBatchedAGKernelMs
// expects on subsequent calls for this comm.
HcclResult RegisterBatchedAGKernelMs(
    HcclComm comm,
    CcuKernelHandle* handle,
    uint32_t rankId,
    uint32_t rankSize,
    const std::vector<ChannelHandle>& channels);

// Launch a previously-registered MS kernel for the given descriptor batch.
HcclResult LaunchBatchedAGKernelMs(
    HcclComm comm,
    ThreadHandle threadHandle,
    CcuKernelHandle kernel,
    const AllGatherBatchTaskArg& taskArg);

// ---- V2: GroupBroadcastBatch-based kernel (CUSTOM_COMM_CCU_MS_IMPL=v2) ----
//
// Same external contract as the v1 functions above; the only differences are
// the kernel signature ("AgBatchMesh1DMsV2_" + rankSize) and the internal DSL
// (one GroupBroadcastBatch call vs the hand-rolled LoopBlock+LoopGroupCall).
HcclResult RegisterBatchedAGKernelMsV2(
    HcclComm comm,
    CcuKernelHandle* handle,
    uint32_t rankId,
    uint32_t rankSize,
    const std::vector<ChannelHandle>& channels);

HcclResult LaunchBatchedAGKernelMsV2(
    HcclComm comm,
    ThreadHandle threadHandle,
    CcuKernelHandle kernel,
    const AllGatherBatchTaskArg& taskArg);

}  // namespace ccu_ms
}  // namespace custom_comm

#endif  // CUSTOM_COMM_OPS_ALLGATHER_BATCH_SRC_CCU_MS_CCU_KERNEL_AG_BATCH_MESH1D_MS_H_
