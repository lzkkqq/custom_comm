// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CCU MS kernel for batched AllGather on Mesh-1D, declared here and
// implemented in ccu_kernel_ag_batch_mesh1d_ms.cc. Registered by
// engine_ctx_ms.cc when the CCU path is activated.

#ifndef CUSTOM_COMM_OPS_ALLGATHER_BATCH_CCU_MS_KERNEL_H_
#define CUSTOM_COMM_OPS_ALLGATHER_BATCH_CCU_MS_KERNEL_H_

#include "common.h"

#include <cstdint>
#include <vector>

#include <hccl/hccl_types.h>
#include <hcomm/ccu/ccu_kernel.h>

namespace custom_comm {
namespace ms {

// Register the MS kernel with HCCL. `handle` is written out and must be
// kept alive as long as LaunchBatchedAGKernelMs may be called.
HcclResult RegisterBatchedAGKernelMs(
    HcclComm comm,
    CcuKernelHandle* handle,
    uint32_t rankId,
    uint32_t rankSize,
    const std::vector<ChannelHandle>& channels);

// Launch the MS kernel for a batched AllGather given per-desc runtime data.
HcclResult LaunchBatchedAGKernelMs(
    HcclComm comm,
    ThreadHandle thread,
    CcuKernelHandle kernel,
    const AllGatherBatchTaskArg& taskArg);

}  // namespace ms
}  // namespace custom_comm

#endif  // CUSTOM_COMM_CCU_MS_KERNEL_AG_BATCH_MESH1D_MS_H_
