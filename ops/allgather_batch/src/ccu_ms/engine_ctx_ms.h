// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Engine context for the MS (LoopGroup MultiShot) variant of the allgather_batch
// CCU backend.  Mirror of ccu_sched/engine_ctx.h but in namespace custom_comm::ccu_ms.

#ifndef CUSTOM_COMM_CCU_MS_ENGINE_CTX_H
#define CUSTOM_COMM_CCU_MS_ENGINE_CTX_H

#include <cstdint>
#include <hccl/hccl_types.h>

namespace custom_comm {
namespace ms {

HcclResult InitCcuContext(HcclComm comm);
HcclResult LaunchCcuKernel(HcclComm comm, const void* taskArg);
HcclResult GetCcuThreadHandle(HcclComm comm, uint64_t *threadHandle);

}  // namespace ms
}  // namespace custom_comm

#endif  // CUSTOM_COMM_CCU_MS_ENGINE_CTX_MS_H_
