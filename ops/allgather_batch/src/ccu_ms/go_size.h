// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// GoSize + microcode-arg helpers ported from HCCL's internal
// ccu_kernel_utils.{h,cc} and ccu_kernel_alg_base.cc::CalGoSize
// (Apache-2.0, (c) Huawei 2025). Keeps custom_comm's MS kernel on
// SDK public symbols only.

#ifndef CUSTOM_COMM_CCU_MS_GO_SIZE_H_
#define CUSTOM_COMM_CCU_MS_GO_SIZE_H_

#include <cstdint>

namespace custom_comm {
namespace ms {

// Mirror of hcomm::CcuRep::CCU_MS_* constants.
constexpr uint32_t kCcuMsSize             = 4096;
constexpr uint32_t kCcuMsInterleave       = 8;
constexpr uint32_t kCcuMsDefaultLoopCount = 64;

struct GoSize {
    uint64_t addrOffset;
    uint64_t loopParam;
    uint64_t parallelParam;
    uint64_t residual;
};

uint64_t GetMaxLoopIterNum();
uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum);
uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum);
uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset);

// parallelDim MUST equal the value passed to AllocGoResource() on the
// consuming kernel; the host-side slice layout and the LoopGroup[0] paraCfg
// both derive from it, and any drift will leave the tail of a large payload
// unprocessed. Defaults to kCcuMsDefaultLoopCount (64) to match HCCL.
GoSize CalGoSize(uint64_t totalBytes, uint32_t parallelDim = kCcuMsDefaultLoopCount);

}  // namespace ms
}  // namespace custom_comm

#endif  // CUSTOM_COMM_CCU_MS_GO_SIZE_H_
