// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from HCCL's
//   hccl/src/ops/op_common/template/ccu/ccu_kernel_utils.cc and
//   hccl/src/ops/op_common/template/ccu/ccu_kernel_alg_base.cc
// (Apache-2.0 licensed, Copyright 2025 Huawei Technologies Co., Ltd.).
//
// These bit-packed encodings MUST match HCCL byte-for-byte; they are the
// inputs to CCU microcode and any drift from HCCL's layout will silently
// yield wrong LoopGroup behavior at runtime.

#include "go_size.h"

#include <cstdint>

namespace custom_comm {
namespace ms {

namespace {

constexpr uint64_t Mask(uint16_t bits) {
    return bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1);
}

}  // namespace

uint64_t GetMaxLoopIterNum() {
    // 12-bit field, max = (1 << 12) - 1 = 4095.
    return Mask(12);
}

uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum) {
    constexpr uint16_t kCtxBits   = 8;
    constexpr uint16_t kCtxShift  = 45;
    constexpr uint16_t kGsaBits   = 32;
    constexpr uint16_t kGsaShift  = 13;
    constexpr uint16_t kIterBits  = 13;
    constexpr uint16_t kIterShift = 0;
    return ((loopCtxId   & Mask(kCtxBits))  << kCtxShift)
         | ((gsaOffset   & Mask(kGsaBits))  << kGsaShift)
         | ((loopIterNum & Mask(kIterBits)) << kIterShift);
}

uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex,
                          uint64_t totalLoopNum) {
    constexpr uint16_t kRepeatBits = 7, kRepeatShift = 55;
    constexpr uint16_t kIdxBits    = 7, kIdxShift    = 48;
    constexpr uint16_t kTotalBits  = 7, kTotalShift  = 41;
    return ((repeatNum       & Mask(kRepeatBits)) << kRepeatShift)
         | ((repeatLoopIndex & Mask(kIdxBits))    << kIdxShift)
         | ((totalLoopNum    & Mask(kTotalBits))  << kTotalShift);
}

uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset) {
    constexpr uint16_t kGsaBits = 32, kGsaShift = 21;
    constexpr uint16_t kMsBits  = 11, kMsShift  = 10;
    constexpr uint16_t kCkeBits = 10, kCkeShift = 0;
    return ((gsaOffset & Mask(kGsaBits)) << kGsaShift)
         | ((msOffset  & Mask(kMsBits))  << kMsShift)
         | ((ckeOffset & Mask(kCkeBits)) << kCkeShift);
}


}  // namespace ms
}  // namespace custom_comm
