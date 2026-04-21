// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// MS (MultiShot LoopGroup) variant of the batched AllGather CCU kernel.
// Scaffold: defines the CcuKernelArg/CcuKernel subclasses and the helper
// entry points Register/Launch that the engine context calls. The real
// LoopBlock-based Algorithm() body lands in a later commit; for now the
// kernel returns HCCL_E_NOT_SUPPORT at register time to keep the
// scaffolding honest.

#include "ccu_ms/ccu_kernel_ag_batch_mesh1d_ms.h"

#include "common.h"
#include "log_util.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <hccl/hccl_types.h>
#include <hcomm/ccu/ccu_kernel.h>
#include <hcomm/ccu/ccu_kernel_arg.h>
#include <hcomm/ccu/ccu_kernel_signature.h>
#include <hcomm/ccu/hccl_ccu_res.h>

namespace custom_comm {
namespace ms {

// -----------------------------------------------------------------------
// Argument bundle passed at kernel registration time.
// -----------------------------------------------------------------------
class CcuKernelArgAgBatchMs : public hcomm::CcuKernelArg {
public:
    CcuKernelArgAgBatchMs(uint32_t rankId, uint32_t rankSize,
                          std::vector<ChannelHandle> ch)
        : rankId_(rankId), rankSize_(rankSize) {
        channels = std::move(ch);
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override {
        hcomm::CcuKernelSignature sig;
        sig.Append("AgBatchMs");
        sig.Append(rankSize_);
        return sig;
    }

    uint32_t rankId_;
    uint32_t rankSize_;
};

// ----------------------------------------------------------------
// CCU kernel class (MS variant)
// ----------------------------------------------------------------
class CcuKernelAgBatchMs : public hcomm::CcuKernel {
public:
    explicit CcuKernelAgBatchMs(const hcomm::CcuKernelArg &arg)
        : hcomm::CcuKernel(arg),
          rankId_(static_cast<const CcuKernelArgAgBatchMs &>(arg).rankId_),
          rankSize_(static_cast<const CcuKernelArgAgBatchMs &>(arg).rankSize_)
    {}

 protected:
    HcclResult Algorithm() override {
        CC_LOG_ERROR("ccu_ms Algorithm not implemented yet (rank=%u size=%u)",
                     rankId_, rankSize_);
        return HCCL_E_NOT_SUPPORT;
    }

    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &) override {
        return {};
    }

 private:
    uint32_t rankId_{0};
    uint32_t rankSize_{0};
};

// Factory used by HCCL to instantiate kernel objects.
static std::unique_ptr<hcomm::CcuKernel>
MakeCcuKernelAgBatchMs(const hcomm::CcuKernelArg &arg) {
    return std::make_unique<CcuKernelAgBatchMs>(arg);
}

HcclResult RegisterBatchedAGKernelMs(HcclComm comm, CcuKernelHandle *handle,
                                     uint32_t rankId, uint32_t rankSize,
                                     const std::vector<ChannelHandle> &channels) {
    CcuKernelArgAgBatchMs arg(rankId, rankSize, channels);
    hcomm::KernelCreator creator = MakeCcuKernelAgBatchMs;
    return HcclCcuKernelRegister(comm, handle, &creator, &arg);
}

HcclResult LaunchBatchedAGKernelMs(HcclComm comm, ThreadHandle thread,
                                   CcuKernelHandle kernel,
                                   const AllGatherBatchTaskArg &taskArg) {
    (void)comm; (void)thread; (void)kernel; (void)taskArg;
    CC_LOG_ERROR("LaunchBatchedAGKernelMs: not yet implemented");
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ms
}  // namespace custom_comm
