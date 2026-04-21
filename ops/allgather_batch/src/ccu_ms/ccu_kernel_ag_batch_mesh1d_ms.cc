// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CCU MS (MultiShot / LoopGroup) kernel for allgather_batch on mesh1d.
//
// Scaffold: defines the kernel arg + kernel class that register via the
// public hcomm CcuKernel API, plus the helper entry points used by the
// ccu_ms engine context. The Algorithm() body is intentionally empty here
// (no microcode emitted) -- it is filled in by the follow-up commit once
// the SCHED baseline has landed. Until that point any Launch at runtime
// will return HCCL_E_NOT_SUPPORT via engine_ctx_ms.cc.

#include "ccu_ms/ccu_kernel_ag_batch_mesh1d_ms.h"

#include "common.h"
#include "log_util.h"

#include <hccl/hccl_types.h>
#include <hcomm/ccu/ccu_kernel.h>
#include <hcomm/ccu/ccu_kernel_arg.h>
#include <hcomm/ccu/ccu_kernel_signature.h>
#include <hcomm/ccu/ccu_task_arg_v1.h>
#include <hcomm/ccu/hccl_ccu_res.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace custom_comm {
namespace ms {

// ============================================================
// Argument class: what the factory receives from engine_ctx.
// ============================================================
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

// ============================================================
// Per-invocation task arg (addresses + per-desc byte counts).
// Matches the SCHED variant's layout so both share AllGatherBatchTaskArg.
// ============================================================
struct CcuTaskArgMs : public hcomm::CcuTaskArg {
    uint32_t descCount;
    uint64_t token;
    uint64_t sendAddr[MAX_DESC_COUNT];
    uint64_t recvAddr[MAX_DESC_COUNT];
    uint64_t sendBytes[MAX_DESC_COUNT];
    uint64_t selfOffset[MAX_DESC_COUNT];
};

// ============================================================
// CCU MS kernel class
// ============================================================
class CcuKernelAgBatchMesh1dMs : public hcomm::CcuKernel {
public:
    explicit CcuKernelAgBatchMesh1dMs(const hcomm::CcuKernelArg &arg)
        : hcomm::CcuKernel(arg)
    {
        const auto &argMs = static_cast<const CcuKernelArgAgBatchMs &>(arg);
        rankId_   = argMs.rankId_;
        rankSize_ = argMs.rankSize_;
    }

protected:
    // Skeleton: real MS LoopGroup microcode lands in the follow-up commit.
    HcclResult Algorithm() override {
        CC_LOG_INFO("[CcuKernelAgBatchMesh1dMs] Algorithm skeleton; returning E_NOT_SUPPORT");
        return HCCL_E_NOT_SUPPORT;
    }

    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg & /*arg*/) override {
        return {};
    }

private:
    uint32_t rankId_{0};
    uint32_t rankSize_{0};
};

// ============================================================
// Factory for HcclCcuKernelRegister
// ============================================================
static std::unique_ptr<hcomm::CcuKernel>
MakeCcuKernelAgBatchMesh1dMs(const hcomm::CcuKernelArg &arg)
{
    return std::make_unique<CcuKernelAgBatchMesh1dMs>(arg);
}

// ============================================================
// Public entry points (declared in ccu_kernel_ag_batch_mesh1d_ms.h)
// ============================================================

HcclResult RegisterBatchedAGKernelMs(HcclComm comm, CcuKernelHandle *handle,
                                     uint32_t rankId, uint32_t rankSize,
                                     const std::vector<ChannelHandle> &channels)
{
    CcuKernelArgAgBatchMs arg(rankId, rankSize, channels);
    hcomm::KernelCreator creator = &MakeCcuKernelAgBatchMesh1dMs;
    return HcclCcuKernelRegister(comm, handle, &creator, &arg);
}

HcclResult LaunchBatchedAGKernelMs(HcclComm comm, ThreadHandle thread,
                                   CcuKernelHandle kernel,
                                   const AllGatherBatchTaskArg &taskArg) {
    // Placeholder: fail fast; will be replaced with real task-arg marshal in
    // the follow-up commit.
    (void)comm; (void)thread; (void)kernel; (void)taskArg;
    CC_LOG_ERROR("LaunchBatchedAGKernelMs: not implemented yet");
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ms
}  // namespace custom_comm
