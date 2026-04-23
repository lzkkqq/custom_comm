// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CcuKernelAlgBase -- MS+Loop DSL helpers ported from hccl's internal
//   src/ops/op_common/template/ccu/ccu_kernel_alg_base.cc (Apache-2.0,
//   Copyright 2025 Huawei Technologies Co., Ltd.)
//
// custom_comm only needs the batched broadcast flavor, so this file exposes
// a single entry point: GroupBroadcastBatch. The two-LoopGroup split (one
// for integer multiples of memSlice*loopCount, one for the tail slab) is
// inlined directly inside GroupBroadcastBatch; there is no single-desc
// GroupBroadcast helper. CreateMultiOpBroadcastBatch registers the shared
// LoopBlock ("broadcast_batch_loop_*") exactly once per kernel instance.
//
// Bit-packing helpers (GetLoopParam / GetParallelParam / GetOffsetParam /
// CalGoSize) live in ops/allgather_batch/src/ccu_ms/go_size.{h,cc} and are
// re-used verbatim here.

#include "ccu_kernel_alg_base.h"

#include "ccu_ms/go_size.h"
#include "log_util.h"

#include <hcomm/ccu/ccu_condition_v1.h>
#include <hcomm/ccu/ccu_loopblock_v1.h>
#include <hcomm/ccu/ccu_loopcall_v1.h>
#include <hcomm/ccu/ccu_loopgroupcall_v1.h>

#include <cstdint>
#include <string>
#include <vector>

// Local CHK_RET -- hccl's version lives in a non-public header.
#define CHK_RET_LOCAL(expr)                                              \
    do {                                                                 \
        HcclResult _ret_local = (expr);                                  \
        if (_ret_local != HCCL_SUCCESS) {                                \
            CC_LOG_ERROR("CcuKernelAlgBase: %s -> %d at %s:%d",          \
                         #expr, static_cast<int>(_ret_local),            \
                         __FILE__, __LINE__);                            \
            return _ret_local;                                           \
        }                                                                \
    } while (0)

namespace custom_comm {
namespace ccu_base {

namespace ms = ::custom_comm::ms;  // go_size.h constants + bit-pack helpers

// ============================================================
// CalGoSize (parallelDim-aware)
//
// Host-side GoSize calculation parametrized by parallelDim so the layout
// emitted here matches the AllocGoResource(parallelDim) used at runtime.
// ============================================================

ms::GoSize CalGoSize(uint64_t size, uint32_t parallelDim) {
    const uint64_t memSlice = ms::kCcuMsSize;
    const uint64_t loopSize = memSlice * parallelDim;

    uint64_t loopIterNum = size / loopSize;
    uint64_t leftover    = size - loopIterNum * loopSize;
    uint64_t n           = leftover / memSlice;
    uint64_t p           = leftover - n * memSlice;

    ms::GoSize go{};
    go.addrOffset = loopIterNum * loopSize;
    // NOTE: loopParam carries only the raw loopIterNum (iter field). The
    // device-side Algorithm in GroupBroadcastBatch bit-packs the full
    // GetLoopParam(0, loopSize, 0) there and does `loopParam += goSize.loopParam`,
    // which only adds to the iter field. If we bit-packed loopSize here too,
    // the gsa field would double and each serial round would stride by
    // 2*loopSize, skipping half of the payload.
    go.loopParam  = loopIterNum;
    if (n == 0 && p == 0) {
        go.parallelParam = 0;
        go.residual      = 0;
    } else if (n != 0 && p == 0) {
        go.parallelParam = ms::GetParallelParam(n - 1, 0, 1);
        go.residual      = memSlice;
    } else if (n == 0 && p != 0) {
        go.parallelParam = ms::GetParallelParam(0, 0, 1);
        go.residual      = p;
    } else {
        go.parallelParam = ms::GetParallelParam(n - 1, 1, 2);
        go.residual      = p;
    }
    return go;
}

// ============================================================
// AllocGoResource
// ============================================================

void CcuKernelAlgBase::AllocGoResource(uint32_t parallelDim, uint32_t msPerLoop) {
    if (moConfig.loopCount    != 0xFFFFFFFFu &&
        moConfig.msInterleave != 0xFFFFFFFFu &&
        moConfig.memSlice     != 0xFFFFFFFFFFFFFFFFull) {
        return;  // already configured
    }
    moConfig = hcomm::GroupOpConfig{
        ms::kCcuMsInterleave,
        ms::kCcuMsDefaultLoopCount,
        ms::kCcuMsSize,
    };
    moConfig.loopCount = parallelDim;
    moConfig.memSlice  = static_cast<uint64_t>(msPerLoop) * ms::kCcuMsSize;

    CC_LOG_INFO("AllocGoResource: loopCount=%u msInterleave=%u memSlice=%llu",
                moConfig.loopCount, moConfig.msInterleave,
                static_cast<unsigned long long>(moConfig.memSlice));

    if (moRes.executor.empty()) {
        moRes.executor       = CreateBlockExecutor(moConfig.loopCount);
        moRes.completedEvent = CreateBlockCompletedEvent(moConfig.loopCount);
        moRes.ccuBuf         = CreateBlockCcuBuf(moConfig.loopCount * moConfig.msInterleave);
    }
}

// ============================================================
// Load(GroupOpSize)
// ============================================================

void CcuKernelAlgBase::Load(const GroupOpSize& goSize) {
    Load(goSize.addrOffset);
    Load(goSize.loopParam);
    Load(goSize.parallelParam);
    Load(goSize.residual);
}

// ============================================================
// CreateMultiOpBroadcastBatch -- one-time LoopBlock registration.
// Uses the "broadcast_batch_loop_*" label namespace so it stays independent
// of any future single-desc GroupBroadcast port.
// ============================================================

HcclResult CcuKernelAlgBase::CreateMultiOpBroadcastBatch(
        const std::vector<ChannelHandle>& channels) {
    using namespace hcomm;  // Loop/WriteNb/WaitEvent/CcuRep names

    AllocGoResource();

    const std::string loopType = "broadcast_batch";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return HCCL_SUCCESS;
    }

    const uint32_t channelSize = static_cast<uint32_t>(channels.size());
    const uint32_t size        = channelSize + 1;   // peers + self

    for (uint32_t index = 0; index < 2; ++index) {  // two parallel LoopBlocks
        CcuRep::LocalAddr src = CreateLocalAddr();
        std::vector<CcuRep::RemoteAddr> dst;
        for (uint32_t i = 0; i < size; ++i) {
            CcuRep::LocalAddr tmp = CreateLocalAddr();
            dst.emplace_back(*reinterpret_cast<CcuRep::RemoteAddr*>(&tmp));
        }
        CcuRep::Variable  len = CreateVariable();
        CcuRep::LoopBlock lb(this, loopType + "_loop_" + std::to_string(index));
        lb(src, dst, len);

        CcuRep::CcuBuf         &buf   = moRes.ccuBuf[index * moConfig.msInterleave];
        CcuRep::CompletedEvent &event = moRes.completedEvent[index];

        event.mask = 1u;
        LocalCopyNb(buf, src, len, event);
        WaitEvent(event);

        for (uint32_t i = 0; i < channelSize; ++i) {
            if (channels[i] == 0) {
                return HCCL_E_PTR;
            }
            event.mask = 1u << i;
            CHK_RET_LOCAL(WriteNb(channels[i], dst[i], buf, len, event));
        }
        CcuRep::LocalAddr &localDst = *reinterpret_cast<CcuRep::LocalAddr*>(&dst[size - 1]);
        event.mask = 1u << channelSize;
        LocalCopyNb(localDst, buf, len, event);
        event.mask = (1u << size) - 1u;
        WaitEvent(event);
    }

    registeredLoop.insert(loopType);
    return HCCL_SUCCESS;
}

// ============================================================
// GroupBroadcastBatch -- self-contained multi-desc broadcast.
//
// For each item: guard with CCU_IF(gate != 0); if active, emit the same
// two-LoopGroup split that CalGoSize produces on the host side:
//   - LoopGroup[0] handles integer multiples of (memSlice * loopCount)
//     using loopIterNum serial iterations
//   - LoopGroup[1] handles the tail (n full slices + residual p) in one
//     parallel expansion
//
// src / dst are taken by value from item so mutations inside CCU_IF blocks
// are local to this iteration.
// ============================================================

HcclResult CcuKernelAlgBase::GroupBroadcastBatch(
        const std::vector<ChannelHandle>& channels,
        const std::vector<BroadcastItem>& items) {
    using namespace hcomm;

    CHK_RET_LOCAL(CreateMultiOpBroadcastBatch(channels));

    const uint32_t size = static_cast<uint32_t>(channels.size()) + 1;

    for (const auto& item : items) {
        if (item.dst.size() != size) {
            CC_LOG_ERROR("GroupBroadcastBatch: item.dst size=%zu != channels+1=%u",
                         item.dst.size(), size);
            return HCCL_E_PARA;
        }

        CCU_IF(item.gate != 0) {
            auto              src    = item.src;
            auto              dst    = item.dst;
            const auto       &goSize = item.goSize;

            // LoopGroup[0]: integer multiples of (memSlice * loopCount)
            CCU_IF(goSize.addrOffset != 0) {
                CcuRep::Variable loopParam = CreateVariable();
                loopParam = ms::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
                loopParam += goSize.loopParam;

                CcuRep::Variable sliceSize = CreateVariable();
                sliceSize = moConfig.memSlice;
                auto lc = Loop("broadcast_batch_loop_0")(src, dst, sliceSize);

                CcuRep::Variable paraCfg = CreateVariable();
                paraCfg = ms::GetParallelParam(moConfig.loopCount - 1, 0, 1);
                CcuRep::Variable offsetCfg = CreateVariable();
                offsetCfg = ms::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

                LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
            }

            // LoopGroup[1]: tail slab (n full slices + residual p)
            CCU_IF(goSize.parallelParam != 0) {
                src.addr += goSize.addrOffset;
                for (uint32_t i = 0; i < size; ++i) {
                    dst[i].addr += goSize.addrOffset;
                }

                auto lc0 = Loop("broadcast_batch_loop_0")(src, dst, goSize.residual);

                src.addr += goSize.residual;
                for (uint32_t i = 0; i < size; ++i) {
                    dst[i].addr += goSize.residual;
                }

                CcuRep::Variable sliceSize = CreateVariable();
                sliceSize = moConfig.memSlice;
                auto lc1 = Loop("broadcast_batch_loop_1")(src, dst, sliceSize);

                CcuRep::Variable loopCfg0 = CreateVariable();
                loopCfg0 = ms::GetLoopParam(0, 0, 1);
                CcuRep::Variable loopCfg1 = CreateVariable();
                loopCfg1 = ms::GetLoopParam(0, 0, 1);
                CcuRep::Variable offsetCfg = CreateVariable();
                offsetCfg = ms::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

                LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
            }
        }
    }
    return HCCL_SUCCESS;
}

// ============================================================
// LoopGroup / CreateGroupOpSize / CreateBlock* helpers
// ============================================================

void CcuKernelAlgBase::LoopGroup(const std::vector<hcomm::CcuRep::LoopCall>& loops,
                                 const std::vector<hcomm::CcuRep::Variable>& loopCfg,
                                 const hcomm::CcuRep::Variable& paraCfg,
                                 const hcomm::CcuRep::Variable& offsetCfg) {
    auto lgc = hcomm::CcuRep::LoopGroupCall(this);
    std::vector<hcomm::CcuRep::Executor> executors;
    executors.reserve(loops.size());
    for (size_t i = 0; i < loops.size(); ++i) {
        executors.push_back(moRes.executor[i]);
    }
    lgc.Run(loops, loopCfg, executors, paraCfg, offsetCfg);
}

GroupOpSize CcuKernelAlgBase::CreateGroupOpSize() {
    return GroupOpSize{CreateVariable(), CreateVariable(),
                       CreateVariable(), CreateVariable()};
}

std::vector<hcomm::CcuRep::CcuBuf> CcuKernelAlgBase::CreateBlockCcuBuf(uint32_t count) {
    std::vector<hcomm::CcuRep::CcuBuf> res(count);
    hcomm::CcuKernel::CreateBlockCcuBuf(count, res.data());
    return res;
}

std::vector<hcomm::CcuRep::Executor> CcuKernelAlgBase::CreateBlockExecutor(uint32_t count) {
    std::vector<hcomm::CcuRep::Executor> res(count);
    hcomm::CcuKernel::CreateBlockExecutor(count, res.data());
    return res;
}

std::vector<hcomm::CcuRep::CompletedEvent> CcuKernelAlgBase::CreateBlockCompletedEvent(uint32_t count) {
    std::vector<hcomm::CcuRep::CompletedEvent> res(count);
    hcomm::CcuKernel::CreateBlockCompletedEvent(count, res.data());
    return res;
}

}  // namespace ccu_base
}  // namespace custom_comm
