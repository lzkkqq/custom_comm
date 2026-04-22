// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Minimal CcuKernelAlgBase for custom_comm, adapted from hccl's
// src/ops/op_common/template/ccu/ccu_kernel_alg_base.{h,cc} (Apache-2.0,
// (c) 2025 Huawei). Provides the AllocGoResource / CreateMultiOpBroadcast /
// GroupBroadcast pattern plus a batched variant GroupBroadcastBatch so
// custom_comm kernels can consume the hccl MS+Loop abstraction using only
// CANN SDK public symbols (hcomm::CcuKernel protected methods + CcuRep DSL).
//
// Scope: AllGather-flavored broadcast only. GroupReduce / GroupLocalReduce /
// GroupCopy / *WithoutMyRank are intentionally NOT ported.
//
// Naming: member fields follow hccl's convention (moConfig / moRes, no
// trailing underscore) so that these DO NOT shadow the same-named
// hcomm::CcuKernel::moConfig_ in the public SDK header.

#ifndef CUSTOM_COMM_OP_COMMON_CCU_KERNEL_ALG_BASE_H_
#define CUSTOM_COMM_OP_COMMON_CCU_KERNEL_ALG_BASE_H_

#include <cstdint>
#include <string>
#include <vector>

#include <hccl/hccl_types.h>
#include <hcomm/ccu/ccu_kernel.h>

namespace custom_comm {
namespace ccu_base {

// Four Variables that CalGoSize produces and GroupBroadcast consumes.
// Kept structurally identical to hccl's ops_hccl::CcuKernelAlgBase::GroupOpSize
// so the port of GroupBroadcast is a direct copy.
struct GroupOpSize {
    hcomm::CcuRep::Variable addrOffset;     // first loopgroup total offset
    hcomm::CcuRep::Variable loopParam;      // serial iter count (ctx | gsa | iter)
    hcomm::CcuRep::Variable parallelParam;  // loopgroup expansion (n | idx | total)
    hcomm::CcuRep::Variable residual;       // tail slice bytes
};

// We reuse hcomm::GroupOpConfig (defined at hcomm/ccu/ccu_kernel.h:62)
// verbatim instead of shadowing it with a sibling struct.

struct GroupOpSizeResource {
    std::vector<hcomm::CcuRep::CompletedEvent> completedEvent;
    std::vector<hcomm::CcuRep::CcuBuf>         ccuBuf;
    std::vector<hcomm::CcuRep::Executor>       executor;
};

// One entry in a batched broadcast. `gate` is a Variable whose runtime value
// is non-zero iff this desc carries data; typically the caller binds it to
// the Variable holding sendBytes so an empty desc emits no DSL.
struct BroadcastItem {
    hcomm::CcuRep::LocalAddr                src;
    std::vector<hcomm::CcuRep::RemoteAddr>  dst;    // size == channels.size() + 1
    GroupOpSize                             goSize;
    hcomm::CcuRep::Variable                 gate;
};

class CcuKernelAlgBase : public hcomm::CcuKernel {
public:
    using hcomm::CcuKernel::CcuKernel;

protected:
    // ---- Resource state (lazy, populated on first AllocGoResource) ----
    // NOTE: names are moConfig / moRes (no underscore) to match hccl and
    // to avoid shadowing hcomm::CcuKernel::moConfig_.
    hcomm::GroupOpConfig moConfig{0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFFFFFFFFFull};
    GroupOpSizeResource  moRes;

    // Bring in the SDK's Load(Variable) so the GroupOpSize overload below
    // doesn't hide it.
    using hcomm::CcuKernel::Load;
    void Load(const GroupOpSize& goSize);

    // ---- Resource creation wrappers (vector-returning) ----
    GroupOpSize                                CreateGroupOpSize();
    std::vector<hcomm::CcuRep::CcuBuf>         CreateBlockCcuBuf(uint32_t count);
    std::vector<hcomm::CcuRep::Executor>       CreateBlockExecutor(uint32_t count);
    std::vector<hcomm::CcuRep::CompletedEvent> CreateBlockCompletedEvent(uint32_t count);

    // Lazily allocate CcuBuf / Executor / CompletedEvent pools sized for the
    // configured MS layout. Idempotent; subsequent calls return early.
    void AllocGoResource(uint32_t parallelDim = 64u, uint32_t msPerLoop = 1u);

    // Thin wrapper around CcuRep::LoopGroupCall that pulls executor slots
    // from moRes.
    void LoopGroup(const std::vector<hcomm::CcuRep::LoopCall>& loops,
                   const std::vector<hcomm::CcuRep::Variable>& loopCfg,
                   const hcomm::CcuRep::Variable& paraCfg,
                   const hcomm::CcuRep::Variable& offsetCfg);

    // Emit the microcode for a single-src / (numPeers + 1)-dst broadcast.
    // Registers the shared LoopBlock the first time it's called (idempotent
    // via registeredLoop).
    HcclResult GroupBroadcast(const std::vector<ChannelHandle>& channels,
                              std::vector<hcomm::CcuRep::RemoteAddr> dst,
                              hcomm::CcuRep::LocalAddr src,
                              GroupOpSize goSize);

    // Emit DSL for a batch of broadcasts, one per item, guarded by
    // CCU_IF(gate != 0) so items with empty payload contribute no microcode.
    HcclResult GroupBroadcastBatch(const std::vector<ChannelHandle>& channels,
                                   const std::vector<BroadcastItem>& items);

private:
    HcclResult CreateMultiOpBroadcast(const std::vector<ChannelHandle>& channels);
};

}  // namespace ccu_base
}  // namespace custom_comm

#endif  // CUSTOM_COMM_OP_COMMON_CCU_KERNEL_ALG_BASE_H_
