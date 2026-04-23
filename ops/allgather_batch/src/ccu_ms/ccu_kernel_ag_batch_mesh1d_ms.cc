// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CCU MS kernel for batched AllGather on Mesh-1D. Built on
// ccu_base::CcuKernelAlgBase so the per-desc DSL collapses into a single
// GroupBroadcastBatch call; host-side CalGoSize feeds the two-stage
// LoopGroup split packed into the GeneArgs payload.

#include "ccu_kernel_ag_batch_mesh1d_ms.h"

#include "ccu_kernel_alg_base.h"
#include "ccu_ms/go_size.h"
#include "common.h"
#include "log_util.h"

#include <hcomm/ccu/ccu_assist_pub.h>
#include <hcomm/ccu/ccu_condition_v1.h>
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
// XN slot layout per channel (for NotifyRecord/NotifyWait)
// ============================================================

static constexpr uint32_t TOKEN_XN_ID       = 0;
static constexpr uint32_t RECV_ADDR_XN_BASE = 1;                    // 1..MAX_DESC_COUNT
static constexpr uint32_t POST_SYNC_XN_ID   = RECV_ADDR_XN_BASE + MAX_DESC_COUNT;
static constexpr uint32_t CKE_IDX           = 0;

static constexpr uint32_t PRE_SYNC_MASK = (1u << (MAX_DESC_COUNT + 1)) - 1;

// ============================================================
// GeneArgs slot layout
// ============================================================
//
//   [0]                         token (RDMA access credential)
//   [1 + 8*d + 0]               sendAddr[d]
//   [1 + 8*d + 1]               recvAddr[d]
//   [1 + 8*d + 2]               selfOffset[d]     = rankId * sendBytes[d]
//   [1 + 8*d + 3]               sendBytes[d]      (gate: CCU_IF(sendBytes != 0))
//   [1 + 8*d + 4]               goSize.addrOffset (host-side CalGoSize)
//   [1 + 8*d + 5]               goSize.loopParam
//   [1 + 8*d + 6]               goSize.parallelParam
//   [1 + 8*d + 7]               goSize.residual
//
// Total: 1 + 8 * MAX_DESC_COUNT = 49 slots (MAX_DESC_COUNT = 6)

static constexpr uint32_t GENE_ARGS_PER_DESC  = 8;
static constexpr uint32_t GENE_ARGS_DESC_BASE = 1;
static constexpr uint32_t GENE_ARGS_TOTAL     =
    GENE_ARGS_DESC_BASE + GENE_ARGS_PER_DESC * MAX_DESC_COUNT;

// Hardware invariant: a LoopGroup fans out exactly 64 Loops in parallel
// (CCU_MS_DEFAULT_LOOP_COUNT). The host-side slice formula m*256KB + n*4KB + p
// is derived from this 64-way parallelism (256KB = 64 * 4KB). Both the
// AllocGoResource parallelDim and the CalGoSize parallelDim MUST be 64.
static constexpr uint32_t kMsParallelDim =
    static_cast<uint32_t>(::custom_comm::ms::kCcuMsDefaultLoopCount);

// ============================================================
// CcuKernelArg subclass (registration-time, per-rank-size signature)
// ============================================================

class CcuKernelArgBatchMs : public hcomm::CcuKernelArg {
public:
    CcuKernelArgBatchMs(uint32_t rankSize, uint32_t rankId,
                          std::vector<ChannelHandle> ch)
        : rankSize_(rankSize), rankId_(rankId) {
        channels = std::move(ch);
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override {
        hcomm::CcuKernelSignature sig;
        sig.Append("AgBatchMesh1DMs_");
        sig.Append(rankSize_);
        return sig;
    }

    uint32_t rankSize_;
    uint32_t rankId_;
};

// ============================================================
// CcuTaskArg subclass (per-invocation host-side data)
// ============================================================

class CcuTaskArgBatchMs : public hcomm::CcuTaskArg {
public:
    uint32_t descCount;
    uint64_t token;
    uint64_t sendAddr[MAX_DESC_COUNT];
    uint64_t recvAddr[MAX_DESC_COUNT];
    uint64_t selfOffset[MAX_DESC_COUNT];
    uint64_t sendBytes[MAX_DESC_COUNT];
    GoSize   goSize[MAX_DESC_COUNT];                          // addrOffset/loopParam/parallelParam/residual
};

// ============================================================
// Kernel class -- inherits from ccu_base::CcuKernelAlgBase
// ============================================================

class CcuKernelAllGatherBatchMesh1DMs
    : public ::custom_comm::ccu_base::CcuKernelAlgBase {
public:
    explicit CcuKernelAllGatherBatchMesh1DMs(const hcomm::CcuKernelArg& arg);

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg& arg) override;

private:
    uint32_t rankId_;
    uint32_t rankSize_;
};

// ---- Constructor ----

CcuKernelAllGatherBatchMesh1DMs::CcuKernelAllGatherBatchMesh1DMs(
        const hcomm::CcuKernelArg& arg)
    : ::custom_comm::ccu_base::CcuKernelAlgBase(arg) {
    auto& ba  = static_cast<const CcuKernelArgBatchMs&>(arg);
    rankId_   = ba.rankId_;
    rankSize_ = ba.rankSize_;
}

// ---- GeneArgs: serialize runtime parameters into the 49-slot SQE payload ----

std::vector<uint64_t>
CcuKernelAllGatherBatchMesh1DMs::GeneArgs(const hcomm::CcuTaskArg& arg) {
    auto& ta = static_cast<const CcuTaskArgBatchMs&>(arg);

    std::vector<uint64_t> slots(GENE_ARGS_TOTAL, 0);
    slots[0] = ta.token;

    for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
        const uint32_t base = GENE_ARGS_DESC_BASE + d * GENE_ARGS_PER_DESC;
        if (d < ta.descCount) {
            slots[base + 0] = ta.sendAddr[d];
            slots[base + 1] = ta.recvAddr[d];
            slots[base + 2] = ta.selfOffset[d];
            slots[base + 3] = ta.sendBytes[d];
            slots[base + 4] = ta.goSize[d].addrOffset;
            slots[base + 5] = ta.goSize[d].loopParam;
            slots[base + 6] = ta.goSize[d].parallelParam;
            slots[base + 7] = ta.goSize[d].residual;
        }
        // else: zeros -> gate (sendBytes) == 0 -> CCU_IF skips desc
    }
    return slots;
}

// ---- Algorithm: declarative DSL using GroupBroadcastBatch ----

HcclResult CcuKernelAllGatherBatchMesh1DMs::Algorithm() {
    using namespace hcomm;
    using ccu_base::BroadcastItem;
    using ccu_base::GroupOpSize;
    using CcuRep::Variable;
    using CcuRep::LocalAddr;
    using CcuRep::RemoteAddr;

    const uint32_t numPeers = rankSize_ - 1;

    // Must match the hardware 64-way LoopGroup fan-out; see kMsParallelDim.
    // An earlier attempt at parallelDim=8 to reduce blockMsReq caused offset
    // drift between host-side slice layout and device-side LoopGroup fan-out,
    // producing the race-like failures seen in size_boundary.
    AllocGoResource(/*parallelDim=*/kMsParallelDim, /*msPerLoop=*/1);

    // ---- Per-desc Variables mirroring the GeneArgs slot layout ----
    Variable token = CreateVariable();
    Variable sendAddr   [MAX_DESC_COUNT];
    Variable recvAddr   [MAX_DESC_COUNT];
    Variable selfOffset [MAX_DESC_COUNT];
    Variable sendBytes  [MAX_DESC_COUNT];                     // serves as gate
    GroupOpSize goSize  [MAX_DESC_COUNT];

    for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
        sendAddr[d]   = CreateVariable();
        recvAddr[d]   = CreateVariable();
        selfOffset[d] = CreateVariable();
        sendBytes[d]  = CreateVariable();
        goSize[d]     = CreateGroupOpSize();
    }

    // ---- Channel-linked Variables: peer token + peer recvAddrs ----
    std::vector<Variable> peerToken(numPeers);
    std::vector<Variable> peerRecvAddr(numPeers * MAX_DESC_COUNT);
    for (uint32_t p = 0; p < numPeers; ++p) {
        CreateVariable(channels_[p], TOKEN_XN_ID, &peerToken[p]);
        for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
            CreateVariable(channels_[p], RECV_ADDR_XN_BASE + d,
                           &peerRecvAddr[p * MAX_DESC_COUNT + d]);
        }
    }

    // ---- LocalAddr templates (will be bound per desc) ----
    LocalAddr localSrc[MAX_DESC_COUNT];
    for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
        localSrc[d] = CreateLocalAddr();
    }

    // ---- Load SQE slots into Variables (order must match GeneArgs) ----
    Load(token);
    for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
        Load(sendAddr[d]);
        Load(recvAddr[d]);
        Load(selfOffset[d]);
        Load(sendBytes[d]);
        Load(goSize[d]);                                      // four Variables
    }

    // ---- PreSync: exchange recvAddr + token with all peers ----
    for (uint32_t p = 0; p < numPeers; ++p) {
        NotifyRecord(channels_[p], CKE_IDX, TOKEN_XN_ID,
                     token, 1u << TOKEN_XN_ID);
        for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
            NotifyRecord(channels_[p], CKE_IDX, RECV_ADDR_XN_BASE + d,
                         recvAddr[d], 1u << (RECV_ADDR_XN_BASE + d));
        }
    }
    for (uint32_t p = 0; p < numPeers; ++p) {
        NotifyWait(channels_[p], CKE_IDX, PRE_SYNC_MASK);
    }

    // ---- Build BroadcastItems for GroupBroadcastBatch ----
    //
    // Per desc: src = localSendBuf[d]; dst[0..numPeers-1] = peer recvBuf[d]
    // + selfOffset; dst[numPeers] = local recvBuf[d] + selfOffset (self-copy).
    std::vector<BroadcastItem> items;
    items.reserve(MAX_DESC_COUNT);
    for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
        // Bind src to sendAddr[d] with our RDMA token.
        localSrc[d].addr  = sendAddr[d];
        localSrc[d].token = token;

        // Build dst vector of size numPeers + 1.
        std::vector<RemoteAddr> dst;
        dst.reserve(numPeers + 1);

        // Peer slots: [0..numPeers-1]
        for (uint32_t p = 0; p < numPeers; ++p) {
            RemoteAddr r = CreateRemoteAddr();
            r.addr  = peerRecvAddr[p * MAX_DESC_COUNT + d];
            r.addr += selfOffset[d];
            r.token = peerToken[p];
            dst.push_back(r);
        }

        // Self slot: [numPeers]. CreateMultiOpBroadcastBatch reinterprets
        // this as a LocalAddr for LocalCopyNb, so the binary layout must
        // stay intact.
        RemoteAddr selfR = CreateRemoteAddr();
        selfR.addr  = recvAddr[d];
        selfR.addr += selfOffset[d];
        selfR.token = token;
        dst.push_back(selfR);

        items.push_back(BroadcastItem{localSrc[d], std::move(dst),
                                      goSize[d], sendBytes[d]});
    }

    // ---- Emit microcode for all active descs in one call ----
    HCCL_CHECK(GroupBroadcastBatch(channels_, items));

    // ---- PostSync: ensure all remote writes globally visible ----
    for (uint32_t p = 0; p < numPeers; ++p) {
        NotifyRecord(channels_[p], CKE_IDX, 1u << POST_SYNC_XN_ID);
    }
    for (uint32_t p = 0; p < numPeers; ++p) {
        NotifyWait(channels_[p], CKE_IDX, 1u << POST_SYNC_XN_ID);
    }

    return HCCL_SUCCESS;
}

// ============================================================
// Factory + exported register/launch entry points
// ============================================================

static std::unique_ptr<hcomm::CcuKernel>
CreateKernel(const hcomm::CcuKernelArg& arg) {
    return std::make_unique<CcuKernelAllGatherBatchMesh1DMs>(arg);
}

HcclResult RegisterBatchedAGKernelMs(
        HcclComm comm, CcuKernelHandle* handle,
        uint32_t rankId, uint32_t rankSize,
        const std::vector<ChannelHandle>& channels) {
    CC_LOG_INFO("[Ms] RegisterBatchedAGKernelMs: rank=%u/%u channels=%zu",
                rankId, rankSize, channels.size());
    CcuKernelArgBatchMs arg(rankSize, rankId, channels);
    hcomm::KernelCreator creator = CreateKernel;
    return HcclCcuKernelRegister(comm, handle, &creator, &arg);
}

HcclResult LaunchBatchedAGKernelMs(
        HcclComm comm, ThreadHandle thread, CcuKernelHandle kernel,
        const AllGatherBatchTaskArg& taskArg) {
    CcuTaskArgBatchMs ccuArg{};
    ccuArg.descCount = taskArg.descCount;

    for (uint32_t d = 0; d < taskArg.descCount; ++d) {
        uint64_t elemSize = DtypeSize(taskArg.descs[d].dataType);
        if (elemSize == 0 ||
            taskArg.descs[d].sendCount > UINT64_MAX / elemSize) {
            return HCCL_E_PARA;
        }
        uint64_t bytes = taskArg.descs[d].sendCount * elemSize;
        ccuArg.sendAddr[d]   = reinterpret_cast<uint64_t>(taskArg.descs[d].sendBuf);
        ccuArg.recvAddr[d]   = reinterpret_cast<uint64_t>(taskArg.descs[d].recvBuf);
        ccuArg.selfOffset[d] = static_cast<uint64_t>(taskArg.rankId) * bytes;
        ccuArg.sendBytes[d]  = bytes;
        // parallelDim here must match AllocGoResource(parallelDim) inside
        // Algorithm(); see kMsParallelDim below.
        ccuArg.goSize[d]     = bytes == 0
            ? GoSize{}
            : ::custom_comm::ccu_base::CalGoSize(bytes, kMsParallelDim);
    }

    // Token from the first active desc (skip zero-sized descs).
    for (uint32_t d = 0; d < taskArg.descCount; ++d) {
        if (ccuArg.sendBytes[d] > 0) {
            ccuArg.token = hcomm::CcuRep::GetTokenInfo(
                ccuArg.sendAddr[d], ccuArg.sendBytes[d]);
            break;
        }
    }

    return HcclCcuKernelLaunch(comm, thread, kernel, &ccuArg);
}

}  // namespace ms
}  // namespace custom_comm
