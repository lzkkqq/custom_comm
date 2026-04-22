// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CCU Kernel: Batched AllGather on Mesh-1D topology (CCU path, zero-copy).
//
// For each active descriptor, WriteNb to every remote rank's recvBuf and
// LocalCopyNb for the self-rank slot.  All descriptors are serialized
// within one CCU kernel launch -- no host-side resubmission.
//
// Reference: hccl/src/ops/all_gather/template/ccu/kernel/
//            ccu_kernel_all_gather_mesh1d_mem2mem.{h,cc}

#include "common.h"
#include "ccu_ms/go_size.h"
#include "log_util.h"

#include <hcomm/ccu/ccu_kernel.h>
#include <hcomm/ccu/ccu_kernel_signature.h>
#include <hcomm/ccu/ccu_kernel_arg.h>
#include <hcomm/ccu/ccu_task_arg_v1.h>
#include <hcomm/ccu/ccu_condition_v1.h>
#include <hcomm/ccu/ccu_assist_pub.h>
#include <hcomm/ccu/hccl_ccu_res.h>
#include <hcomm/ccu/ccu_loopblock_v1.h>
#include <hcomm/ccu/ccu_loopgroupcall_v1.h>
#include <hcomm/ccu/ccu_microcode_v1.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace custom_comm {
namespace ms {

// ============================================================
// XN slot layout per channel (for NotifyRecord/NotifyWait)
// ============================================================

static constexpr uint32_t TOKEN_XN_ID       = 0;
static constexpr uint32_t RECV_ADDR_XN_BASE = 1;   // 1..MAX_DESC_COUNT
static constexpr uint32_t POST_SYNC_XN_ID   = RECV_ADDR_XN_BASE + MAX_DESC_COUNT;  // 9
static constexpr uint32_t CKE_IDX           = 0;

// PreSync mask: bits 0..(MAX_DESC_COUNT), covering token + all recvAddr slots
static constexpr uint32_t PRE_SYNC_MASK = (1u << (MAX_DESC_COUNT + 1)) - 1;  // 0x1FF

// GeneArgs layout (Phase 2b-gamma: full HCCL GroupBroadcast pattern).
//
//  [0]                    token (RDMA access credential)
//  For d in [0, MAX_DESC_COUNT):
//    [1 + 8*d + 0]        sendAddr[d]
//    [1 + 8*d + 1]        recvAddr[d]        (rank-local recvBuf base)
//    [1 + 8*d + 2]        sendBytes[d]       (0 -> CCU_IF guard skips desc)
//    [1 + 8*d + 3]        selfOffset[d]      (rankId * sendBytes[d])
//    [1 + 8*d + 4]        goAddrOffset[d]   (main-loop skip in bytes)
//    [1 + 8*d + 5]        goLoopParam[d]    (packed main-loop loopCfg)
//    [1 + 8*d + 6]        goParallelParam[d](packed residual paraCfg)
//    [1 + 8*d + 7]        goResidual[d]     (residual lc0 sliceSize)
//
// 1 + 8 * MAX_DESC_COUNT words total.  With MAX_DESC_COUNT = 6 this is 49.

static constexpr uint32_t GENE_ARGS_PER_DESC  = 8;
static constexpr uint32_t GENE_ARGS_DESC_BASE = 1;
static constexpr uint32_t GENE_ARGS_TOTAL     =
    GENE_ARGS_DESC_BASE + GENE_ARGS_PER_DESC * MAX_DESC_COUNT;  // 1 + 8*MAX

// ============================================================
// CcuKernelArg subclass -- passed at registration time
// ============================================================

class CcuKernelArgBatchMs : public hcomm::CcuKernelArg {
public:
    CcuKernelArgBatchMs(uint32_t rankSize, uint32_t rankId,
                      std::vector<ChannelHandle> ch)
        : rankSize_(rankSize), rankId_(rankId)
    {
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
// CcuTaskArg subclass -- per-invocation data (host-side only)
// ============================================================

class CcuTaskArgBatchMs : public hcomm::CcuTaskArg {
public:
    uint32_t descCount;
    uint64_t token;
    uint64_t sendAddr[MAX_DESC_COUNT];
    uint64_t recvAddr[MAX_DESC_COUNT];
    uint64_t sendBytes[MAX_DESC_COUNT];
    uint64_t selfOffset[MAX_DESC_COUNT];   // rankId * sendBytes[d]
    // CalGoSize outputs used by the HCCL-style two-branch LoopGroupCall.
    uint64_t goAddrOffset[MAX_DESC_COUNT];     // = m * loopSize (main-branch prefix)
    uint64_t goLoopParam[MAX_DESC_COUNT];      // packed GetLoopParam for main branch
    uint64_t goParallelParam[MAX_DESC_COUNT];  // packed paraCfg for residual branch
    uint64_t goResidual[MAX_DESC_COUNT];       // residual-branch lc0 sliceSize
};

// ============================================================
// CCU Kernel class
// ============================================================

class CcuKernelAllGatherBatchMesh1DMs : public hcomm::CcuKernel {
public:
    explicit CcuKernelAllGatherBatchMesh1DMs(const hcomm::CcuKernelArg &arg);

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    uint32_t rankId_;
    uint32_t rankSize_;
};

// ---- Constructor ----

CcuKernelAllGatherBatchMesh1DMs::CcuKernelAllGatherBatchMesh1DMs(
        const hcomm::CcuKernelArg &arg)
    : hcomm::CcuKernel(arg)
{
    auto &ba = static_cast<const CcuKernelArgBatchMs &>(arg);
    rankId_   = ba.rankId_;
    rankSize_ = ba.rankSize_;
}

// ---- GeneArgs: serialize runtime parameters into uint64_t vector ----

std::vector<uint64_t>
CcuKernelAllGatherBatchMesh1DMs::GeneArgs(const hcomm::CcuTaskArg &arg) {
    auto &ta = static_cast<const CcuTaskArgBatchMs &>(arg);

    std::vector<uint64_t> slots(GENE_ARGS_TOTAL, 0);
    slots[0] = ta.token;

    for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
        const uint32_t base = GENE_ARGS_DESC_BASE + d * GENE_ARGS_PER_DESC;
        if (d < ta.descCount) {
            slots[base + 0] = ta.sendAddr[d];
            slots[base + 1] = ta.recvAddr[d];
            slots[base + 2] = ta.sendBytes[d];
            slots[base + 3] = ta.selfOffset[d];
            slots[base + 4] = ta.goAddrOffset[d];
            slots[base + 5] = ta.goLoopParam[d];
            slots[base + 6] = ta.goParallelParam[d];
            slots[base + 7] = ta.goResidual[d];
        }
        // else: zero-filled => sendBytes=0 => CCU_IF guard skips this desc
    }
    return slots;
}

// ---- Algorithm: generates CCU microcode at registration time ----
//
// Phase diagram:
//   InitResource -> LoadArgs -> PreSync(addr exchange)
//     -> DoAllGather(WriteNb + LocalCopyNb) -> PostSync(barrier)

HcclResult CcuKernelAllGatherBatchMesh1DMs::Algorithm() {
    using namespace hcomm;  // CCU_IF macro needs unqualified CcuRep::Condition
    using CcuRep::Variable;
    using CcuRep::LocalAddr;
    using CcuRep::RemoteAddr;
    using CcuRep::CompletedEvent;
    using CcuRep::CcuBuf;

    const uint32_t numPeers = rankSize_ - 1;

    // ================================================================
    // Phase 2b diagnostic: log the microcode-arg bit patterns CalGoSize
    // would produce for representative payload sizes, plus DSL call-site
    // constants.  Opt-in via CUSTOM_COMM_CCU_MS_DIAG=1 to avoid
    // polluting logs during normal runs.
    // ================================================================
    if (const char *diag = std::getenv("CUSTOM_COMM_CCU_MS_DIAG");
        diag && diag[0] == '1') {
        CC_LOG_INFO("[ms-diag] Algorithm() entered: rankId=%u rankSize=%u "
                    "numPeers=%u MAX_DESC=%u",
                    rankId_, rankSize_, numPeers, MAX_DESC_COUNT);
        CC_LOG_INFO("[ms-diag] constants: CCU_MS_SIZE=%llu CCU_MS_INTERLEAVE=%llu "
                    "CCU_MS_DEFAULT_LOOP_COUNT=%llu",
                    static_cast<unsigned long long>(kCcuMsSize),
                    static_cast<unsigned long long>(kCcuMsInterleave),
                    static_cast<unsigned long long>(kCcuMsDefaultLoopCount));

        // Probe CalGoSize output at canonical sizes so we can diff against
        // HCCL's values in a known-good mesh1d run on the same rank.
        for (uint64_t sz : {uint64_t{4096},        // 1 slot
                            uint64_t{32768},       // 8 slots
                            uint64_t{262144},      // 64 slots, 1 iter
                            uint64_t{524288},      // 128 slots, 2 iter
                            uint64_t{4 * 1024 * 1024}}) {  // 4 MiB
            GoSize g = CalGoSize(sz);
            CC_LOG_INFO("[ms-probe] CalGoSize(%llu) -> addrOffset=0x%llx "
                        "loopParam=0x%llx parallelParam=0x%llx residual=%llu",
                        static_cast<unsigned long long>(sz),
                        static_cast<unsigned long long>(g.addrOffset),
                        static_cast<unsigned long long>(g.loopParam),
                        static_cast<unsigned long long>(g.parallelParam),
                        static_cast<unsigned long long>(g.residual));
        }
        CC_LOG_INFO("[ms-diagnose] rank=%u rankSize=%u numPeers=%u "
                    "MAX_DESC_COUNT=%u; Algorithm about to emit DSL",
                    rankId_, rankSize_, numPeers, MAX_DESC_COUNT);
    }

    // ================================================================
    // InitResource: create all DSL resources upfront (before any
    // CCU_IF).  Each Create* call registers a resource slot in the
    // CcuRep IR.
    // ================================================================

    // --- GeneArgs Variables (1 + 6*MAX_DESC_COUNT = 37 total) ---
    Variable token = CreateVariable();
    Variable sendAddr[MAX_DESC_COUNT];
    Variable recvAddr[MAX_DESC_COUNT];
    Variable sendBytes[MAX_DESC_COUNT];
    Variable selfOffset[MAX_DESC_COUNT];
    Variable goAddrOffset[MAX_DESC_COUNT];
    Variable goLoopParam[MAX_DESC_COUNT];
    Variable goParallelParam[MAX_DESC_COUNT];
    Variable goResidual[MAX_DESC_COUNT];
    for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
        sendAddr[d]        = CreateVariable();
        recvAddr[d]        = CreateVariable();
        sendBytes[d]       = CreateVariable();
        selfOffset[d]      = CreateVariable();
        goAddrOffset[d]    = CreateVariable();
        goLoopParam[d]     = CreateVariable();
        goParallelParam[d] = CreateVariable();
        goResidual[d]      = CreateVariable();
    }

    // --- Channel-linked Variables: receive peer's token + recvAddrs ---
    // After NotifyWait, peerToken[p] holds peer p's RDMA token and
    // peerRecvAddr[p*MAX_DESC_COUNT+d] holds peer p's recvBuf[d] address.
    std::vector<Variable> peerToken(numPeers);
    std::vector<Variable> peerRecvAddr(numPeers * MAX_DESC_COUNT);
    for (uint32_t p = 0; p < numPeers; ++p) {
        CreateVariable(channels_[p], TOKEN_XN_ID, &peerToken[p]);
        for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
            CreateVariable(channels_[p], RECV_ADDR_XN_BASE + d,
                           &peerRecvAddr[p * MAX_DESC_COUNT + d]);
        }
    }

    // --- Addresses ---
    LocalAddr localSrc[MAX_DESC_COUNT];   // sendBuf for each desc
    LocalAddr selfDst[MAX_DESC_COUNT];    // recvBuf[rankId] for self-copy
    for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
        localSrc[d] = CreateLocalAddr();
        selfDst[d]  = CreateLocalAddr();
    }

    // remoteDst[p * MAX_DESC_COUNT + d]: peer p's recvBuf[d] + selfOffset
    std::vector<RemoteAddr> remoteDst;
    remoteDst.reserve(numPeers * MAX_DESC_COUNT);
    for (uint32_t p = 0; p < numPeers; ++p) {
        for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
            remoteDst.push_back(CreateRemoteAddr());
        }
    }

    // Single CompletedEvent tracks all WriteNb/LocalCopyNb completions
    CompletedEvent event = CreateCompletedEvent();

    // Staging area sized to the LoopGroup's hardware pipeline width.  Even
    // when loopIterNum == 1 the translator strides MS slots by
    // kCcuMsInterleave per iteration, so a single-slot CcuBuf leaves later
    // lanes (self-copy in particular) writing into uninitialized storage.
    // Matches HCCL's `CreateBlockCcuBuf(loopCount * msInterleave)` pattern
    // (see ccu_kernel_alg_base.cc CreateMultiOp{Broadcast,Reduce}*).
    // Two LoopBlocks share the staging ring: each gets kCcuMsInterleave
    // slots, matching HCCL's CreateMultiOp{Broadcast,Reduce} layout.
    constexpr uint32_t kNumBlocks = 2;
    std::vector<CcuBuf> stagingBuf(kNumBlocks * kCcuMsInterleave);
    CreateBlockCcuBuf(stagingBuf.size(), stagingBuf.data());
    std::vector<CcuRep::Executor> executors(kNumBlocks);
    CreateBlockExecutor(kNumBlocks, executors.data());

    // ================================================================
    // LoadArgs: bind each Variable to the next GeneArgs slot.
    // Order must exactly match GeneArgs encoding above.
    // ================================================================

    Load(token);
    for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
        Load(sendAddr[d]);
        Load(recvAddr[d]);
        Load(sendBytes[d]);
        Load(selfOffset[d]);
        Load(goAddrOffset[d]);
        Load(goLoopParam[d]);
        Load(goParallelParam[d]);
        Load(goResidual[d]);
    }

    // ================================================================
    // PreSync: exchange recvAddr + token with all peers.
    // Each rank pushes its local values via NotifyRecord, then waits
    // for all peers to push theirs via NotifyWait.
    // ================================================================

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

    // ================================================================
    // DoAllGather (Phase 2b-beta): Declare one LoopBlock whose body
    // stages a tile from HBM to CcuBuf, fans it out over WriteNb to
    // every peer, writes a self copy, and waits the batch.  Each desc
    // then triggers a LoopGroupCall that drives `loopIterNum[d]`
    // invocations of the block, with offsets advancing by `sliceBytes`.
    //
    //   HBM --LocalCopyNb--> CcuBuf
    //        --WriteNb-----> peer recvBuf[]   (fan-out)
    //        --LocalCopyNb-> self recvBuf     (rank-local slot)
    //        --WaitEvent---> drain the tile
    // ================================================================

    // --- Two LoopBlocks (ag_batch_bcast_0 / ag_batch_bcast_1) ---
    // HCCL-style: both blocks share the {src, peerDsts, self, sliceArg}
    // signature; the only structural difference is which CcuBuf-staging
    // slot lane the runtime uses.  Per-descriptor dispatch later chooses
    // which block(s) to instantiate via the LoopGroupCall paraCfg.
    std::array<LocalAddr, 2> srcArg;
    std::array<LocalAddr, 2> selfArg;
    std::array<Variable, 2>  sliceArg;
    std::array<std::vector<RemoteAddr>, 2> peerArg;
    for (uint32_t bi = 0; bi < 2; ++bi) {
        srcArg[bi]   = CreateLocalAddr();
        selfArg[bi]  = CreateLocalAddr();
        sliceArg[bi] = CreateVariable();
        peerArg[bi].resize(numPeers);
        for (uint32_t p = 0; p < numPeers; ++p) {
            peerArg[bi][p] = CreateRemoteAddr();
        }
    }

    static constexpr std::array<const char*, 2> kBlockNames = {
        "ag_batch_bcast_0", "ag_batch_bcast_1"
    };
    for (uint32_t bi = 0; bi < 2; ++bi) {
        CcuRep::LoopBlock block(this, kBlockNames[bi]);
        block(srcArg[bi], peerArg[bi], selfArg[bi], sliceArg[bi]);

        CcuBuf &stageSlot = stagingBuf[bi * kCcuMsInterleave];

        LocalCopyNb(stageSlot, srcArg[bi], sliceArg[bi], event);
        event.mask = 1u;
        WaitEvent(event);

        for (uint32_t p = 0; p < numPeers; ++p) {
            event.mask = 1u << p;
            WriteNb(channels_[p], peerArg[bi][p], stageSlot, sliceArg[bi], event);
        }

        event.mask = 1u << numPeers;
        LocalCopyNb(selfArg[bi], stagingBuf[bi * kCcuMsInterleave], sliceArg[bi], event);

        event.mask = (1u << (numPeers + 1)) - 1u;
        WaitEvent(event);
    }

    // ================================================================
    // Per-desc invocation: bind runtime addresses and fire LoopGroupCall.
    // ================================================================
    for (uint32_t d = 0; d < MAX_DESC_COUNT; ++d) {
        CCU_IF(sendBytes[d] != 0) {
            // Source: sendBuf[d]
            localSrc[d].addr  = sendAddr[d];
            localSrc[d].token = token;

            // Self dest: own recvBuf[d] + rankOffset
            selfDst[d].addr  = recvAddr[d];
            selfDst[d].addr += selfOffset[d];
            selfDst[d].token = token;

            // Peer dests: peerRecvAddr[p][d] + rankOffset.  Construct via
            // push_back so each slot is copy-initialized from a context-bearing
            // remoteDst element; a default-constructed RemoteAddr has null
            // context and any subsequent operator= would emit ASSIGN reps with
            // a null context, tripping AppendToContext().
            std::vector<RemoteAddr> peerDsts;
            peerDsts.reserve(numPeers);
            for (uint32_t p = 0; p < numPeers; ++p) {
                peerDsts.push_back(remoteDst[p * MAX_DESC_COUNT + d]);
                peerDsts[p].addr  = peerRecvAddr[p * MAX_DESC_COUNT + d];
                peerDsts[p].addr += selfOffset[d];
                peerDsts[p].token = peerToken[p];
            }

            // Memory-slice constant used by both LoopGroupCall branches.
            Variable memSliceVar = CreateVariable();
            memSliceVar = static_cast<uint64_t>(kCcuMsSize);

            Variable offCfg = CreateVariable();
            offCfg = GetOffsetParam(kCcuMsSize, kCcuMsInterleave, 1);

            // ---- Branch 1: main-loop (m * loopSize bytes) ----
            CCU_IF(goAddrOffset[d] != 0) {
                auto lcMain = Loop("ag_batch_bcast_0")(
                    localSrc[d], peerDsts, selfDst[d], memSliceVar);

                Variable loopCfgMain = CreateVariable();
                loopCfgMain = GetLoopParam(0, kCcuMsSize * kCcuMsDefaultLoopCount, 0);
                loopCfgMain += goLoopParam[d];

                Variable paraCfgMain = CreateVariable();
                paraCfgMain = GetParallelParam(kCcuMsDefaultLoopCount - 1, 0, 1);

                CcuRep::LoopGroupCall lgc(this, "ag_batch_bcast_main");
                lgc.Run({lcMain}, {loopCfgMain}, {executors[0]}, paraCfgMain, offCfg);
            }

            // ---- Branch 2: residual tail (n * memSlice + p bytes). ----
            CCU_IF(goParallelParam[d] != 0) {
                // Shift past the main-loop-consumed prefix.
                localSrc[d].addr += goAddrOffset[d];
                selfDst[d].addr  += goAddrOffset[d];
                for (uint32_t p = 0; p < numPeers; ++p) {
                    peerDsts[p].addr += goAddrOffset[d];
                }

                // lc0 handles the `p` bytes (residual < memSlice).
                auto lc0 = Loop("ag_batch_bcast_0")(
                    localSrc[d], peerDsts, selfDst[d], goResidual[d]);

                // Shift past residual to reach the `n * memSlice` chunks.
                localSrc[d].addr += goResidual[d];
                selfDst[d].addr  += goResidual[d];
                for (uint32_t p = 0; p < numPeers; ++p) {
                    peerDsts[p].addr += goResidual[d];
                }

                auto lc1 = Loop("ag_batch_bcast_1")(
                    localSrc[d], peerDsts, selfDst[d], memSliceVar);

                Variable loopCfg0 = CreateVariable();
                loopCfg0 = GetLoopParam(0, 0, 1);
                Variable loopCfg1 = CreateVariable();
                loopCfg1 = GetLoopParam(0, 0, 1);

                CcuRep::LoopGroupCall lgc(this, "agbatch_bcast_tail");
                lgc.Run({lc0, lc1}, {loopCfg0, loopCfg1},
                        {executors[0], executors[1]},
                        goParallelParam[d], offCfg);
            }
        }  // CCU_IF(sendBytes[d] != 0)
    }  // for d

    // ================================================================
    // PostSync: wait for all transfers, notify peers, drain the barrier.
    // ================================================================

    WaitEvent(event);
    for (uint32_t p = 0; p < numPeers; ++p) {
        NotifyRecord(channels_[p], CKE_IDX, 1u << POST_SYNC_XN_ID);
    }
    for (uint32_t p = 0; p < numPeers; ++p) {
        NotifyWait(channels_[p], CKE_IDX, 1u << POST_SYNC_XN_ID);
    }

    // TODO: AddCcuProfiling(...) disabled -- same bug as ccu_sched. Keep
    // disabled until the SDK call pattern is verified on Atlas.

    return HCCL_SUCCESS;
}

// ============================================================
// Factory function for CcuKernelRegister
// ============================================================

static std::unique_ptr<hcomm::CcuKernel>
CreateKernel(const hcomm::CcuKernelArg &arg) {
    return std::make_unique<CcuKernelAllGatherBatchMesh1DMs>(arg);
}

// ============================================================
// Exported helpers (called from engine_ctx.cc via forward decl)
// ============================================================

HcclResult RegisterBatchedAGKernelMs(
    HcclComm comm, CcuKernelHandle *handle,
    uint32_t rankId, uint32_t rankSize,
    const std::vector<ChannelHandle> &channels)
{
    if (const char *v = std::getenv("CUSTOM_COMM_CCU_MS_DIAG"); v && v[0] == '1') {
        CC_LOG_INFO("[MS] RegisterBatchedAGKernelMs: rankId=%u rankSize=%u channels=%zu",
                    rankId, rankSize, channels.size());
    }
    CcuKernelArgBatchMs arg(rankSize, rankId, channels);
    hcomm::KernelCreator creator = CreateKernel;
    return HcclCcuKernelRegister(comm, handle, &creator, &arg);
}

HcclResult LaunchBatchedAGKernelMs(
    HcclComm comm, ThreadHandle thread, CcuKernelHandle kernel,
    const AllGatherBatchTaskArg &taskArg)
{
    CcuTaskArgBatchMs ccuArg{};
    ccuArg.descCount = taskArg.descCount;

    const bool diag = []() {
        const char *v = std::getenv("CUSTOM_COMM_CCU_MS_DIAG");
        return v != nullptr && v[0] == '1';
    }();

    for (uint32_t d = 0; d < taskArg.descCount; ++d) {
        uint64_t elemSize = DtypeSize(taskArg.descs[d].dataType);
        if (elemSize == 0 ||
            taskArg.descs[d].sendCount > UINT64_MAX / elemSize) {
            return HCCL_E_PARA;
        }
        uint64_t bytes       = taskArg.descs[d].sendCount * elemSize;
        ccuArg.sendAddr[d]    = reinterpret_cast<uint64_t>(taskArg.descs[d].sendBuf);
        ccuArg.recvAddr[d]    = reinterpret_cast<uint64_t>(taskArg.descs[d].recvBuf);
        ccuArg.sendBytes[d]   = bytes;
        ccuArg.selfOffset[d]  = static_cast<uint64_t>(taskArg.rankId) * bytes;
        // Phase 2b-γ: feed the HCCL-style two-branch LoopGroupCall via CalGoSize.
        GoSize go = CalGoSize(bytes);
        ccuArg.goAddrOffset[d]    = go.addrOffset;
        ccuArg.goLoopParam[d]     = go.loopParam;
        ccuArg.goParallelParam[d] = go.parallelParam;
        ccuArg.goResidual[d]      = go.residual;

        if (diag) {
            GoSize gs = CalGoSize(bytes);
            CC_LOG_INFO("[ms-diag] desc=%u bytes=%llu -> goSize{addrOff=0x%llx "
                        "loopParam=0x%llx parallelParam=0x%llx residual=%llu}",
                        d, static_cast<unsigned long long>(bytes),
                        static_cast<unsigned long long>(gs.addrOffset),
                        static_cast<unsigned long long>(gs.loopParam),
                        static_cast<unsigned long long>(gs.parallelParam),
                        static_cast<unsigned long long>(gs.residual));
        }
    }

    // RDMA token from first active desc (skip zero-length descs to avoid size=0)
    for (uint32_t d = 0; d < taskArg.descCount; ++d) {
        if (ccuArg.sendBytes[d] > 0) {
            ccuArg.token = hcomm::CcuRep::GetTokenInfo(
                ccuArg.sendAddr[d], ccuArg.sendBytes[d]);
            break;
        }
    }

    if (diag) {
        CC_LOG_INFO("[MS] launching kernel: rank=%u descCount=%u token=0x%llx",
                    taskArg.rankId, taskArg.descCount,
                    static_cast<unsigned long long>(ccuArg.token));
    }

    return HcclCcuKernelLaunch(comm, thread, kernel, &ccuArg);
}


}  // namespace ms
}  // namespace custom_comm
