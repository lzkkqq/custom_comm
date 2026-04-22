// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CCU path: CCU context management -- one-time registration + cached launch.
//
// InitCcuContext: HcclEngineCtxCreate -> HcclChannelAcquire -> CcuKernelRegister
//                 -> HcclThreadAcquire -> KernelRegisterFinish  (first call)
//                 HcclEngineCtxGet (subsequent calls, cached)
//
// LaunchCcuKernel: HcclEngineCtxGet -> LaunchBatchedAGKernel{Ms, MsV2}
//
// CUSTOM_COMM_CCU_MS_IMPL env (unset/v1 -> v1 hand-rolled kernel;
// v2 -> GroupBroadcastBatch-based kernel). V1 and V2 live in separate
// EngineCtx slots so they don't cross-contaminate a cached comm.

#include "engine_ctx_ms.h"
#include "ccu_kernel_ag_batch_mesh1d_ms.h"
#include "common.h"
#include "log_util.h"

#include <hccl/hccl_comm.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res.h>
#include <hcomm/ccu/ccu_kernel.h>
#include <hcomm/ccu/hccl_ccu_res.h>

#include <acl/acl.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace custom_comm {
namespace ms {

// ============================================================
// MsImpl selection (CUSTOM_COMM_CCU_MS_IMPL env var)
// ============================================================

enum class MsImpl : uint8_t {
    kV1 = 0,
    kV2 = 1,
};

// Read-per-call: mirrors UseCcuPath() in all_gather_batch.cc so tests that
// flip the env mid-process (e.g. test_ccu_ms_size_boundary) observe the new
// value on the next invocation. Cheap: one getenv() + a couple of strcmp().
static MsImpl GetMsImpl() {
    const char *env = std::getenv("CUSTOM_COMM_CCU_MS_IMPL");
    if (env == nullptr || env[0] == '\0') {
        return MsImpl::kV1;
    }
    if (std::strcmp(env, "v1") == 0 || std::strcmp(env, "V1") == 0) {
        return MsImpl::kV1;
    }
    if (std::strcmp(env, "v2") == 0 || std::strcmp(env, "V2") == 0) {
        return MsImpl::kV2;
    }
    CC_LOG_ERROR("CUSTOM_COMM_CCU_MS_IMPL=%s not recognized; falling back to V1",
                 env);
    return MsImpl::kV1;
}

static const char *CtxTagFor(MsImpl impl) {
    return impl == MsImpl::kV2 ? "custom_comm_ag_batch_ms_v2"
                               : "custom_comm_ag_batch_ms";
}

// ============================================================
// CcuContext -- stored in HcclEngineCtx, cached per (comm, tag)
// ============================================================

// XN IDs used: TOKEN(0), RECV_ADDR(1..MAX_DESC_COUNT), POST_SYNC(MAX_DESC_COUNT+1)
static constexpr uint32_t NOTIFY_COUNT = 1 + MAX_DESC_COUNT + 1;  // 10

struct CcuContext {
    CcuKernelHandle kernelHandle{};
    ThreadHandle    threadHandle{};
    MsImpl          impl = MsImpl::kV1;
    bool            initialized = false;
};

// ============================================================
// InitCcuContext
// ============================================================

HcclResult InitCcuContext(HcclComm comm) {
    // Fast path: return cached context
    if (comm == nullptr) {
        CC_LOG_ERROR("InitCcuContext: comm is null");
        return HCCL_E_PARA;
    }
    const MsImpl impl = GetMsImpl();
    const char  *tag  = CtxTagFor(impl);

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, tag, COMM_ENGINE_CCU,
                         &ctx, &ctxSize) == HCCL_SUCCESS && ctx != nullptr) {
        auto *ccuCtx = static_cast<CcuContext *>(ctx);
        if (ccuCtx->initialized) {
            CC_LOG_DEBUG("InitCcuContext: cached ctx hit (impl=V%d)",
                         static_cast<int>(impl) + 1);
            return HCCL_SUCCESS;
        }
        // Partial init from a prior failed attempt: SDK resource cleanup
        // semantics are opaque, so retrying is unsafe.  Surface the error.
        CC_LOG_ERROR("InitCcuContext: partial context detected; refusing retry");
        return HCCL_E_INTERNAL;
    }

    // Slow path: first call -- allocate + register
    CC_LOG_INFO("InitCcuContext: first-time registration (impl=V%d, tag=%s)",
                static_cast<int>(impl) + 1, tag);

    // 1. Create engine context slot
    HCCL_CHECK(HcclEngineCtxCreate(comm, tag, COMM_ENGINE_CCU,
                                   sizeof(CcuContext), &ctx));
    auto *ccuCtx = static_cast<CcuContext *>(ctx);
    ccuCtx->impl = impl;

    // 2. Get topology info
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    HCCL_CHECK(HcclGetRankId(comm, &rankId));
    HCCL_CHECK(HcclGetRankSize(comm, &rankSize));

    if (rankSize < 2) {
        CC_LOG_ERROR("InitCcuContext: rankSize=%u too small for CCU (need >=2)", rankSize);
        return HCCL_E_PARA;
    }

    const uint32_t numPeers = rankSize - 1;


    static constexpr CommProtocol CCU_PROTO_PRIORITY[] = {
        COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UBC_TP
    };

    std::vector<HcclChannelDesc> channelDescs;
    for (uint32_t r = 0; r < rankSize; ++r) {
        if (r == rankId) continue;
        uint32_t linkNum = 0;
        CommLink *links = nullptr;
        HCCL_CHECK(HcclRankGraphGetLinks(comm, /*netLayer=*/0, rankId, r, &links, &linkNum));

        // Pick the highest-priority protocol actually present for this peer.
        CommProtocol chosenProto = COMM_PROTOCOL_RESERVED;
        bool found = false;
        for (CommProtocol p : CCU_PROTO_PRIORITY) {
            for (uint32_t i = 0; i < linkNum; ++i) {
                if (links[i].linkAttr.linkProtocol == p) {
                    chosenProto = p;
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        if (!found) {
            CC_LOG_ERROR("InitCcuContext: no UBC_CTP/UBC_TP link to rank %u", r);
            return HCCL_E_NOT_SUPPORT;
        }

        for (uint32_t i = 0; i < linkNum; ++i) {
            if (links[i].linkAttr.linkProtocol != chosenProto) continue;
            HcclChannelDesc desc{};
            HcclChannelDescInit(&desc, 1);
            desc.remoteRank              = r;
            desc.notifyNum               = 16;
            desc.memHandles              = nullptr;
            desc.memHandleNum            = 0;
            // Mirror CreateChannelFromLink: copy only protocol / commAddr / loc.
            desc.localEndpoint.protocol  = links[i].srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr  = links[i].srcEndpointDesc.commAddr;
            desc.localEndpoint.loc       = links[i].srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = links[i].dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = links[i].dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc      = links[i].dstEndpointDesc.loc;
            desc.channelProtocol         = chosenProto;
            channelDescs.push_back(desc);
        }
    }

    std::vector<ChannelHandle> channels(channelDescs.size());
    HCCL_CHECK(HcclChannelAcquire(comm, COMM_ENGINE_CCU,
                                  channelDescs.data(), channelDescs.size(),
                                  channels.data()));

    // 4. Register CCU kernel (compiles Algorithm -> microcode IR)
    if (impl == MsImpl::kV2) {
        HCCL_CHECK(RegisterBatchedAGKernelMsV2(comm, &ccuCtx->kernelHandle,
                                               rankId, rankSize, channels));
    } else {
        HCCL_CHECK(RegisterBatchedAGKernelMs(comm, &ccuCtx->kernelHandle,
                                             rankId, rankSize, channels));
    }

    // 5. Acquire CCU thread with enough notification slots
    HCCL_CHECK(HcclThreadAcquire(comm, COMM_ENGINE_CCU,
                                  1, NOTIFY_COUNT,
                                  &ccuCtx->threadHandle));

    // 6. Finalize: CcuKernelMgr translates IR to hardware microcode
    HCCL_CHECK(HcclCcuKernelRegisterFinish(comm));

    ccuCtx->initialized = true;
    CC_LOG_INFO("InitCcuContext: ready (impl=V%d, rank=%u/%u, peers=%u)",
                static_cast<int>(impl) + 1, rankId, rankSize, numPeers);
    return HCCL_SUCCESS;
}

// ============================================================
// LaunchCcuKernel
// ============================================================

HcclResult LaunchCcuKernel(HcclComm comm, const void *taskArg) {
    if (comm == nullptr || taskArg == nullptr) {
        CC_LOG_ERROR("LaunchCcuKernel: null comm or taskArg");
        return HCCL_E_PARA;
    }
    auto *arg = static_cast<const AllGatherBatchTaskArg *>(taskArg);
    CC_LOG_INFO("LaunchCcuKernel: descCount=%u rank=%u/%u",
                arg->descCount, arg->rankId, arg->rankSize);
    if (arg->descCount == 0 || arg->descCount > MAX_DESC_COUNT) {
        CC_LOG_ERROR("LaunchCcuKernel: descCount=%u out of range (max=%u)",
                     arg->descCount, MAX_DESC_COUNT);
        return HCCL_E_PARA;
    }

    const MsImpl impl = GetMsImpl();
    const char  *tag  = CtxTagFor(impl);

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    HCCL_CHECK(HcclEngineCtxGet(comm, tag, COMM_ENGINE_CCU,
                                &ctx, &ctxSize));
    if (ctx == nullptr || ctxSize < sizeof(CcuContext)) {
        CC_LOG_ERROR("LaunchCcuKernel: invalid ctx (ptr=%p, size=%llu, tag=%s)",
                     ctx, static_cast<unsigned long long>(ctxSize), tag);
        return HCCL_E_INTERNAL;
    }
    auto *ccuCtx = static_cast<CcuContext *>(ctx);
    if (!ccuCtx->initialized) {
        CC_LOG_ERROR("LaunchCcuKernel: ctx not initialized (did InitCcuContext succeed?)");
        return HCCL_E_INTERNAL;
    }

    if (ccuCtx->impl == MsImpl::kV2) {
        return LaunchBatchedAGKernelMsV2(comm, ccuCtx->threadHandle,
                                         ccuCtx->kernelHandle, *arg);
    }
    return LaunchBatchedAGKernelMs(comm, ccuCtx->threadHandle,
                                   ccuCtx->kernelHandle, *arg);
}

// ============================================================
// GetCcuThreadHandle -- expose thread handle for aclGraph capture
// ============================================================

HcclResult GetCcuThreadHandle(HcclComm comm, uint64_t *threadHandle) {
    if (comm == nullptr || threadHandle == nullptr) {
        CC_LOG_ERROR("GetCcuThreadHandle: null comm or out-param");
        return HCCL_E_PARA;
    }
    const MsImpl impl = GetMsImpl();
    const char  *tag  = CtxTagFor(impl);

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    HCCL_CHECK(HcclEngineCtxGet(comm, tag, COMM_ENGINE_CCU,
                                &ctx, &ctxSize));
    if (ctx == nullptr || ctxSize < sizeof(CcuContext)) {
        CC_LOG_ERROR("GetCcuThreadHandle: ctx missing or too small (tag=%s)", tag);
        return HCCL_E_INTERNAL;
    }
    auto *ccuCtx = static_cast<CcuContext *>(ctx);
    if (!ccuCtx->initialized || ccuCtx->threadHandle == 0) {
        CC_LOG_ERROR("GetCcuThreadHandle: ctx not initialized");
        return HCCL_E_INTERNAL;
    }
    *threadHandle = ccuCtx->threadHandle;
    return HCCL_SUCCESS;
}

}  // namespace ms
}  // namespace custom_comm
