// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// FROZEN CONTRACT -- do not change after decomposed path ships.

#ifndef CUSTOM_COMM_ALLGATHER_BATCH_H_
#define CUSTOM_COMM_ALLGATHER_BATCH_H_

#include <cstdint>
#include <hccl/hccl_types.h>

// Forward-declare aclrtStream to avoid pulling in the full ACL runtime header.
// This prevents redefinition conflicts when torch_npu bundles its own ACL headers.
typedef void *aclrtStream;

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// C API data structures
// ============================================================

typedef struct {
    void        *sendBuf;   // device memory, non-null
    uint64_t     sendCount; // element count (not bytes)
    HcclDataType dataType;  // hccl_types.h:90-108
    void        *recvBuf;   // device memory, size >= sendCount * sizeof(dataType) * worldSize
} HcclAllGatherDesc;

// ============================================================
// C API entry point
// ============================================================

// Semantically equivalent to descCount independent HcclAllGather calls,
// but executed as a single operation (CCU path: single CCU kernel).
//
// Phase dispatch: if env CUSTOM_COMM_USE_CCU == "1", takes CCU path path;
// otherwise takes decomposed path (decomposed byte-packing) path.
//
// Constraints:
//   - 1 <= descCount <= MAX_DESC_COUNT (8)
//   - Each desc's sendBuf/recvBuf must be valid device memory
//   - comm must be an initialized HcclComm
//   - stream must be a valid aclrtStream
HcclResult HcclAllGatherBatch(
    const HcclAllGatherDesc *descs,
    uint32_t descCount,
    HcclComm comm,
    aclrtStream stream);

// Resolve the CCU EngineCtx slave stream for `comm`. The handle is owned by
// the CCU EngineCtx; callers must not free it. Initializes the CCU EngineCtx
// on first call. Used by torch_ext to register the slave stream into an
// active aclGraph capture so the CCU kernel runs in the captured graph.
HcclResult HcclAllGatherBatchGetCcuSlaveStream(
    HcclComm comm,
    aclrtStream *slaveStream);

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_COMM_ALLGATHER_BATCH_H_
