// Copyright (c) 2026 custom_comm Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Plog (CANN slog) wrappers for custom_comm host-side code.
//
// On Linux with a CANN SDK the macros expand to dlog_* and route to
// ~/ascend/log/debug/plog/, respecting ASCEND_GLOBAL_LOG_LEVEL.
// Without slog.h (e.g. local macOS syntax check) they fall back to fprintf
// so the code still compiles and prints.

#ifndef OPS_ALLGATHER_BATCH_INC_LOG_UTIL_H_
#define OPS_ALLGATHER_BATCH_INC_LOG_UTIL_H_

#if defined(__has_include)
#  if __has_include(<toolchain/slog.h>)
#    include <toolchain/slog.h>
#    define CUSTOM_COMM_HAS_SLOG 1
#  endif
#endif

#ifndef CUSTOM_COMM_HAS_SLOG
#  define CUSTOM_COMM_HAS_SLOG 0
#  include <cstdio>
#endif

// Module id HCCL (=3 in CANN log_types.h) — keeps custom_comm logs alongside
// the HCCL runtime's own plog output.
#define CUSTOM_COMM_LOG_MODULE 3

#if CUSTOM_COMM_HAS_SLOG
#define CC_LOG_DEBUG(fmt, ...) \
    dlog_debug(CUSTOM_COMM_LOG_MODULE, "[custom_comm] " fmt, ##__VA_ARGS__)
#define CC_LOG_INFO(fmt, ...) \
    dlog_info(CUSTOM_COMM_LOG_MODULE, "[custom_comm] " fmt, ##__VA_ARGS__)
#define CC_LOG_WARN(fmt, ...) \
    dlog_warn(CUSTOM_COMM_LOG_MODULE, "[custom_comm] " fmt, ##__VA_ARGS__)
#define CC_LOG_ERROR(fmt, ...) \
    dlog_error(CUSTOM_COMM_LOG_MODULE, "[custom_comm] " fmt, ##__VA_ARGS__)
#else
#define CC_LOG_DEBUG(fmt, ...) ((void)0)
#define CC_LOG_INFO(fmt, ...)  ((void)0)
#define CC_LOG_WARN(fmt, ...) \
    std::fprintf(stderr, "[custom_comm][WARN] " fmt "\n", ##__VA_ARGS__)
#define CC_LOG_ERROR(fmt, ...) \
    std::fprintf(stderr, "[custom_comm][ERROR] " fmt "\n", ##__VA_ARGS__)
#endif

#endif  // OPS_ALLGATHER_BATCH_INC_CUSTOM_COMM_LOG_H_
