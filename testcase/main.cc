#include <acl/acl_rt.h>
#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "allgather_batch.h"

namespace {

constexpr uint32_t kMaxDescCount = 2;
constexpr size_t kMaxDeviceCount = 8;
constexpr uint64_t kDefaultTokenBytes = 320ULL * 1024ULL;
constexpr uint64_t kDefaultScaleCount = 128ULL;
constexpr uint32_t kCommOpExpansionModeCcuMs = 5;
constexpr uint32_t kCommOpExpansionModeCcuSched = 6;

struct TestOptions {
    std::vector<uint32_t> deviceList {0, 1};
    uint32_t descCount = 2;
    uint64_t tokenBytes = kDefaultTokenBytes;
    uint64_t scaleCount = kDefaultScaleCount;
    uint32_t warmupIters = 1;
    uint32_t measureIters = 1;
    bool verifyOutput = true;
};

struct BufferDesc {
    void *sendDevice = nullptr;
    void *recvDevice = nullptr;
    void *sendHost = nullptr;
    void *recvHost = nullptr;
    uint64_t sendCount = 0;
    size_t sendBytes = 0;
    size_t recvBytes = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
};

struct ThreadContext {
    const HcclRootInfo *rootInfo = nullptr;
    uint32_t logicalRank = 0;
    uint32_t physicalDevice = 0;
    uint32_t rankSize = 0;
    TestOptions options;
    std::atomic<int> *firstFailure = nullptr;
};

#define ACLCHECK_GOTO(call)                                                                  \
    do {                                                                                     \
        aclError aclRet = (call);                                                            \
        if (aclRet != ACL_SUCCESS) {                                                         \
            std::cerr << "acl call failed: " << #call << " at " << __FILE__ << ":"          \
                      << __LINE__ << ", ret=" << static_cast<int>(aclRet) << std::endl;      \
            status = 1;                                                                      \
            goto CLEANUP;                                                                    \
        }                                                                                    \
    } while (0)

#define HCCLCHECK_GOTO(call)                                                                 \
    do {                                                                                     \
        HcclResult hcclRet = (call);                                                         \
        if (hcclRet != HCCL_SUCCESS) {                                                       \
            std::cerr << "hccl call failed: " << #call << " at " << __FILE__ << ":"         \
                      << __LINE__ << ", ret=" << static_cast<int>(hcclRet) << std::endl;     \
            status = 1;                                                                      \
            goto CLEANUP;                                                                    \
        }                                                                                    \
    } while (0)

void PrintUsage(const char *prog)
{
    std::cout << "Usage: " << prog
              << " [--device-list 0,1] [--desc-count 1|2] [--bytes N]"
              << " [--scale-count N] [--warmup N] [--iters N] [--no-verify]"
              << std::endl;
    std::cout << "  --device-list   physical device ids, e.g. 4,5,6,7; logical ranks are list indexes" << std::endl;
    std::cout << "  --desc-count    1=INT8 token only, 2=INT8 token + FP32 scale; default 2" << std::endl;
    std::cout << "  --bytes         INT8 token bytes per rank, default 327680" << std::endl;
    std::cout << "  --scale-count   FP32 scale element count per rank, default 128" << std::endl;
    std::cout << "  --warmup        warmup iteration count, default 1" << std::endl;
    std::cout << "  --iters         measured iteration count, default 1" << std::endl;
    std::cout << "  --no-verify     skip host-side output verification" << std::endl;
}

bool ParseUint64(const char *value, uint64_t &result)
{
    if (value == nullptr || *value == '\0') {
        return false;
    }
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        return false;
    }
    result = static_cast<uint64_t>(parsed);
    return true;
}

bool ParseUint32(const char *value, uint32_t &result)
{
    uint64_t parsed = 0;
    if (!ParseUint64(value, parsed) || parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    result = static_cast<uint32_t>(parsed);
    return true;
}

bool ParseDeviceList(const char *value, std::vector<uint32_t> &devices)
{
    if (value == nullptr || *value == '\0') {
        return false;
    }
    std::vector<uint32_t> parsedDevices;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            return false;
        }
        uint32_t device = 0;
        if (!ParseUint32(token.c_str(), device)) {
            return false;
        }
        if (std::find(parsedDevices.begin(), parsedDevices.end(), device) != parsedDevices.end()) {
            return false;
        }
        parsedDevices.push_back(device);
    }
    if (parsedDevices.empty()) {
        return false;
    }
    devices.swap(parsedDevices);
    return true;
}

bool ParseArgs(int argc, char *argv[], TestOptions &options)
{
    for (int idx = 1; idx < argc; ++idx) {
        const std::string arg = argv[idx];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return false;
        }
        if (arg == "--no-verify") {
            options.verifyOutput = false;
            continue;
        }
        if (idx + 1 >= argc) {
            std::cerr << "missing value for argument: " << arg << std::endl;
            return false;
        }

        if (arg == "--device-list") {
            if (!ParseDeviceList(argv[++idx], options.deviceList)) {
                std::cerr << "invalid device list, expected e.g. 0,1 or 4,5,6,7" << std::endl;
                return false;
            }
        } else if (arg == "--desc-count") {
            if (!ParseUint32(argv[++idx], options.descCount) ||
                options.descCount == 0 || options.descCount > kMaxDescCount) {
                std::cerr << "invalid desc count, expected 1 or 2" << std::endl;
                return false;
            }
        } else if (arg == "--bytes") {
            if (!ParseUint64(argv[++idx], options.tokenBytes) || options.tokenBytes == 0) {
                std::cerr << "invalid token bytes" << std::endl;
                return false;
            }
        } else if (arg == "--scale-count") {
            if (!ParseUint64(argv[++idx], options.scaleCount)) {
                std::cerr << "invalid scale count" << std::endl;
                return false;
            }
        } else if (arg == "--warmup") {
            if (!ParseUint32(argv[++idx], options.warmupIters)) {
                std::cerr << "invalid warmup count" << std::endl;
                return false;
            }
        } else if (arg == "--iters") {
            if (!ParseUint32(argv[++idx], options.measureIters) || options.measureIters == 0) {
                std::cerr << "invalid iteration count" << std::endl;
                return false;
            }
        } else {
            std::cerr << "unknown argument: " << arg << std::endl;
            return false;
        }
    }

    if (options.descCount == 2 && options.scaleCount == 0) {
        std::cerr << "scale-count must be positive when desc-count is 2" << std::endl;
        return false;
    }
    if (options.deviceList.empty() || options.deviceList.size() > kMaxDeviceCount) {
        std::cerr << "device-list rank count must be in [1, " << kMaxDeviceCount << "]" << std::endl;
        return false;
    }
    return true;
}

void FillTokenInput(uint8_t *buf, uint64_t bytes, uint32_t logicalRank)
{
    for (uint64_t idx = 0; idx < bytes; ++idx) {
        buf[idx] = static_cast<uint8_t>((logicalRank * 17U + idx) & 0xffU);
    }
}

void FillScaleInput(float *buf, uint64_t count, uint32_t logicalRank)
{
    for (uint64_t idx = 0; idx < count; ++idx) {
        buf[idx] = static_cast<float>(logicalRank * 1000U + idx);
    }
}

bool VerifyTokenOutput(const uint8_t *buf, uint64_t bytes, uint32_t rankSize)
{
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        const uint8_t *rankBase = buf + static_cast<size_t>(rank) * bytes;
        for (uint64_t idx = 0; idx < bytes; ++idx) {
            const uint8_t expected = static_cast<uint8_t>((rank * 17U + idx) & 0xffU);
            if (rankBase[idx] != expected) {
                std::cerr << "token verify failed: srcRank=" << rank
                          << ", index=" << idx
                          << ", actual=" << static_cast<uint32_t>(rankBase[idx])
                          << ", expected=" << static_cast<uint32_t>(expected) << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool VerifyScaleOutput(const float *buf, uint64_t count, uint32_t rankSize)
{
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        const float *rankBase = buf + static_cast<size_t>(rank) * count;
        for (uint64_t idx = 0; idx < count; ++idx) {
            const float expected = static_cast<float>(rank * 1000U + idx);
            if (rankBase[idx] != expected) {
                std::cerr << "scale verify failed: srcRank=" << rank
                          << ", index=" << idx
                          << ", actual=" << rankBase[idx]
                          << ", expected=" << expected << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool CopyAndVerifyOutputs(const ThreadContext &ctx, const BufferDesc &token, const BufferDesc &scale)
{
    if (!ctx.options.verifyOutput) {
        return true;
    }

    aclError aclRet = aclrtMemcpy(token.recvHost, token.recvBytes,
        token.recvDevice, token.recvBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "copy token recv to host failed, ret=" << static_cast<int>(aclRet) << std::endl;
        return false;
    }
    if (!VerifyTokenOutput(static_cast<const uint8_t *>(token.recvHost),
        ctx.options.tokenBytes, ctx.rankSize)) {
        return false;
    }

    if (ctx.options.descCount == 2) {
        aclRet = aclrtMemcpy(scale.recvHost, scale.recvBytes,
            scale.recvDevice, scale.recvBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            std::cerr << "copy scale recv to host failed, ret=" << static_cast<int>(aclRet) << std::endl;
            return false;
        }
        if (!VerifyScaleOutput(static_cast<const float *>(scale.recvHost),
            ctx.options.scaleCount, ctx.rankSize)) {
            return false;
        }
    }
    return true;
}

bool CheckBufferSize(uint64_t elemCount, size_t elemSize, uint32_t rankSize, size_t &sendBytes, size_t &recvBytes)
{
    const auto maxSize = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    if (elemSize == 0 || elemCount > maxSize / elemSize) {
        return false;
    }
    const uint64_t oneRankBytes = elemCount * elemSize;
    if (rankSize != 0 && oneRankBytes > maxSize / rankSize) {
        return false;
    }
    sendBytes = static_cast<size_t>(oneRankBytes);
    recvBytes = static_cast<size_t>(oneRankBytes * rankSize);
    return true;
}

bool UseCcuCommConfig()
{
    const char *useCcu = std::getenv("CUSTOM_COMM_USE_CCU");
    return useCcu != nullptr && (std::strcmp(useCcu, "1") == 0 || std::strcmp(useCcu, "true") == 0);
}

uint32_t GetCcuCommExpansionMode()
{
    const char *ccuMode = std::getenv("CUSTOM_COMM_CCU_MODE");
    if (ccuMode != nullptr && (std::strcmp(ccuMode, "ms") == 0 || std::strcmp(ccuMode, "MS") == 0)) {
        return kCommOpExpansionModeCcuMs;
    }
    return kCommOpExpansionModeCcuSched;
}

int RunOnDevice(void *arg)
{
    ThreadContext *ctx = static_cast<ThreadContext *>(arg);
    int status = 0;

    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    BufferDesc token;
    BufferDesc scale;
    HcclAllGatherDesc descs[kMaxDescCount] = {};

    token.sendCount = ctx->options.tokenBytes;
    if (!CheckBufferSize(ctx->options.tokenBytes, sizeof(uint8_t), ctx->rankSize, token.sendBytes, token.recvBytes)) {
        std::cerr << "token buffer size overflow: bytes=" << ctx->options.tokenBytes
                  << ", rankSize=" << ctx->rankSize << std::endl;
        status = 1;
        goto CLEANUP;
    }
    token.dataType = HCCL_DATA_TYPE_INT8;

    if (ctx->options.descCount == 2) {
        scale.sendCount = ctx->options.scaleCount;
        if (!CheckBufferSize(ctx->options.scaleCount, sizeof(float), ctx->rankSize, scale.sendBytes, scale.recvBytes)) {
            std::cerr << "scale buffer size overflow: count=" << ctx->options.scaleCount
                      << ", rankSize=" << ctx->rankSize << std::endl;
            status = 1;
            goto CLEANUP;
        }
        scale.dataType = HCCL_DATA_TYPE_FP32;
    }

    ACLCHECK_GOTO(aclrtSetDevice(static_cast<int32_t>(ctx->physicalDevice)));
    if (UseCcuCommConfig()) {
        HcclCommConfig config;
        HcclCommConfigInit(&config);
        // Current hcomm implementation interprets 5 as CCU_MS and 6 as CCU_SCHED.
        config.hcclOpExpansionMode = GetCcuCommExpansionMode();
        HCCLCHECK_GOTO(HcclCommInitRootInfoConfig(ctx->rankSize, ctx->rootInfo, ctx->logicalRank, &config, &comm));
    } else {
        HCCLCHECK_GOTO(HcclCommInitRootInfo(ctx->rankSize, ctx->rootInfo, ctx->logicalRank, &comm));
    }
    ACLCHECK_GOTO(aclrtCreateStream(&stream));

    ACLCHECK_GOTO(aclrtMalloc(&token.sendDevice, token.sendBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK_GOTO(aclrtMalloc(&token.recvDevice, token.recvBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK_GOTO(aclrtMallocHost(&token.sendHost, token.sendBytes));
    ACLCHECK_GOTO(aclrtMallocHost(&token.recvHost, token.recvBytes));
    FillTokenInput(static_cast<uint8_t *>(token.sendHost), ctx->options.tokenBytes, ctx->logicalRank);
    ACLCHECK_GOTO(aclrtMemcpy(token.sendDevice, token.sendBytes,
        token.sendHost, token.sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    descs[0].sendBuf = token.sendDevice;
    descs[0].sendCount = token.sendCount;
    descs[0].dataType = token.dataType;
    descs[0].recvBuf = token.recvDevice;

    if (ctx->options.descCount == 2) {
        ACLCHECK_GOTO(aclrtMalloc(&scale.sendDevice, scale.sendBytes, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK_GOTO(aclrtMalloc(&scale.recvDevice, scale.recvBytes, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK_GOTO(aclrtMallocHost(&scale.sendHost, scale.sendBytes));
        ACLCHECK_GOTO(aclrtMallocHost(&scale.recvHost, scale.recvBytes));
        FillScaleInput(static_cast<float *>(scale.sendHost), ctx->options.scaleCount, ctx->logicalRank);
        ACLCHECK_GOTO(aclrtMemcpy(scale.sendDevice, scale.sendBytes,
            scale.sendHost, scale.sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        descs[1].sendBuf = scale.sendDevice;
        descs[1].sendCount = scale.sendCount;
        descs[1].dataType = scale.dataType;
        descs[1].recvBuf = scale.recvDevice;
    }

    for (uint32_t iter = 0; iter < ctx->options.warmupIters; ++iter) {
        HCCLCHECK_GOTO(HcclAllGatherBatch(descs, ctx->options.descCount, comm, stream));
        ACLCHECK_GOTO(aclrtSynchronizeDevice());
    }

    {
        const auto start = std::chrono::steady_clock::now();
        for (uint32_t iter = 0; iter < ctx->options.measureIters; ++iter) {
            HCCLCHECK_GOTO(HcclAllGatherBatch(descs, ctx->options.descCount, comm, stream));
            ACLCHECK_GOTO(aclrtSynchronizeDevice());
        }
        const auto end = std::chrono::steady_clock::now();
        const double totalUs = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        if (ctx->logicalRank == 0) {
            const double avgUs = totalUs / static_cast<double>(ctx->options.measureIters);
            const uint64_t perRankBytes = ctx->options.tokenBytes +
                ((ctx->options.descCount == 2) ? ctx->options.scaleCount * sizeof(float) : 0);
            const uint64_t totalBytes = perRankBytes * ctx->rankSize;
            constexpr double kUsToGBps = 1.0E6 / 1.0E9;
            const double bandwidthGBps = (avgUs > 0.0) ?
                static_cast<double>(totalBytes) / avgUs * kUsToGBps : 0.0;
            std::cout << "avgTime(us)=" << std::fixed << std::setprecision(2) << avgUs
                      << ", dataSize(B)=" << totalBytes
                      << ", algoBandwidth(GB/s)=" << std::setprecision(5) << bandwidthGBps
                      << std::endl;
        }
    }

    if (!CopyAndVerifyOutputs(*ctx, token, scale)) {
        status = 1;
        goto CLEANUP;
    }
    if (ctx->options.verifyOutput) {
        std::cout << "[rank " << ctx->logicalRank << ", device " << ctx->physicalDevice
                  << "] verify success" << std::endl;
    }

CLEANUP:
    if (comm != nullptr) {
        (void)HcclCommDestroy(comm);
    }
    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    if (token.sendDevice != nullptr) {
        (void)aclrtFree(token.sendDevice);
    }
    if (token.recvDevice != nullptr) {
        (void)aclrtFree(token.recvDevice);
    }
    if (token.sendHost != nullptr) {
        (void)aclrtFreeHost(token.sendHost);
    }
    if (token.recvHost != nullptr) {
        (void)aclrtFreeHost(token.recvHost);
    }
    if (scale.sendDevice != nullptr) {
        (void)aclrtFree(scale.sendDevice);
    }
    if (scale.recvDevice != nullptr) {
        (void)aclrtFree(scale.recvDevice);
    }
    if (scale.sendHost != nullptr) {
        (void)aclrtFreeHost(scale.sendHost);
    }
    if (scale.recvHost != nullptr) {
        (void)aclrtFreeHost(scale.recvHost);
    }
    (void)aclrtResetDevice(ctx->physicalDevice);

    if (status != 0 && ctx->firstFailure != nullptr) {
        int expected = 0;
        (void)ctx->firstFailure->compare_exchange_strong(expected, status);
    }
    return status;
}

}  // namespace

int main(int argc, char *argv[])
{
    TestOptions options;
    if (!ParseArgs(argc, argv, options)) {
        return 1;
    }

    aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "aclInit failed, ret=" << static_cast<int>(aclRet) << std::endl;
        return 1;
    }

    int status = 0;
    HcclRootInfo *rootInfo = nullptr;
    uint32_t deviceCount = 0;

    do {
        aclRet = aclrtGetDeviceCount(&deviceCount);
        if (aclRet != ACL_SUCCESS || deviceCount == 0) {
            std::cerr << "aclrtGetDeviceCount failed or no device found, ret="
                      << static_cast<int>(aclRet) << ", deviceCount=" << deviceCount << std::endl;
            status = 1;
            break;
        }
        for (uint32_t device : options.deviceList) {
            if (device >= deviceCount) {
                std::cerr << "device " << device << " is out of range, deviceCount=" << deviceCount << std::endl;
                status = 1;
                break;
            }
        }
        if (status != 0) {
            break;
        }

        aclRet = aclrtSetDevice(static_cast<int32_t>(options.deviceList[0]));
        if (aclRet != ACL_SUCCESS) {
            std::cerr << "aclrtSetDevice failed, ret=" << static_cast<int>(aclRet) << std::endl;
            status = 1;
            break;
        }

        aclRet = aclrtMallocHost(reinterpret_cast<void **>(&rootInfo), sizeof(HcclRootInfo));
        if (aclRet != ACL_SUCCESS || rootInfo == nullptr) {
            std::cerr << "aclrtMallocHost(rootInfo) failed, ret=" << static_cast<int>(aclRet) << std::endl;
            status = 1;
            break;
        }

        HcclResult hcclRet = HcclGetRootInfo(rootInfo);
        if (hcclRet != HCCL_SUCCESS) {
            std::cerr << "HcclGetRootInfo failed, ret=" << static_cast<int>(hcclRet) << std::endl;
            status = 1;
            break;
        }

        std::cout << "custom_comm HcclAllGatherBatch testcase starts"
                  << ", deviceList=";
        for (size_t idx = 0; idx < options.deviceList.size(); ++idx) {
            std::cout << (idx == 0 ? "" : ",") << options.deviceList[idx];
        }
        std::cout << ", rankSize=" << options.deviceList.size()
                  << ", descCount=" << options.descCount
                  << ", tokenBytes=" << options.tokenBytes
                  << ", scaleCount=" << options.scaleCount
                  << ", warmup=" << options.warmupIters
                  << ", iters=" << options.measureIters
                  << ", verify=" << (options.verifyOutput ? "on" : "off")
                  << std::endl;

        std::atomic<int> firstFailure(0);
        const uint32_t rankSize = static_cast<uint32_t>(options.deviceList.size());
        std::vector<std::thread> threads(rankSize);
        std::vector<ThreadContext> contexts(rankSize);
        for (uint32_t rank = 0; rank < rankSize; ++rank) {
            contexts[rank].rootInfo = rootInfo;
            contexts[rank].logicalRank = rank;
            contexts[rank].physicalDevice = options.deviceList[rank];
            contexts[rank].rankSize = rankSize;
            contexts[rank].options = options;
            contexts[rank].firstFailure = &firstFailure;
            threads[rank] = std::thread(RunOnDevice, static_cast<void *>(&contexts[rank]));
        }

        for (uint32_t rank = 0; rank < rankSize; ++rank) {
            threads[rank].join();
        }
        status = firstFailure.load();
    } while (false);

    if (rootInfo != nullptr) {
        (void)aclrtFreeHost(rootInfo);
    }
    if (!options.deviceList.empty()) {
        (void)aclrtResetDevice(options.deviceList[0]);
    }
    (void)aclFinalize();

    if (status != 0) {
        std::cerr << "custom_comm HcclAllGatherBatch testcase failed" << std::endl;
        return status;
    }

    std::cout << "custom_comm HcclAllGatherBatch testcase finished successfully" << std::endl;
    return 0;
}
