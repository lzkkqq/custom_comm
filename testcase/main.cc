#include <acl/acl_rt.h>
#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
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
#include "common.h"

namespace {

constexpr size_t kMaxDeviceCount = 8;
constexpr uint64_t kDefaultTokenBytes = 320ULL * 1024ULL;
constexpr uint64_t kDefaultScaleCount = 128ULL;
constexpr uint32_t kCommOpExpansionModeCcuMs = 5;
constexpr uint32_t kCommOpExpansionModeCcuSched = 6;

enum class BenchmarkMode {
    kCustom,
    kBaseline,
    kBoth,
};

enum class TimingMode {
    kHost,
    kDevice,
};

struct ItemSpec {
    std::string typeName;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    uint64_t sendCount = 0;
};

std::vector<ItemSpec> BuildDefaultItemSpecs()
{
    return {
        {"int8", HCCL_DATA_TYPE_INT8, kDefaultTokenBytes},
        {"fp32", HCCL_DATA_TYPE_FP32, kDefaultScaleCount},
    };
}

struct TestOptions {
    std::vector<uint32_t> deviceList {0, 1};
    std::vector<ItemSpec> items = BuildDefaultItemSpecs();
    uint32_t warmupIters = 1;
    uint32_t measureIters = 1;
    bool verifyOutput = true;
    BenchmarkMode mode = BenchmarkMode::kCustom;
    TimingMode timingMode = TimingMode::kDevice;
};

struct ItemBuffer {
    ItemSpec spec;
    void *sendDevice = nullptr;
    void *recvDevice = nullptr;
    void *sendHost = nullptr;
    void *recvHost = nullptr;
    size_t sendBytes = 0;
    size_t recvBytes = 0;
};

struct PackedBuffer {
    void *sendDevice = nullptr;
    void *recvDevice = nullptr;
    void *sendHost = nullptr;
    void *recvHost = nullptr;
    uint64_t sendCount = 0;
    size_t sendBytes = 0;
    size_t recvBytes = 0;
};

struct ThreadContext {
    const HcclRootInfo *rootInfo = nullptr;
    uint32_t logicalRank = 0;
    uint32_t physicalDevice = 0;
    uint32_t rankSize = 0;
    TestOptions options;
    std::atomic<int> *firstFailure = nullptr;
};

struct BenchmarkResult {
    unsigned long long dataSize = 0;
    double averageTimeUs = 0.0;
    double algorithmBandwidthGBps = 0.0;
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

const char *ToModeString(BenchmarkMode mode)
{
    switch (mode) {
        case BenchmarkMode::kCustom:
            return "custom";
        case BenchmarkMode::kBaseline:
            return "baseline";
        case BenchmarkMode::kBoth:
            return "both";
        default:
            return "unknown";
    }
}

const char *ToTimingModeString(TimingMode mode)
{
    switch (mode) {
        case TimingMode::kHost:
            return "host";
        case TimingMode::kDevice:
            return "device";
        default:
            return "unknown";
    }
}

void PrintUsage(const char *prog)
{
    std::cout << "Usage: " << prog
              << " [--device-list 0,1] [--item dtype:count]..."
              << " [--warmup N] [--iters N]"
              << " [--mode custom|baseline|both]"
              << " [--timing-mode host|device] [--no-verify]"
              << std::endl;
    std::cout << "  --device-list   physical device ids, e.g. 4,5,6,7; logical ranks are list indexes" << std::endl;
    std::cout << "  --item          repeatable item spec, e.g. int8:327680 or fp32:128; default int8:327680 fp32:128" << std::endl;
    std::cout << "  --warmup        warmup iteration count, default 1" << std::endl;
    std::cout << "  --iters         measured iteration count, default 1" << std::endl;
    std::cout << "  --mode          custom, baseline, or both; default custom" << std::endl;
    std::cout << "  --timing-mode   host or device; default device" << std::endl;
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

bool ParseMode(const char *value, BenchmarkMode &mode)
{
    if (value == nullptr) {
        return false;
    }
    const std::string modeStr(value);
    if (modeStr == "custom") {
        mode = BenchmarkMode::kCustom;
        return true;
    }
    if (modeStr == "baseline") {
        mode = BenchmarkMode::kBaseline;
        return true;
    }
    if (modeStr == "both") {
        mode = BenchmarkMode::kBoth;
        return true;
    }
    return false;
}

bool ParseTimingMode(const char *value, TimingMode &mode)
{
    if (value == nullptr) {
        return false;
    }
    const std::string modeStr(value);
    if (modeStr == "host") {
        mode = TimingMode::kHost;
        return true;
    }
    if (modeStr == "device") {
        mode = TimingMode::kDevice;
        return true;
    }
    return false;
}

bool ParseItemType(const std::string &value, std::string &typeName, HcclDataType &dataType)
{
    if (value == "int8") {
        typeName = "int8";
        dataType = HCCL_DATA_TYPE_INT8;
        return true;
    }
    if (value == "uint8") {
        typeName = "uint8";
        dataType = HCCL_DATA_TYPE_UINT8;
        return true;
    }
    if (value == "int16") {
        typeName = "int16";
        dataType = HCCL_DATA_TYPE_INT16;
        return true;
    }
    if (value == "uint16") {
        typeName = "uint16";
        dataType = HCCL_DATA_TYPE_UINT16;
        return true;
    }
    if (value == "fp16") {
        typeName = "fp16";
        dataType = HCCL_DATA_TYPE_FP16;
        return true;
    }
    if (value == "bf16") {
        typeName = "bf16";
        dataType = HCCL_DATA_TYPE_BFP16;
        return true;
    }
    if (value == "int32") {
        typeName = "int32";
        dataType = HCCL_DATA_TYPE_INT32;
        return true;
    }
    if (value == "uint32") {
        typeName = "uint32";
        dataType = HCCL_DATA_TYPE_UINT32;
        return true;
    }
    if (value == "fp32") {
        typeName = "fp32";
        dataType = HCCL_DATA_TYPE_FP32;
        return true;
    }
    if (value == "int64") {
        typeName = "int64";
        dataType = HCCL_DATA_TYPE_INT64;
        return true;
    }
    if (value == "uint64") {
        typeName = "uint64";
        dataType = HCCL_DATA_TYPE_UINT64;
        return true;
    }
    if (value == "fp64") {
        typeName = "fp64";
        dataType = HCCL_DATA_TYPE_FP64;
        return true;
    }
    if (value == "int128") {
        typeName = "int128";
        dataType = HCCL_DATA_TYPE_INT128;
        return true;
    }
    if (value == "hif8") {
        typeName = "hif8";
        dataType = HCCL_DATA_TYPE_HIF8;
        return true;
    }
    if (value == "fp8e4m3") {
        typeName = "fp8e4m3";
        dataType = HCCL_DATA_TYPE_FP8E4M3;
        return true;
    }
    if (value == "fp8e5m2") {
        typeName = "fp8e5m2";
        dataType = HCCL_DATA_TYPE_FP8E5M2;
        return true;
    }
    if (value == "fp8e8m0") {
        typeName = "fp8e8m0";
        dataType = HCCL_DATA_TYPE_FP8E8M0;
        return true;
    }
    return false;
}

bool ParseItemSpec(const char *value, ItemSpec &item)
{
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string itemSpec(value);
    const size_t colonPos = itemSpec.find(':');
    if (colonPos == std::string::npos || colonPos == 0 || colonPos + 1 >= itemSpec.size()) {
        return false;
    }

    std::string typeName;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    if (!ParseItemType(itemSpec.substr(0, colonPos), typeName, dataType)) {
        return false;
    }

    uint64_t sendCount = 0;
    if (!ParseUint64(itemSpec.c_str() + colonPos + 1, sendCount) || sendCount == 0) {
        return false;
    }

    item.typeName = typeName;
    item.dataType = dataType;
    item.sendCount = sendCount;
    return true;
}

bool CheckBufferSize(uint64_t elemCount, uint64_t elemSize, uint32_t rankSize, size_t &sendBytes, size_t &recvBytes)
{
    const uint64_t maxSize = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
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

bool ParseArgs(int argc, char *argv[], TestOptions &options)
{
    bool itemListOverridden = false;
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
        } else if (arg == "--item") {
            if (!itemListOverridden) {
                options.items.clear();
                itemListOverridden = true;
            }
            ItemSpec item;
            if (!ParseItemSpec(argv[++idx], item)) {
                std::cerr << "invalid item spec, expected dtype:count, e.g. int8:327680 or fp32:128" << std::endl;
                return false;
            }
            options.items.push_back(item);
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
        } else if (arg == "--mode") {
            if (!ParseMode(argv[++idx], options.mode)) {
                std::cerr << "invalid mode, expected custom|baseline|both" << std::endl;
                return false;
            }
        } else if (arg == "--timing-mode") {
            if (!ParseTimingMode(argv[++idx], options.timingMode)) {
                std::cerr << "invalid timing mode, expected host|device" << std::endl;
                return false;
            }
        } else {
            std::cerr << "unknown argument: " << arg << std::endl;
            return false;
        }
    }

    if (options.items.empty() || options.items.size() > MAX_DESC_COUNT) {
        std::cerr << "item count must be in [1, " << MAX_DESC_COUNT << "]" << std::endl;
        return false;
    }
    if (options.deviceList.empty() || options.deviceList.size() > kMaxDeviceCount) {
        std::cerr << "device-list rank count must be in [1, " << kMaxDeviceCount << "]" << std::endl;
        return false;
    }
    return true;
}

uint8_t BuildExpectedByte(uint32_t logicalRank, uint32_t itemIndex, size_t byteOffset)
{
    const uint32_t foldedOffset = static_cast<uint32_t>(byteOffset & 0xffU);
    return static_cast<uint8_t>((logicalRank * 29U + itemIndex * 53U + foldedOffset) & 0xffU);
}

void FillItemInput(void *buf, size_t bytes, uint32_t logicalRank, uint32_t itemIndex)
{
    auto *out = static_cast<uint8_t *>(buf);
    for (size_t idx = 0; idx < bytes; ++idx) {
        out[idx] = BuildExpectedByte(logicalRank, itemIndex, idx);
    }
}

void BuildPackedExpectedSegment(const std::vector<ItemBuffer> &items, uint32_t logicalRank, std::vector<uint8_t> &segment)
{
    size_t cursor = 0;
    for (size_t itemIdx = 0; itemIdx < items.size(); ++itemIdx) {
        const size_t bytes = items[itemIdx].sendBytes;
        for (size_t offset = 0; offset < bytes; ++offset) {
            segment[cursor + offset] = BuildExpectedByte(logicalRank, static_cast<uint32_t>(itemIdx), offset);
        }
        cursor += bytes;
    }
}

bool VerifyCustomOutputs(const ThreadContext &ctx, std::vector<ItemBuffer> &items)
{
    if (!ctx.options.verifyOutput) {
        return true;
    }

    for (size_t itemIdx = 0; itemIdx < items.size(); ++itemIdx) {
        ItemBuffer &item = items[itemIdx];
        aclError aclRet = aclrtMemcpy(item.recvHost, item.recvBytes,
            item.recvDevice, item.recvBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            std::cerr << "copy custom recv to host failed, item=" << itemIdx
                      << ", ret=" << static_cast<int>(aclRet) << std::endl;
            return false;
        }

        const auto *recvBytes = static_cast<const uint8_t *>(item.recvHost);
        for (uint32_t srcRank = 0; srcRank < ctx.rankSize; ++srcRank) {
            const uint8_t *rankBase = recvBytes + static_cast<size_t>(srcRank) * item.sendBytes;
            for (size_t offset = 0; offset < item.sendBytes; ++offset) {
                const uint8_t expected = BuildExpectedByte(srcRank, static_cast<uint32_t>(itemIdx), offset);
                if (rankBase[offset] != expected) {
                    std::cerr << "custom verify failed: item=" << itemIdx
                              << ", type=" << item.spec.typeName
                              << ", srcRank=" << srcRank
                              << ", byteOffset=" << offset
                              << ", actual=" << static_cast<uint32_t>(rankBase[offset])
                              << ", expected=" << static_cast<uint32_t>(expected) << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

bool VerifyBaselineOutput(const ThreadContext &ctx, const std::vector<ItemBuffer> &items, PackedBuffer &baseline)
{
    if (!ctx.options.verifyOutput) {
        return true;
    }

    aclError aclRet = aclrtMemcpy(baseline.recvHost, baseline.recvBytes,
        baseline.recvDevice, baseline.recvBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "copy baseline recv to host failed, ret=" << static_cast<int>(aclRet) << std::endl;
        return false;
    }

    std::vector<uint8_t> expectedSegment(baseline.sendBytes, 0);
    const auto *recvBytes = static_cast<const uint8_t *>(baseline.recvHost);
    for (uint32_t srcRank = 0; srcRank < ctx.rankSize; ++srcRank) {
        BuildPackedExpectedSegment(items, srcRank, expectedSegment);
        const uint8_t *rankBase = recvBytes + static_cast<size_t>(srcRank) * baseline.sendBytes;
        for (size_t offset = 0; offset < baseline.sendBytes; ++offset) {
            if (rankBase[offset] != expectedSegment[offset]) {
                std::cerr << "baseline verify failed: srcRank=" << srcRank
                          << ", byteOffset=" << offset
                          << ", actual=" << static_cast<uint32_t>(rankBase[offset])
                          << ", expected=" << static_cast<uint32_t>(expectedSegment[offset]) << std::endl;
                return false;
            }
        }
    }
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

bool NeedCustomPath(BenchmarkMode mode)
{
    return mode == BenchmarkMode::kCustom || mode == BenchmarkMode::kBoth;
}

bool NeedBaselinePath(BenchmarkMode mode)
{
    return mode == BenchmarkMode::kBaseline || mode == BenchmarkMode::kBoth;
}

bool NeedDeviceTiming(TimingMode mode)
{
    return mode == TimingMode::kDevice;
}

template <typename LaunchFn>
bool RunBenchmark(const char *modeLabel,
    const ThreadContext &ctx,
    aclrtStream stream,
    aclrtStream syncStream,
    aclrtEvent startEvent,
    aclrtEvent endEvent,
    aclrtEvent syncEvent,
    unsigned long long dataSize,
    LaunchFn launchFn,
    BenchmarkResult &result)
{
    auto checkAcl = [](aclError aclRet, const char *file, int line) -> bool {
        if (aclRet != ACL_SUCCESS) {
            std::cerr << "acl call failed at " << file << ":" << line
                      << ", ret=" << static_cast<int>(aclRet) << std::endl;
            return false;
        }
        return true;
    };

    auto checkHccl = [](HcclResult hcclRet, const char *file, int line) -> bool {
        if (hcclRet != HCCL_SUCCESS) {
            std::cerr << "hccl call failed at " << file << ":" << line
                      << ", ret=" << static_cast<int>(hcclRet) << std::endl;
            return false;
        }
        return true;
    };

    if (NeedDeviceTiming(ctx.options.timingMode)) {
        if (!checkAcl(aclrtStreamWaitEvent(stream, syncEvent), __FILE__, __LINE__)) {
            return false;
        }
        if (!checkAcl(aclrtResetEvent(syncEvent, stream), __FILE__, __LINE__)) {
            return false;
        }
    }

    for (uint32_t iter = 0; iter < ctx.options.warmupIters; ++iter) {
        if (!checkHccl(launchFn(), __FILE__, __LINE__)) {
            return false;
        }
    }

    if (!checkAcl(aclrtRecordEvent(startEvent, stream), __FILE__, __LINE__)) {
        return false;
    }
    for (uint32_t iter = 0; iter < ctx.options.measureIters; ++iter) {
        if (!checkHccl(launchFn(), __FILE__, __LINE__)) {
            return false;
        }
    }
    if (!checkAcl(aclrtRecordEvent(endEvent, stream), __FILE__, __LINE__)) {
        return false;
    }

    if (NeedDeviceTiming(ctx.options.timingMode)) {
        const int sleepMs = 50 +
            static_cast<int>(ctx.options.warmupIters) * 2 +
            static_cast<int>(ctx.options.measureIters) * 2;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
        if (!checkAcl(aclrtRecordEvent(syncEvent, syncStream), __FILE__, __LINE__)) {
            return false;
        }
    }

    if (!checkAcl(aclrtSynchronizeStream(stream), __FILE__, __LINE__)) {
        return false;
    }

    float elapsedMs = 0.0F;
    if (!checkAcl(aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent), __FILE__, __LINE__)) {
        return false;
    }

    result.dataSize = dataSize;
    result.averageTimeUs = static_cast<double>(elapsedMs * 1000.0F) / ctx.options.measureIters;
    constexpr double kBytesPerUsToGBps = 1.0E6 / 1.0E9;
    result.algorithmBandwidthGBps = (result.averageTimeUs > 0.0) ?
        static_cast<double>(dataSize) / result.averageTimeUs * kBytesPerUsToGBps : 0.0;

    if (ctx.logicalRank == 0) {
        std::cout << std::left << std::setw(10) << modeLabel
                  << " | " << std::right << std::setw(17) << result.dataSize
                  << " | " << std::setw(14) << std::fixed << std::setprecision(2) << result.averageTimeUs
                  << " | " << std::setw(20) << std::fixed << std::setprecision(5) << result.algorithmBandwidthGBps
                  << " | success" << std::endl;
    }
    return true;
}

std::string FormatItemList(const std::vector<ItemSpec> &items)
{
    std::ostringstream oss;
    for (size_t idx = 0; idx < items.size(); ++idx) {
        if (idx != 0) {
            oss << ",";
        }
        oss << items[idx].typeName << ":" << items[idx].sendCount;
    }
    return oss.str();
}

uint64_t GetTotalSendBytes(const std::vector<ItemBuffer> &items)
{
    uint64_t totalBytes = 0;
    for (const auto &item : items) {
        totalBytes += item.sendBytes;
    }
    return totalBytes;
}

int RunOnDevice(void *arg)
{
    ThreadContext *ctx = static_cast<ThreadContext *>(arg);
    int status = 0;

    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    aclrtStream syncStream = nullptr;
    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    aclrtEvent syncEvent = nullptr;
    PackedBuffer baseline;
    BenchmarkResult customResult = {};
    BenchmarkResult baselineResult = {};
    std::vector<ItemBuffer> items(ctx->options.items.size());
    std::vector<HcclAllGatherDesc> descs(ctx->options.items.size());

    ACLCHECK_GOTO(aclrtSetDevice(static_cast<int32_t>(ctx->physicalDevice)));
    if (UseCcuCommConfig()) {
        HcclCommConfig config;
        HcclCommConfigInit(&config);
        config.hcclOpExpansionMode = GetCcuCommExpansionMode();
        HCCLCHECK_GOTO(HcclCommInitRootInfoConfig(ctx->rankSize, ctx->rootInfo, ctx->logicalRank, &config, &comm));
    } else {
        HCCLCHECK_GOTO(HcclCommInitRootInfo(ctx->rankSize, ctx->rootInfo, ctx->logicalRank, &comm));
    }
    ACLCHECK_GOTO(aclrtCreateStream(&stream));
    ACLCHECK_GOTO(aclrtCreateEvent(&startEvent));
    ACLCHECK_GOTO(aclrtCreateEvent(&endEvent));
    if (NeedDeviceTiming(ctx->options.timingMode)) {
        ACLCHECK_GOTO(aclrtCreateStream(&syncStream));
        ACLCHECK_GOTO(aclrtCreateEventWithFlag(&syncEvent, ACL_EVENT_SYNC));
    }

    for (size_t itemIdx = 0; itemIdx < ctx->options.items.size(); ++itemIdx) {
        ItemBuffer &item = items[itemIdx];
        item.spec = ctx->options.items[itemIdx];
        if (!CheckBufferSize(item.spec.sendCount,
                DtypeSize(item.spec.dataType),
                ctx->rankSize,
                item.sendBytes,
                item.recvBytes)) {
            std::cerr << "item buffer size overflow: item=" << itemIdx
                      << ", type=" << item.spec.typeName
                      << ", count=" << item.spec.sendCount
                      << ", rankSize=" << ctx->rankSize << std::endl;
            status = 1;
            goto CLEANUP;
        }

        ACLCHECK_GOTO(aclrtMallocHost(&item.sendHost, item.sendBytes));
        FillItemInput(item.sendHost, item.sendBytes, ctx->logicalRank, static_cast<uint32_t>(itemIdx));

        if (NeedCustomPath(ctx->options.mode)) {
            ACLCHECK_GOTO(aclrtMalloc(&item.sendDevice, item.sendBytes, ACL_MEM_MALLOC_HUGE_ONLY));
            ACLCHECK_GOTO(aclrtMalloc(&item.recvDevice, item.recvBytes, ACL_MEM_MALLOC_HUGE_ONLY));
            ACLCHECK_GOTO(aclrtMemcpy(item.sendDevice, item.sendBytes,
                item.sendHost, item.sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));
            if (ctx->options.verifyOutput) {
                ACLCHECK_GOTO(aclrtMallocHost(&item.recvHost, item.recvBytes));
            }

            descs[itemIdx].sendBuf = item.sendDevice;
            descs[itemIdx].sendCount = item.spec.sendCount;
            descs[itemIdx].dataType = item.spec.dataType;
            descs[itemIdx].recvBuf = item.recvDevice;
        }
    }

    if (NeedBaselinePath(ctx->options.mode)) {
        const uint64_t totalSendBytes = GetTotalSendBytes(items);
        if (!CheckBufferSize(totalSendBytes, 1, ctx->rankSize, baseline.sendBytes, baseline.recvBytes)) {
            std::cerr << "baseline buffer size overflow: bytes=" << totalSendBytes
                      << ", rankSize=" << ctx->rankSize << std::endl;
            status = 1;
            goto CLEANUP;
        }

        baseline.sendCount = totalSendBytes;
        ACLCHECK_GOTO(aclrtMallocHost(&baseline.sendHost, baseline.sendBytes));
        size_t cursor = 0;
        for (const auto &item : items) {
            std::memcpy(static_cast<uint8_t *>(baseline.sendHost) + cursor, item.sendHost, item.sendBytes);
            cursor += item.sendBytes;
        }

        ACLCHECK_GOTO(aclrtMalloc(&baseline.sendDevice, baseline.sendBytes, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK_GOTO(aclrtMalloc(&baseline.recvDevice, baseline.recvBytes, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK_GOTO(aclrtMemcpy(baseline.sendDevice, baseline.sendBytes,
            baseline.sendHost, baseline.sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        if (ctx->options.verifyOutput) {
            ACLCHECK_GOTO(aclrtMallocHost(&baseline.recvHost, baseline.recvBytes));
        }
    }

    {
        const unsigned long long dataSize =
            static_cast<unsigned long long>(GetTotalSendBytes(items) * ctx->rankSize);
        auto launchCustom = [&]() -> HcclResult {
            return HcclAllGatherBatch(descs.data(), static_cast<uint32_t>(descs.size()), comm, stream);
        };
        auto launchBaseline = [&]() -> HcclResult {
            return HcclAllGather(baseline.sendDevice,
                baseline.recvDevice,
                baseline.sendCount,
                HCCL_DATA_TYPE_INT8,
                comm,
                stream);
        };

        if (NeedCustomPath(ctx->options.mode)) {
            if (!RunBenchmark("custom",
                    *ctx,
                    stream,
                    syncStream,
                    startEvent,
                    endEvent,
                    syncEvent,
                    dataSize,
                    launchCustom,
                    customResult)) {
                status = 1;
                goto CLEANUP;
            }
            if (!VerifyCustomOutputs(*ctx, items)) {
                status = 1;
                goto CLEANUP;
            }
        }

        if (NeedBaselinePath(ctx->options.mode)) {
            if (!RunBenchmark("baseline",
                    *ctx,
                    stream,
                    syncStream,
                    startEvent,
                    endEvent,
                    syncEvent,
                    dataSize,
                    launchBaseline,
                    baselineResult)) {
                status = 1;
                goto CLEANUP;
            }
            if (!VerifyBaselineOutput(*ctx, items, baseline)) {
                status = 1;
                goto CLEANUP;
            }
        }

        if (ctx->logicalRank == 0 && ctx->options.mode == BenchmarkMode::kBoth) {
            const double deltaUs = baselineResult.averageTimeUs - customResult.averageTimeUs;
            const double speedup = (customResult.averageTimeUs > 0.0) ?
                (baselineResult.averageTimeUs / customResult.averageTimeUs) : 0.0;
            std::cout << "compare    | delta(us)=" << std::fixed << std::setprecision(2) << deltaUs
                      << ", speedup=" << std::setprecision(5) << speedup << "x" << std::endl;
        }
    }

    if (ctx->options.verifyOutput) {
        std::cout << "[rank " << ctx->logicalRank << ", device " << ctx->physicalDevice
                  << "] verify success" << std::endl;
    }

CLEANUP:
    if (startEvent != nullptr) {
        (void)aclrtDestroyEvent(startEvent);
    }
    if (endEvent != nullptr) {
        (void)aclrtDestroyEvent(endEvent);
    }
    if (syncEvent != nullptr) {
        (void)aclrtDestroyEvent(syncEvent);
    }
    if (syncStream != nullptr) {
        (void)aclrtDestroyStream(syncStream);
    }
    if (comm != nullptr) {
        (void)HcclCommDestroy(comm);
    }
    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    for (auto &item : items) {
        if (item.sendDevice != nullptr) {
            (void)aclrtFree(item.sendDevice);
        }
        if (item.recvDevice != nullptr) {
            (void)aclrtFree(item.recvDevice);
        }
        if (item.sendHost != nullptr) {
            (void)aclrtFreeHost(item.sendHost);
        }
        if (item.recvHost != nullptr) {
            (void)aclrtFreeHost(item.recvHost);
        }
    }
    if (baseline.sendDevice != nullptr) {
        (void)aclrtFree(baseline.sendDevice);
    }
    if (baseline.recvDevice != nullptr) {
        (void)aclrtFree(baseline.recvDevice);
    }
    if (baseline.sendHost != nullptr) {
        (void)aclrtFreeHost(baseline.sendHost);
    }
    if (baseline.recvHost != nullptr) {
        (void)aclrtFreeHost(baseline.recvHost);
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
                  << ", items=" << FormatItemList(options.items)
                  << ", warmup=" << options.warmupIters
                  << ", iters=" << options.measureIters
                  << ", mode=" << ToModeString(options.mode)
                  << ", timingMode=" << ToTimingModeString(options.timingMode)
                  << ", verify=" << (options.verifyOutput ? "on" : "off")
                  << std::endl;
        std::cout << "mode       | dataSize(B)        | avgTime(us)    | algoBandwidth(GB/s)  | status" << std::endl;

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
