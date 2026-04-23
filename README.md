# custom_comm

`custom_comm` 是面向 Ascend NPU 的自定义高性能通信算子工程。当前核心算子是
`allgather_batch`：把多个 dtype/shape 不同的 tensor 合并成一次批量 AllGather 调用，
减少多次独立通信带来的 host 调度、HCCL launch 和中间拷贝开销。

工程基于 HCCL/HComm 实现底层通信，并通过 `torch_npu` 注册为 PyTorch custom op，
支持 eager 模式、`torch.compile`/torchair 图模式、aclGraph capture/replay，以及基础
profiling/DFX 能力。

## 目录结构

```text
custom_comm/
  ops/
    allgather_batch/
      inc/                  # 对外 C API，extern "C" + POD 参数
      src/                  # ABI=0 shim 实现，连接 HCCL/HComm/CCU
  torch_ext/csrc/           # ABI=1 PyTorch/torch_npu binding
  python/custom_comm/       # Python 包入口、converter 注册、生成的 so
  tests/                    # smoke、ABI、eager、graph、benchmark 测试
  setup.py                  # Python 扩展构建入口
  CMakeLists.txt            # 仅构建 C++ 库时使用
```

## 架构设计

### 双 so 与 ABI 防火墙

工程会生成两个共享库：

```text
python/custom_comm/
  libcustom_comm_impl.so   # ABI=0，连接 libhccl/libhcomm/libascendcl
  _C.cpython-<ver>.so      # ABI=1，PyTorch/torch_npu extension
```

拆成两个库是为了隔离 C++ ABI。PyTorch wheel 通常使用新的 libstdc++
CXX11 ABI，也就是 `_GLIBCXX_USE_CXX11_ABI=1`；CANN/HComm 侧通常使用旧 ABI，
也就是 `_GLIBCXX_USE_CXX11_ABI=0`。如果把 torch 侧和 hcomm 侧放在同一个 so
里，`std::string`、`std::vector`、`std::list` 等 C++ 对象跨 ABI 边界传递时
可能发生内存破坏。

两个 so 之间只允许通过 `ops/<op>/inc/*.h` 里声明的 `extern "C"` 接口通信。
接口参数必须是 POD 类型、裸指针或 opaque pointer，不能暴露 C++ STL 类型。

### 添加新算子

新增算子时按下面三类文件组织，`setup.py` 会自动扫描：

| 文件 | 作用 | ABI |
| --- | --- | --- |
| `ops/<op>/inc/<op>.h` | 对外 C API，`extern "C"`，只放 POD/opaque pointer | ABI 无关 |
| `ops/<op>/src/**/*.cc` | shim 实现，可包含 HCCL/HComm/CCU 头文件 | ABI=0 |
| `torch_ext/csrc/<op>.cpp` | PyTorch binding、dispatcher 注册、pybind11 入口 | ABI=1 |

约束：

- `ops/<op>/src` 可以包含 hcomm/ccu C++ 头文件，但不能包含 torch/ATen 头文件。
- `torch_ext/csrc` 可以使用 torch、c10、ATen，但只能包含 `ops/<op>/inc/*.h`
  暴露的 C API。
- `vendor`、`build`、`__pycache__` 等目录不会参与源码扫描。
- 加完文件后重新执行 `pip install -e .` 即可触发构建。

## allgather_batch

### Python API

```python
import custom_comm

outputs = custom_comm.allgather_batch(inputs, hcom, world_size)
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `inputs` | 非空 tensor 列表；每个 tensor 至少 1 维，当前 eager 路径要求非空、contiguous |
| `hcom` | HCCL communicator 标识；当前实现优先支持 `get_hccl_comm(rank)` 返回的 raw handle 字符串，也兼容 legacy group name |
| `world_size` | 通信域 rank 数，必须大于 0 |
| 返回值 | 输出 tensor 列表；每个输出的第 0 维扩大为 `input.shape[0] * world_size` |

### C API

底层 C API 定义在 `ops/allgather_batch/inc/allgather_batch.h`：

```cpp
typedef struct {
    void        *sendBuf;
    uint64_t     sendCount;
    HcclDataType dataType;
    void        *recvBuf;
} HcclAllGatherDesc;

HcclResult HcclAllGatherBatch(
    const HcclAllGatherDesc *descs,
    uint32_t descCount,
    HcclComm comm,
    aclrtStream stream);
```

当前运行时代码中的 `MAX_DESC_COUNT` 为 6，因此 `descCount` 有效范围是
`1 <= descCount <= 6`。如果后续修改 `ops/allgather_batch/inc/common.h` 中的
`MAX_DESC_COUNT`，README、测试用例和 graph/meta 覆盖也需要同步更新。

### 执行路径

默认路径是 decomposed path：

- 把多个输入 tensor 按字节打包成一个连续 buffer。
- 调用一次 `HcclAllGather` 完成跨 rank 聚合。
- 再把聚合后的连续结果拆回每个用户输出 tensor。
- 优点是通用性强，适合作为 fallback 和正确性基线。

CCU 路径通过环境变量启用：

```bash
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 your_script.py
```

CCU backend 由 `CUSTOM_COMM_CCU_MODE` 选择：

| 环境变量 | 行为 |
| --- | --- |
| 未设置、空、`sched`、`SCHED` | 使用 CCU SCHED backend |
| `ms`、`MS` | 使用 CCU MS backend |
| 其他值 | 打印 fallback 日志并退回 CCU SCHED |

`CUSTOM_COMM_USE_CCU` 当前只识别 `1` 和 `true`。例如 `True` 不会启用 CCU 路径。

## 环境依赖

- Ascend NPU，推荐 Atlas A5/Ascend 950 类环境。
- CANN 9.0+，需要 HCCL/HComm/ACL 运行库和头文件。
- Python 3.10+。
- PyTorch 2.9.0。
- torch_npu 2.9.0。
- pytest，用于运行测试。

依赖版本以 `requirements.txt` 为准。

## 安装与构建

### Python 包安装

在 `custom_comm` 目录下执行：

```bash
source ~/Ascend/set_env.sh
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` 会先构建 ABI=0 的 `libcustom_comm_impl.so`，再构建 ABI=1 的
`custom_comm._C` 扩展。

### 仅构建 C++ 库

```bash
cmake -B build
cmake --build build
```

日常 Python/torch_npu 调试建议使用 `pip install -e .`，因为它会同时构建 shim
和 PyTorch extension，并把产物放到 Python 包目录。

## 快速开始

示例脚本：

```python
import torch
import torch.distributed as dist
import torch_npu
import custom_comm

dist.init_process_group(backend="hccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.npu.set_device(rank)
device = torch.device(f"npu:{rank}")

pg = dist.distributed_c10d._get_default_group()
backend = pg._get_backend(device)
if hasattr(backend, "get_hccl_comm"):
    hcom = str(backend.get_hccl_comm(rank))
else:
    hcom = backend.get_hccl_comm_name(rank)

data = torch.randint(0, 127, (2048,), dtype=torch.int8, device=device)
scale = torch.randn(56, dtype=torch.float32, device=device)
ids = torch.randint(0, 8, (8,), dtype=torch.int32, device=device)

out_data, out_scale, out_ids = custom_comm.allgather_batch(
    [data, scale, ids], hcom, world_size
)

torch.npu.synchronize()
dist.destroy_process_group()
```

运行：

```bash
torchrun --nproc_per_node=8 example.py
```

如果机器不足 8 卡，把 `--nproc_per_node` 改成实际可用 rank 数。

## Graph 模式

### torch.compile / torchair

`custom_comm` 会注册 GE converter。使用 `torch.compile(backend="torchair")` 时，
`allgather_batch` 可以被 lowering 到 GE IR：

```python
@torch.compile(backend="torchair")
def fused_gather(x, s, ids, hcom, ws):
    return custom_comm.allgather_batch([x, s, ids], hcom, ws)
```

当前 converter 会把 `allgather_batch` 拆成多个 `HcomAllGather` GE op，保持每个
tensor 的 dtype 和 shape 语义。

### aclGraph capture/replay

NPU eager binding 中对 direct HCCL capture 做了 stream/event 同步 workaround：
HCCL 调用会被派发到缓存的通信 stream，再通过 event 与当前 compute stream 同步。
这样可以支持 `torch.npu.NPUGraph` 的 capture 和 replay。

## Profiling 与 DFX

当前工程提供以下观测手段：

| 能力 | 说明 |
| --- | --- |
| `RECORD_FUNCTION` | PyTorch profiler 中可看到 `custom_comm::allgather_batch` |
| plog/slog | runtime 日志会进入 Ascend plog，包含路径选择、参数、错误码等信息 |
| `aclprofMarkEx` | CANN profiler marker，可在 Ascend Insight 中辅助定位 |
| CCU stream event | CCU 路径可通过 slave stream event 观察 device 侧耗时 |

常用日志环境变量：

| 环境变量 | 作用 |
| --- | --- |
| `ASCEND_GLOBAL_LOG_LEVEL` | `0=DEBUG`、`1=INFO`、`2=WARN`、`3=ERROR` |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 设置为 `1` 时同时打印到 stderr，便于开发调试 |
| `ASCEND_PROCESS_LOG_PATH` | 覆盖默认日志目录 |

查看 custom_comm 相关 plog：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
tail -F ~/ascend/log/debug/plog/plog-*.log | grep custom_comm
```

## 测试

测试入口位于 `tests/`。推荐在 `custom_comm` 目录下执行下面命令。

### 1. Smoke 测试

验证 NPU、HCCL、AllGather、AllReduce、Broadcast 基础链路：

```bash
torchrun --nproc_per_node=8 tests/smoke_test.py
```

可以显式指定 HCCL expansion mode：

```bash
torchrun --nproc_per_node=8 tests/smoke_test.py \
  --expansion-mode CCU_MS \
  --fallback-mode AICPU_TS
```

说明：

- `--expansion-mode` 用于 AllGather 子通信域，是 custom_comm 关注的路径。
- `--fallback-mode` 用于默认通信域上的 AllReduce/Broadcast。
- `HCCL_OP_EXPANSION_MODE` 环境变量在当前 torch_npu 后端中不可靠，测试脚本会通过
  `ProcessGroupHCCL.Options().hccl_config` 显式设置。

### 2. 不上板快测

用于本地快速检查 shape inference、converter 注册、ABI 防火墙等不需要多 rank NPU
的内容：

```bash
pytest tests/ -m "not npu and not dist"
```

如果只想检查 ABI 防火墙：

```bash
pytest tests/test_abi_firewall.py -v
```

ABI 测试会检查：

- `libcustom_comm_impl.so` 不包含 `__cxx11` 符号。
- shim 至少导出一个非 C++ mangled 的 `extern "C"` 入口。
- `custom_comm._C.so` 包含 CXX11 ABI 符号，匹配 torch。
- `_C.so` 的 undefined symbol 能和 shim 的 defined symbol 对上。
- `_C.so` 的 `NEEDED` 包含 `libcustom_comm_impl.so`，并带 `$ORIGIN` RUNPATH。

### 3. Eager 功能测试

默认 decomposed 路径：

```bash
torchrun --nproc_per_node=8 -m pytest tests/allgather_batch/test_eager.py -m npu -k TestNpuFunctional
```

覆盖点：

- 单 desc，dtype 覆盖 `int8`、`float16`、`float32`、`bfloat16`。
- 异构 dtype 多 desc，例如 INT8 + FP32。
- OPT-AG 类三 tensor 场景，例如 INT8 data + FP32 scale + INT32 ids。
- 重复调用稳定性。

CCU 路径：

```bash
CUSTOM_COMM_USE_CCU=1 \
torchrun --nproc_per_node=8 -m pytest tests/allgather_batch/test_eager.py -m npu -k TestCcuPath
```

覆盖点：

- CCU SCHED 正确性。
- CCU MS 与 SCHED 结果对比。
- `1024/2048/4096/4097/8192/65536` bytes boundary。

### 4. Graph 与 aclGraph 测试

```bash
torchrun --nproc_per_node=8 -m pytest tests/allgather_batch/test_graph.py -m npu
```

覆盖点：

- Meta shape inference。
- torchair converter 是否注册成功。
- `torch.compile(backend="torchair")` 端到端执行。
- eager 输出与 graph 输出一致。
- `torch.npu.NPUGraph` capture/replay。
- 多次 replay 后 shape 和执行稳定性。

### 5. Benchmark

默认路径 benchmark：

```bash
torchrun --nproc_per_node=8 tests/allgather_batch/bench.py
```

CCU 路径 benchmark：

```bash
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 tests/allgather_batch/bench.py
```

CCU MS/HCCL expansion mode 专项 benchmark：

```bash
torchrun --nproc_per_node=8 tests/allgather_batch/ccu_ms_bench.py --expansion-mode CCU_MS
```

benchmark 会对比：

- 多次 `dist.all_gather(list)`。
- 多次 `dist.all_gather_into_tensor`。
- Python 侧手动 pack 后一次 AllGather。
- `torch.ops.custom_comm.allgather_batch`。
- pybind11 eager 入口。
- in-place 输出入口。

### 推荐执行顺序

```bash
source ~/Ascend/set_env.sh
pip install -r requirements.txt
pip install -e .

pytest tests/test_abi_firewall.py -v
pytest tests/ -m "not npu and not dist"

torchrun --nproc_per_node=8 tests/smoke_test.py
torchrun --nproc_per_node=8 -m pytest tests/allgather_batch/test_eager.py -m npu -k TestNpuFunctional
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 -m pytest tests/allgather_batch/test_eager.py -m npu -k TestCcuPath
torchrun --nproc_per_node=8 -m pytest tests/allgather_batch/test_graph.py -m npu

torchrun --nproc_per_node=8 tests/allgather_batch/bench.py
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 tests/allgather_batch/bench.py
```

## 常见问题

### 1. 为什么 meta 测试里可能看到 8 desc，而 runtime 上限是 6？

当前 runtime 真实上限来自 `ops/allgather_batch/inc/common.h` 中的 `MAX_DESC_COUNT=6`。
如果 meta/文档/历史测试仍覆盖 8 desc，这通常表示接口设计目标和当前 CCU/MS runtime
实现上限还没有完全对齐。上板 runtime 测试应以 `MAX_DESC_COUNT=6` 为准。

### 2. 为什么推荐 hcom 使用 raw HcclComm handle 字符串？

部分 torch_npu 版本通过 group name 调 `HcomGetCommHandleByGroup` 可能拿到内部
`collComm` 为空的 comm。当前 binding 会优先解析 `get_hccl_comm(rank)` 返回的整数
handle 字符串，解析失败时再回退到 legacy group name。

### 3. CCU_MS 测试为什么要单独关注 4096/4097？

CCU MS 路径存在单 slot 大小边界。`4096` 是典型边界点，`4097` 是刚超过单 slot
的第一类场景，容易暴露多 slot 切分、go size、loop iter 和结果拼接问题。

## License

Apache-2.0
