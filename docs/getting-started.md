# custom_comm 新手入门指南

## 1. 项目简介

### 这个项目是什么？

custom_comm 是一个面向华为昇腾 NPU 的**自定义高性能通信算子库**。它解决的核心问题是：

> 在 MoE 量化推理场景中，INT8 数据和 FP32 scale 需要分别执行独立的 AllGather 操作。每次 AllGather 都会触发完整的 kernel launch 调度（约 8-10us），两次独立调用产生约 16-20us 的冗余开销。

custom_comm 提供的 `allgather_batch` 算子可以在**单次调用中同时 gather 多个不同数据类型的 tensor**，消除冗余 launch 开销。

### 一句话理解

把原来的"发 N 个快递分 N 次叫快递员"变成"一次叫快递员发 N 个快递"。

### 支持的功能

| 功能 | 说明 |
|------|------|
| 异构 dtype 批量 AllGather | 一次调用 gather 最多 8 个不同类型的 tensor |
| Eager 模式 | 直接调用，支持 Phase 1（字节打包）和 Phase 2（CCU 零拷贝） |
| Graph 模式 | 支持 `torch.compile(backend="torchair")` |
| aclGraph Capture | 支持图捕获与回放 |

---

## 2. 项目架构

### 分层结构

```
用户代码 (Python)
    │
    ▼
Python 层: python/custom_comm/ops.py        ← 类型提示 wrapper
    │
    ▼
Torch 算子: torch_ext/csrc/                 ← TORCH_LIBRARY 注册 + NPU/Meta 实现
    │
    ▼
C API 层: ops/allgather_batch/              ← 核心通信逻辑 + CCU kernel
    │
    ▼
CANN SDK: libhcomm.so, libascendcl.so       ← 华为提供的底层库
```

### 目录结构

```
custom_comm/
├── CMakeLists.txt                 # 纯 C++ 构建（语法检查用）
├── setup.py                       # Python 包构建（pip install 用）
├── cmake/FindCANN.cmake           # SDK 自动发现脚本
│
├── ops/allgather_batch/           # C/C++ 核心实现
│   ├── inc/                       # 头文件
│   │   ├── hccl_custom_allgather_batch.h  # C API 声明
│   │   ├── common.h               # 共享类型、错误宏
│   │   ├── engine_ctx.h           # CCU 引擎上下文管理
│   │   └── ccu_kernel_ag_batch_mesh1d.h   # CCU kernel 类
│   └── src/                       # 实现
│       ├── all_gather_batch.cc    # 入口 + Phase1/2 策略分发
│       ├── decomposed_strategy.cc # Phase 1: 字节打包方案
│       ├── engine_ctx.cc          # CCU 引擎上下文
│       └── ccu_kernel_ag_batch_mesh1d.cc  # Phase 2: CCU kernel
│




├── torch_ext/csrc/                # PyTorch C++ 扩展
│   ├── ops_registration.cpp       # TORCH_LIBRARY schema 定义
│   └── allgather_batch.cpp        # NPU 设备实现 + Meta shape 推导
│
├── python/custom_comm/            # Python 包
│   ├── __init__.py                # 加载 .so + 导出接口
│   ├── ops.py                     # allgather_batch 类型提示 wrapper
│   └── converters/                # torch.compile 图模式支持
│       ├── __init__.py
│       └── allgather_batch_converter.py   # GE converter
│
├── tests/                         # 测试
│   ├── test_allgather_batch.py    # Meta + NPU 功能测试
│   ├── test_graph_mode.py         # 图模式测试
│   └── bench_allgather_batch.py   # 性能基准测试
│
├── docs/                          # 文档和设计资料
├── sync.sh                        # rsync 同步代码到蓝区
├── requirements.txt               # Python 依赖
└── version.txt                    # 版本号
```

### 两种执行路径

| | Phase 1: Decomposed（默认） | Phase 2: CCU Batched |
|---|---|---|
| 原理 | 把多个 tensor 打包成字节流，执行一次 AllGather，再拆包 | 注册一个 CCU 硬件微码 kernel，零拷贝直接 RDMA gather |
| kernel launch 次数 | 3 次（pack + AG + unpack） | 1 次 |
| 额外内存流量 | ~45MB (W=8 场景) | ~5MB（仅 self-copy） |
| 启用方式 | 默认 | `CUSTOM_COMM_USE_CCU=1` |
| 平台要求 | 所有 CANN 版本 | CANN 9.0+, A5 (Ascend 910_95) |
| 适用场景 | 验证正确性、不支持 CCU 的平台 | 生产环境性能优化 |

---

## 3. 环境准备

### 硬件要求

| 平台 | 芯片 | 说明 |
|------|------|------|
| Atlas A5 | Ascend 910_95 | 完整支持（Phase 1 + Phase 2 CCU） |
| Atlas A3 | Ascend 910B | 仅 Phase 1（Decomposed），不支持 CCU 路径 |

### 软件依赖

| 软件 | 版本要求 | 用途 |
|------|---------|------|
| CANN Toolkit | 9.0+ | SDK 头文件和运行时库 |
| Python | 3.10+ | 运行环境 |
| PyTorch | 2.6+ | 深度学习框架 |
| torch_npu | 2.6+ | PyTorch 昇腾适配 |
| torchair | 可选 | 图模式 `torch.compile` 支持 |

### 安装 PyTorch + torch_npu

```bash
# 安装 PyTorch
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu

# 安装 torch_npu（如果版本不匹配用 --no-deps）
pip install torch_npu==2.9.0
# 如果报版本冲突:
pip install torch_npu==2.9.0 --no-deps
pip install pyyaml numpy   # 手动补装 torch_npu 的运行时依赖
```

---

## 4. 编译安装

### 方式一: Python 包安装（推荐，需要 CANN SDK + torch_npu）

这是日常开发和测试时使用的方式，编译后可直接 `import custom_comm` 使用。

```bash
# 1. 配置 CANN 环境变量
source ~/Ascend/set_env.sh

# 2. 重要！补充 LD_LIBRARY_PATH（set_env.sh 默认漏了 x86_64-linux/lib64）
export LD_LIBRARY_PATH=~/Ascend/cann-9.0.0/x86_64-linux/lib64:$LD_LIBRARY_PATH

# 3. 进入项目目录
cd custom_comm

# 4. 开发模式安装（-e 表示 editable，修改代码后无需重装 Python 部分，
#    但修改 C++ 代码后需要重新执行此命令）
pip install -e .
```

**常见问题**:

- **`libhccl.so not found`**: 忘了执行第 2 步，补上 `x86_64-linux/lib64` 路径
- **rsync 同步后扩展丢失**: rsync 会覆盖 `*.egg-info`，需要重新 `pip install -e .`
- **Docker 中 build 目录权限问题**: `chown -R $(id -u):$(id -g) /workspace/custom_comm`

### 方式二: 纯 C++ 编译（语法检查、CI 用）

不需要 torch_npu，仅编译 C++ 共享库 `libcustom_comm_ops.so`。

```bash
# 配置环境变量
source ~/Ascend/set_env.sh

# 编译
cmake -B build && cmake --build build
```

CMake 会自动通过 `FindCANN.cmake` 查找 SDK，搜索顺序:
1. `-DASCEND_CANN_PACKAGE_PATH=...` 手动指定
2. `$ASCEND_CANN_PACKAGE_PATH` 环境变量
3. `$ASCEND_HOME_PATH` 环境变量
4. `/usr/local/Ascend/ascend-toolkit/latest` 默认路径

---

## 5. 在 A5 上测试 CCU 单算子（Phase 2）

A5 (Ascend 910_95) 支持 CCU 硬件微码调度，是 Phase 2 零拷贝路径的唯一支持平台。

### 5.1 环境准备

```bash
# 配置 CANN 环境
source ~/Ascend/set_env.sh
export LD_LIBRARY_PATH=~/Ascend/cann-9.0.0/x86_64-linux/lib64:$LD_LIBRARY_PATH

# 编译安装 custom_comm
cd custom_comm
pip install -e .

# 验证安装
python -c "import custom_comm; print('custom_comm loaded OK')"
```

### 5.2 运行 CCU 功能测试

CCU 路径通过环境变量 `CUSTOM_COMM_USE_CCU=1` 启用。

```bash
# CCU 正确性测试: 验证 Phase 2 输出与 Phase 1 bit-exact 一致
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 \
    -m pytest tests/test_allgather_batch.py -k "TestCcuPath" -v

# 也可以用 2 卡 / 4 卡测试
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=2 \
    -m pytest tests/test_allgather_batch.py -k "TestCcuPath" -v
```

### 5.3 CCU 性能基准

```bash
# 同构 INT8 benchmark（不同 desc 数量和消息大小）
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 \
    tests/bench_allgather_batch.py

# OPT-AG-09 场景: INT8 数据 + FP32 scale + INT32 topk_ids
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 \
    tests/bench_allgather_batch.py --ag09
```

### 5.4 对比 Phase 1 和 Phase 2 性能

```bash
# Phase 1（Decomposed，默认）
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py --ag09

# Phase 2（CCU）
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 tests/bench_allgather_batch.py --ag09

# 对比两次输出的 us 列即可看到性能差异
```

### 5.5 稳定性验证

```bash
# 100 次重复调用测试（内置在 test_repeated_calls 中）
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 \
    -m pytest tests/test_allgather_batch.py -k "repeated" -v
```

### 5.6 图模式测试（aclGraph Capture）

```bash
# GE 图模式编译 + 正确性
torchrun --nproc_per_node=8 \
    -m pytest tests/test_graph_mode.py -k "TestGraphModeE2E" -v

# aclGraph capture + replay
torchrun --nproc_per_node=8 \
    -m pytest tests/test_graph_mode.py -k "TestAclGraphCapture" -v
```

---

## 6. 在 A3 上测试（Phase 1 only）

A3 (Ascend 910B) **不支持 CCU 路径**，只能运行 Phase 1 Decomposed 策略。

### 6.1 环境准备

与 A5 相同的安装步骤:

```bash
source ~/Ascend/set_env.sh
export LD_LIBRARY_PATH=~/Ascend/cann-9.0.0/x86_64-linux/lib64:$LD_LIBRARY_PATH
cd custom_comm
pip install -e .
```

### 6.2 功能测试

```bash
# Phase 1 功能正确性测试（不设 CUSTOM_COMM_USE_CCU）
torchrun --nproc_per_node=8 \
    -m pytest tests/test_allgather_batch.py -k "TestNpuFunctional" -v

# 单 dtype 测试
torchrun --nproc_per_node=8 \
    -m pytest tests/test_allgather_batch.py -k "test_single_desc" -v

# 异构 dtype 测试（INT8 + FP32）
torchrun --nproc_per_node=8 \
    -m pytest tests/test_allgather_batch.py -k "test_heterogeneous" -v

# 三 tensor 打包测试（INT8 + FP32 + INT32）
torchrun --nproc_per_node=8 \
    -m pytest tests/test_allgather_batch.py -k "test_three_tensor" -v

# 重复调用稳定性测试
torchrun --nproc_per_node=8 \
    -m pytest tests/test_allgather_batch.py -k "test_repeated" -v
```

### 6.3 性能基准

```bash
# 同构 benchmark
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py

# OPT-AG-09 场景
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py --ag09
```

### 6.4 A3 上的限制

| 项目 | 说明 |
|------|------|
| Phase 2 (CCU) | **不可用**，设置 `CUSTOM_COMM_USE_CCU=1` 会失败 |
| TestCcuPath 测试 | 跳过，不要在 A3 上运行 |
| 图模式 | GE converter 可用（分解为 N 个独立 HcomAllGather） |
| aclGraph Capture | 依赖 CCU，在 A3 上**不可用** |

---

## 7. 无 NPU 环境测试（macOS / CPU 开发机）

即使没有 NPU 硬件，也可以做部分验证工作:

### Meta 设备 shape 推导测试

```bash
# 不需要 NPU，只需要 pip install -e . 成功
pytest tests/test_allgather_batch.py -k "meta or Meta" -v
pytest tests/test_graph_mode.py -k "TestMetaShapeInference" -v
```

### C++ 语法检查

```bash
cmake -B build && cmake --build build
```

### 图模式 converter 注册测试

```bash
# 需要 torchair 已安装
pytest tests/test_graph_mode.py -k "TestConverterRegistration" -v
```

---

## 8. 代码同步（开发机 → 蓝区服务器）

项目提供 `sync.sh` 脚本，通过 rsync 快速同步代码:

```bash
# 在开发机（macOS）上执行
bash sync.sh
```

**注意**: rsync 同步后会覆盖 `*.egg-info`，需要在蓝区重新执行 `pip install -e .`。

如果蓝区下载 pip 包较慢，可通过 SSH 反向隧道代理加速:

```bash
# 蓝区使用代理
export http_proxy=http://127.0.0.1:10077
export https_proxy=http://127.0.0.1:10077
```

---

## 9. 快速使用示例

```python
import torch
import torch_npu
import custom_comm

# 初始化分布式环境
torch.distributed.init_process_group(backend="hccl")
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.npu.set_device(rank)

# 获取 HCCL communicator 名称
pg = torch.distributed.group.WORLD
hcom = pg._get_backend(torch.device(f"npu:{rank}")).get_hccl_comm_name(rank)

# 创建不同类型的输入 tensor
data = torch.randn(2048, 4096, device=f"npu:{rank}").to(torch.int8)
scale = torch.randn(2048, device=f"npu:{rank}", dtype=torch.float32)
ids = torch.randint(0, 8, (2048,), device=f"npu:{rank}", dtype=torch.int32)

# 一次调用 gather 3 个不同类型的 tensor
results = custom_comm.allgather_batch([data, scale, ids], hcom, world_size)

# results[0]: shape (2048*W, 4096), dtype int8
# results[1]: shape (2048*W,),      dtype float32
# results[2]: shape (2048*W,),      dtype int32
```

启动脚本:

```bash
# Phase 1（默认）
torchrun --nproc_per_node=8 your_script.py

# Phase 2（CCU，仅 A5）
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 your_script.py
```

---

## 10. 常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| `libhccl.so not found` | `LD_LIBRARY_PATH` 缺少 `x86_64-linux/lib64` | `export LD_LIBRARY_PATH=~/Ascend/cann-9.0.0/x86_64-linux/lib64:$LD_LIBRARY_PATH` |
| `torch_npu` 安装版本不匹配 | PyPI torch_npu 要求精确 torch 版本 | `pip install torch_npu==2.9.0 --no-deps` |
| rsync 后 `import custom_comm` 失败 | editable install 符号链接被覆盖 | 重新 `pip install -e .` |
| Docker 中 build 目录无法删除 | root 权限文件 | `chown -R $(id -u):$(id -g) /workspace/custom_comm` |
| A3 上 `CUSTOM_COMM_USE_CCU=1` 失败 | A3 不支持 CCU 路径 | 去掉此环境变量，使用 Phase 1 |
| `HcclThreadResGetInfo` 符号找不到 | CANN 9.0 mirror 版本缺少此 API | 从 hcomm 源码编译安装（需 Docker 环境） |
| ACL header 冲突/重定义 | setup.py include 路径与 torch_npu 内置头文件冲突 | 只添加 `include/hccl/`, `pkg_inc/` 路径，不添加 `include/` 根目录 |

---

## 11. 测试矩阵总结

| 测试类型 | 命令 | A5 | A3 | 无 NPU |
|---------|------|----|----|--------|
| Meta shape 推导 | `pytest tests/ -k "meta"` | OK | OK | OK |
| Phase 1 功能 | `torchrun -m pytest tests/ -k "TestNpuFunctional"` | OK | OK | - |
| Phase 2 CCU | `CUSTOM_COMM_USE_CCU=1 torchrun -m pytest tests/ -k "TestCcuPath"` | OK | - | - |
| GE 图模式 | `torchrun -m pytest tests/test_graph_mode.py -k "E2E"` | OK | OK | - |
| aclGraph Capture | `torchrun -m pytest tests/test_graph_mode.py -k "AclGraph"` | OK | - | - |
| Converter 注册 | `pytest tests/test_graph_mode.py -k "Registration"` | OK | OK | OK* |
| 性能基准 | `torchrun tests/bench_allgather_batch.py` | OK | OK | - |
| CCU 性能基准 | `CUSTOM_COMM_USE_CCU=1 torchrun tests/bench_allgather_batch.py` | OK | - | - |

*需要 torchair 已安装
