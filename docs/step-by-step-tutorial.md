# custom_comm 保姆级编译与测试教程

本教程面向第一次接触本项目的开发者，每一步都附带验证命令和预期输出，确保你不会走偏。

---

## 目录

- [第一部分: 环境检查](#第一部分-环境检查)
- [第二部分: 安装 Python 依赖](#第二部分-安装-python-依赖)
- [第三部分: 编译 custom_comm](#第三部分-编译-custom_comm)
- [第四部分: 无 NPU 测试（Meta 设备）](#第四部分-无-npu-测试meta-设备)
- [第五部分: A3 上测试 Phase 1（Decomposed）](#第五部分-a3-上测试-phase-1decomposed)
- [第六部分: A5 上测试 Phase 2（CCU 单算子）](#第六部分-a5-上测试-phase-2ccu-单算子)
- [第七部分: 性能基准测试](#第七部分-性能基准测试)
- [第八部分: 图模式测试](#第八部分-图模式测试)
- [附录: 排错指南](#附录-排错指南)

---

## 第一部分: 环境检查

在做任何事情之前，先确认你的环境满足基本条件。

### Step 1.1: 确认操作系统和架构

```bash
uname -m && cat /etc/os-release | head -3
```

**预期输出**（示例）:
```
x86_64
NAME="Ubuntu"
VERSION_ID="22.04"
...
```

或者 `aarch64`（ARM 服务器）。两种架构都支持。

### Step 1.2: 确认 Python 版本

```bash
python3 --version
```

**预期输出**: `Python 3.10.x` 或更高。

**如果版本过低**: 需要安装 Python 3.10+。

### Step 1.3: 确认 CANN SDK 是否已安装

```bash
# 检查 set_env.sh 是否存在
ls ~/Ascend/set_env.sh 2>/dev/null || ls /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null
```

**预期输出**: 显示文件路径，说明 CANN SDK 已安装。

**如果找不到**: 需要先安装 CANN 9.0 Toolkit。安装方式:
```bash
# 完整安装（包含运行时库，推荐）
bash inspkg.sh -V 9.0.0 -s 950

# 注意: 不要用 -c 参数，-c 是 COMPILE_ONLY 模式，不含 libhccl.so 等运行时库
```

### Step 1.4: 加载 CANN 环境并验证

```bash
# 加载环境变量（路径根据你的实际安装位置调整）
source ~/Ascend/set_env.sh
```

**验证**: 检查 ASCEND_HOME_PATH 是否已设置
```bash
echo $ASCEND_HOME_PATH
```

**预期输出**: 类似 `/home/xxx/Ascend/ascend-toolkit/latest` 或 `~/Ascend/cann-9.0.0`。

### Step 1.5: 补充 LD_LIBRARY_PATH（关键！）

`set_env.sh` 默认只设置了 `lib64/`，但 CANN 9.0 的 `libhccl.so` 在 `x86_64-linux/lib64/` 下。不补这一步后面会报 `libhccl.so not found`。

```bash
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/lib64:$LD_LIBRARY_PATH
```

**验证**: 确认 libhccl.so 可找到
```bash
ls $ASCEND_HOME_PATH/x86_64-linux/lib64/libhccl.so 2>/dev/null && echo "OK: libhccl.so found" || echo "WARN: libhccl.so not found in x86_64-linux/lib64"
```

**预期输出**: `OK: libhccl.so found`

如果找不到，也检查一下标准路径:
```bash
find $ASCEND_HOME_PATH -name "libhccl.so" 2>/dev/null
```

根据实际路径调整 `LD_LIBRARY_PATH`。

### Step 1.6: 确认 SDK 头文件存在

```bash
# Layout A（安装版）
ls $ASCEND_HOME_PATH/include/hccl/hccl_types.h 2>/dev/null && echo "Layout A: installed toolkit"

# Layout B（开发版）
ls $ASCEND_HOME_PATH/hcomm/hcomm/include/hccl/hccl_types.h 2>/dev/null && echo "Layout B: dev-sdk"
```

**预期输出**: 至少显示其中一个 Layout。编译系统会自动适配。

### Step 1.7: 确认 NPU 设备（仅在有 NPU 的机器上）

```bash
npu-smi info
```

**预期输出**: 显示 NPU 设备信息表格（芯片型号、温度、功耗等）。

**没有 NPU 也没关系**: 可以跳过 NPU 相关测试，仍然可以编译和运行 Meta 测试。

---

## 第二部分: 安装 Python 依赖

### Step 2.1: 创建虚拟环境（推荐但可选）

```bash
python3 -m venv ~/.venvs/custom_comm
source ~/.venvs/custom_comm/bin/activate
```

**验证**:
```bash
which python && which pip
```

**预期输出**: 路径包含 `.venvs/custom_comm`。

### Step 2.2: 安装 PyTorch

```bash
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
```

**验证**:
```bash
python -c "import torch; print(f'torch {torch.__version__}')"
```

**预期输出**: `torch 2.9.0` 或 `torch 2.9.0+cpu`

### Step 2.3: 安装 torch_npu

```bash
pip install torch_npu==2.9.0
```

**如果报版本冲突**（常见，因为 torch 是 cpu 版本）:
```bash
pip install torch_npu==2.9.0 --no-deps
pip install pyyaml numpy   # 手动补装 torch_npu 的运行时依赖
```

**验证**: 确认 import 不报错
```bash
python -c "import torch_npu; print(f'torch_npu {torch_npu.__version__}')"
```

**预期输出**: `torch_npu 2.9.0`

**如果报 `libhccl.so not found`**: 回到 Step 1.5 补充 `LD_LIBRARY_PATH`。

### Step 2.4: 安装 pytest

```bash
pip install pytest
```

**验证**:
```bash
pytest --version
```

**预期输出**: `pytest 8.x.x` 或类似版本号。

### Step 2.5: (可选) 安装 torchair（图模式需要）

```bash
pip install torchair
```

不安装也不影响 eager 模式测试，图模式测试会自动 skip。

---

## 第三部分: 编译 custom_comm

### Step 3.1: 进入项目目录

```bash
cd /home/jiangshui/ag-scale-improved/custom_comm
```

**验证**: 确认目录结构正确
```bash
ls setup.py CMakeLists.txt ops/ torch_ext/ python/ tests/
```

**预期输出**: 列出这些文件和目录，无报错。

### Step 3.2: 确认环境变量已加载

编译前必须确保环境变量已设置。如果你刚打开新终端，需要重新执行:

```bash
source ~/Ascend/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/lib64:$LD_LIBRARY_PATH
```

**快速检查**:
```bash
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
python -c "import torch; import torch_npu; print('torch + torch_npu OK')"
```

**预期输出**:
```
ASCEND_HOME_PATH=/home/xxx/Ascend/...
torch + torch_npu OK
```

### Step 3.3: pip install 编译安装

```bash
pip install -e .
```

这个命令做了以下事情:
1. 调用 `setup.py`，发现 CANN SDK 路径
2. 使用 `NpuExtension` 编译 C++ 源文件（6 个文件）
3. 链接 `libhcomm.so` 和 `libascendcl.so`
4. 生成 `custom_comm/_C.*.so`
5. 以 editable 模式安装 Python 包

**编译大约需要 1-3 分钟**，期间会输出大量编译日志。

**预期输出**（最后几行）:
```
...
running build_ext
...
Successfully installed custom_comm-0.1.0
```

### Step 3.4: 验证安装成功

```bash
python -c "import custom_comm; print('custom_comm imported OK')"
```

**预期输出**: `custom_comm imported OK`

### Step 3.5: 验证 C 扩展加载成功

```bash
python -c "
import custom_comm._C
import torch
# 检查算子是否注册
op = torch.ops.custom_comm.allgather_batch
print(f'Op registered: {op}')
print('C extension loaded OK')
"
```

**预期输出**:
```
Op registered: <OpOverloadPacket(allgather_batch)>
C extension loaded OK
```

### Step 3.6: 验证 Meta 实现可工作

这一步不需要 NPU，在任何机器上都能运行:

```bash
python -c "
import torch
import custom_comm

# 创建 meta 设备上的 tensor（不占实际内存）
x = torch.empty(128, 64, dtype=torch.float32, device='meta')
out = torch.ops.custom_comm.allgather_batch([x], 'dummy', 4)
print(f'Input:  {x.shape}')
print(f'Output: {out[0].shape}')
assert out[0].shape == (512, 64), 'Shape mismatch!'
print('Meta kernel OK: shape inference correct')
"
```

**预期输出**:
```
Input:  torch.Size([128, 64])
Output: torch.Size([512, 64])
Meta kernel OK: shape inference correct
```

**到这里，编译就完成了。** 下面进入测试环节。

---

## 第四部分: 无 NPU 测试（Meta 设备）

即使没有 NPU 硬件，也能验证 shape 推导逻辑。这部分可以在任何机器（macOS、CPU 服务器）上运行。

### Step 4.1: 运行全部 Meta 测试

```bash
cd /home/jiangshui/ag-scale-improved/custom_comm
pytest tests/test_allgather_batch.py -k "Meta" -v
```

**预期输出**:
```
tests/test_allgather_batch.py::TestMetaKernel::test_single_desc[int8-1] PASSED
tests/test_allgather_batch.py::TestMetaKernel::test_single_desc[int8-2] PASSED
tests/test_allgather_batch.py::TestMetaKernel::test_single_desc[int8-4] PASSED
tests/test_allgather_batch.py::TestMetaKernel::test_single_desc[int8-8] PASSED
tests/test_allgather_batch.py::TestMetaKernel::test_single_desc[float16-...] PASSED
...
tests/test_allgather_batch.py::TestMetaKernel::test_heterogeneous_dtypes[2] PASSED
tests/test_allgather_batch.py::TestMetaKernel::test_heterogeneous_dtypes[4] PASSED
tests/test_allgather_batch.py::TestMetaKernel::test_heterogeneous_dtypes[8] PASSED
tests/test_allgather_batch.py::TestMetaKernel::test_ag09_meta PASSED
tests/test_allgather_batch.py::TestMetaKernel::test_max_desc_count PASSED
tests/test_allgather_batch.py::TestMetaKernel::test_empty_dim0 PASSED
tests/test_allgather_batch.py::TestMetaKernel::test_preserves_dtype PASSED
...
```

所有标记 `PASSED` 即通过。

### Step 4.2: 运行图模式 shape 推导测试

```bash
pytest tests/test_graph_mode.py -k "TestMetaShapeInference" -v
```

**预期输出**:
```
tests/test_graph_mode.py::TestMetaShapeInference::test_single_input PASSED
tests/test_graph_mode.py::TestMetaShapeInference::test_heterogeneous_dtypes[1] PASSED
tests/test_graph_mode.py::TestMetaShapeInference::test_heterogeneous_dtypes[2] PASSED
tests/test_graph_mode.py::TestMetaShapeInference::test_heterogeneous_dtypes[8] PASSED
tests/test_graph_mode.py::TestMetaShapeInference::test_max_descs PASSED
tests/test_graph_mode.py::TestMetaShapeInference::test_empty_dim0 PASSED
tests/test_graph_mode.py::TestMetaShapeInference::test_multidim PASSED
```

### Step 4.3: （可选）GE Converter 注册验证

需要 torchair 已安装:

```bash
pytest tests/test_graph_mode.py -k "TestConverterRegistration" -v
```

**预期输出**: `PASSED`（如果 torchair 未安装，会显示 `SKIPPED`，这是正常的）。

### 小结

全部 Meta 测试通过，说明:
- C++ 扩展编译正确
- 算子 schema 注册正确
- shape 推导逻辑正确

---

## 第五部分: A3 上测试 Phase 1（Decomposed）

> A3 (Ascend 910B) 仅支持 Phase 1 Decomposed 路径，**不支持 CCU**。

以下步骤需要在**有 NPU 设备的 A3 机器**上执行。

### Step 5.1: 确认 NPU 可用

```bash
python -c "
import torch
import torch_npu
print(f'NPU available: {torch.npu.is_available()}')
print(f'NPU count: {torch.npu.device_count()}')
"
```

**预期输出**:
```
NPU available: True
NPU count: 8      # 或 2, 4 等
```

**如果 NPU available 为 False**: 检查驱动和 CANN 安装。

### Step 5.2: 2 卡功能测试 — 单 dtype

从最简单的开始，用 2 卡测试单个 tensor 的 AllGather:

```bash
torchrun --nproc_per_node=2 -m pytest tests/test_allgather_batch.py \
    -k "TestNpuFunctional and test_single_desc" -v
```

**预期输出**（每个 rank 都会打印，这里只看 rank 0 的汇总）:
```
tests/test_allgather_batch.py::TestNpuFunctional::test_single_desc[int8] PASSED
tests/test_allgather_batch.py::TestNpuFunctional::test_single_desc[float16] PASSED
tests/test_allgather_batch.py::TestNpuFunctional::test_single_desc[float32] PASSED
tests/test_allgather_batch.py::TestNpuFunctional::test_single_desc[bfloat16] PASSED
```

**如果出错**: 跳到 [附录: 排错指南](#附录-排错指南)。

### Step 5.3: 2 卡功能测试 — 异构 dtype

验证 INT8 + FP32 混合类型 AllGather（这是 MoE 量化推理的核心场景）:

```bash
torchrun --nproc_per_node=2 -m pytest tests/test_allgather_batch.py \
    -k "TestNpuFunctional and test_heterogeneous" -v
```

**预期输出**:
```
tests/test_allgather_batch.py::TestNpuFunctional::test_heterogeneous_int8_fp32 PASSED
```

### Step 5.4: 2 卡功能测试 — 三 tensor 打包

验证 OPT-AG-09 场景: INT8 data + FP32 scale + INT32 topk_ids:

```bash
torchrun --nproc_per_node=2 -m pytest tests/test_allgather_batch.py \
    -k "TestNpuFunctional and test_three_tensor" -v
```

**预期输出**:
```
tests/test_allgather_batch.py::TestNpuFunctional::test_three_tensor_pack PASSED
```

### Step 5.5: 8 卡全量功能测试

如果机器有 8 张卡，跑满配测试:

```bash
torchrun --nproc_per_node=8 -m pytest tests/test_allgather_batch.py \
    -k "TestNpuFunctional" -v
```

**预期输出**: 所有用例 `PASSED`。

### Step 5.6: 稳定性测试 — 100 次重复调用

```bash
torchrun --nproc_per_node=2 -m pytest tests/test_allgather_batch.py \
    -k "TestNpuFunctional and test_repeated" -v
```

**预期输出**:
```
tests/test_allgather_batch.py::TestNpuFunctional::test_repeated_calls PASSED
```

这个测试在循环内调用 100 次 `allgather_batch`，验证无崩溃、无内存泄漏。

### Step 5.7: 汇总 — A3 全量测试

一条命令跑完 A3 上所有该跑的测试:

```bash
# A3 全量测试（排除 CCU 路径测试）
torchrun --nproc_per_node=8 -m pytest tests/test_allgather_batch.py \
    -k "TestNpuFunctional" -v
```

**注意: A3 上不要设置 `CUSTOM_COMM_USE_CCU=1`，不要运行 `TestCcuPath` 测试。**

---

## 第六部分: A5 上测试 Phase 2（CCU 单算子）

> A5 (Ascend 910_95) 支持 Phase 1 和 Phase 2（CCU 零拷贝）。
> 以下步骤需要在**有 NPU 设备的 A5 机器**上执行。

### Step 6.1: 确认是 A5 机器

```bash
npu-smi info | head -20
```

检查芯片型号是否为 `910_95`（A5）而非 `910B`（A3）。

### Step 6.2: 确认 Phase 1 先通过

**先跑 Phase 1**，确保基础功能正常。Phase 2 的正确性验证依赖 Phase 1 的输出作为基准。

```bash
torchrun --nproc_per_node=2 -m pytest tests/test_allgather_batch.py \
    -k "TestNpuFunctional" -v
```

**预期输出**: 全部 `PASSED`。如果 Phase 1 都不过，不要继续测 Phase 2。

### Step 6.3: CCU 正确性测试 — Phase 2 vs Phase 1 bit-exact 对比

这是最核心的测试。它会:
1. 先不设 `CUSTOM_COMM_USE_CCU`，用 Phase 1（Decomposed）跑一次，拿到输出
2. 再设 `CUSTOM_COMM_USE_CCU=1`，用 Phase 2（CCU）跑一次，拿到输出
3. 对比两次输出是否 bit-exact 一致

```bash
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=2 -m pytest \
    tests/test_allgather_batch.py -k "TestCcuPath and test_ccu_matches_decomposed" -v
```

**预期输出**:
```
tests/test_allgather_batch.py::TestCcuPath::test_ccu_matches_decomposed PASSED
```

**这一步通过说明 CCU kernel 的通信结果与字节打包方案完全一致。**

### Step 6.4: 8 卡 CCU 正确性测试

```bash
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 -m pytest \
    tests/test_allgather_batch.py -k "TestCcuPath" -v
```

**预期输出**: `PASSED`。

### Step 6.5: CCU 稳定性测试

CCU 路径下 100 次重复调用:

```bash
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 -m pytest \
    tests/test_allgather_batch.py -k "TestNpuFunctional and test_repeated" -v
```

**预期输出**: `PASSED`，无 crash、无 hang。

### Step 6.6: 手动验证 CCU 单算子（交互式）

如果你想更直观地看到 CCU 算子的行为，可以写一个简单的 Python 脚本:

创建 `test_ccu_manual.py`:
```python
import os
import torch
import torch_npu
import custom_comm

# 启用 CCU 路径
os.environ["CUSTOM_COMM_USE_CCU"] = "1"

# 初始化分布式
torch.distributed.init_process_group(backend="hccl")
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.npu.set_device(rank)

# 获取 communicator
pg = torch.distributed.group.WORLD
hcom = pg._get_backend(torch.device(f"npu:{rank}")).get_hccl_comm_name(rank)

# === 测试 1: 单 tensor AllGather ===
data = torch.arange(8, device=f"npu:{rank}", dtype=torch.int8) + rank * 10
result = custom_comm.allgather_batch([data], hcom, world_size)
torch.npu.synchronize()
if rank == 0:
    print(f"[Test 1] Single tensor")
    print(f"  Input:  {data}")
    print(f"  Output: {result[0]}")
    print(f"  Shape:  {data.shape} -> {result[0].shape}")
    print()

# === 测试 2: 异构 dtype (INT8 + FP32) ===
x_int8 = torch.full((4,), rank + 1, device=f"npu:{rank}", dtype=torch.int8)
x_fp32 = torch.full((2,), (rank + 1) * 0.5, device=f"npu:{rank}", dtype=torch.float32)
result = custom_comm.allgather_batch([x_int8, x_fp32], hcom, world_size)
torch.npu.synchronize()
if rank == 0:
    print(f"[Test 2] Heterogeneous dtype")
    print(f"  INT8 input:  {x_int8}  -> output: {result[0]}")
    print(f"  FP32 input:  {x_fp32}  -> output: {result[1]}")
    print()

# === 测试 3: 重复调用 ===
data = torch.ones(64, device=f"npu:{rank}", dtype=torch.float16)
for i in range(100):
    custom_comm.allgather_batch([data], hcom, world_size)
torch.npu.synchronize()
if rank == 0:
    print(f"[Test 3] 100 repeated calls: OK")

torch.distributed.destroy_process_group()
```

运行:
```bash
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=2 test_ccu_manual.py
```

**预期输出**（2 卡示例）:
```
[Test 1] Single tensor
  Input:  tensor([0, 1, 2, 3, 4, 5, 6, 7], device='npu:0', dtype=torch.int8)
  Output: tensor([ 0,  1,  2,  3,  4,  5,  6,  7, 10, 11, 12, 13, 14, 15, 16, 17], device='npu:0', dtype=torch.int8)
  Shape:  torch.Size([8]) -> torch.Size([16])

[Test 2] Heterogeneous dtype
  INT8 input:  tensor([1, 1, 1, 1], device='npu:0', dtype=torch.int8)  -> output: tensor([1, 1, 1, 1, 2, 2, 2, 2], device='npu:0', dtype=torch.int8)
  FP32 input:  tensor([0.5000, 0.5000], device='npu:0')  -> output: tensor([0.5000, 0.5000, 1.0000, 1.0000], device='npu:0')

[Test 3] 100 repeated calls: OK
```

### Step 6.7: 汇总 — A5 全量测试

一条命令跑完 A5 上所有测试（Phase 1 + Phase 2）:

```bash
# Phase 1 测试
torchrun --nproc_per_node=8 -m pytest tests/test_allgather_batch.py \
    -k "TestNpuFunctional" -v

# Phase 2 CCU 测试
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 -m pytest \
    tests/test_allgather_batch.py -k "TestCcuPath" -v
```

---

## 第七部分: 性能基准测试

### Step 7.1: Phase 1 同构 benchmark

```bash
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
```

**预期输出**（rank 0 打印）:
```
AllGatherBatch Benchmark (Phase: Decomposed, W=8)
  descs       size         us
      1        4KB       xx.x
      1      256KB       xx.x
      1     2.5MB       xx.x
      1       10MB      xxx.x
      2        4KB       xx.x
      ...
```

### Step 7.2: Phase 1 OPT-AG-09 场景

```bash
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py --ag09
```

**预期输出**:
```
OPT-AG-09 Benchmark (Phase: Decomposed, W=8)
  config                              us
  ------------------------------  ----------
  2.5MB+scale+ids                     xxx.x
  256KB+scale+ids                      xx.x
  64KB+scale+ids                       xx.x
```

**记下这些数字，等会与 CCU 路径对比。**

### Step 7.3: Phase 2 CCU benchmark（仅 A5）

```bash
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 tests/bench_allgather_batch.py --ag09
```

**预期输出**:
```
OPT-AG-09 Benchmark (Phase: CCU, W=8)
  config                              us
  ------------------------------  ----------
  2.5MB+scale+ids                     xxx.x   ← 应该比 Phase 1 小
  256KB+scale+ids                      xx.x
  64KB+scale+ids                       xx.x
```

### Step 7.4: 性能对比分析

将 Phase 1 和 Phase 2 的 us 数值放在一起对比:

| 场景 | Phase 1 (us) | Phase 2 CCU (us) | 加速比 |
|------|-------------|------------------|--------|
| 2.5MB+scale+ids | (你的数据) | (你的数据) | P1/P2 |
| 256KB+scale+ids | (你的数据) | (你的数据) | P1/P2 |
| 64KB+scale+ids  | (你的数据) | (你的数据) | P1/P2 |

根据设计分析，Phase 2 在 OPT-AG-09 场景下应节省约 41us/call（约 77% 的额外开销）。

---

## 第八部分: 图模式测试

### Step 8.1: GE 图模式编译 + 正确性（需要 torchair + NPU）

```bash
torchrun --nproc_per_node=2 -m pytest tests/test_graph_mode.py \
    -k "TestGraphModeE2E" -v
```

**预期输出**:
```
tests/test_graph_mode.py::TestGraphModeE2E::test_ge_graph_compile PASSED
tests/test_graph_mode.py::TestGraphModeE2E::test_ge_graph_correctness PASSED
```

### Step 8.2: aclGraph Capture + Replay（需要 A5）

```bash
torchrun --nproc_per_node=2 -m pytest tests/test_graph_mode.py \
    -k "TestAclGraphCapture" -v
```

**预期输出**:
```
tests/test_graph_mode.py::TestAclGraphCapture::test_aclgraph_capture_replay PASSED
tests/test_graph_mode.py::TestAclGraphCapture::test_aclgraph_repeated_replay PASSED
```

### Step 8.3: 图模式全量

```bash
torchrun --nproc_per_node=8 -m pytest tests/test_graph_mode.py -v
```

---

## 附录: 排错指南

### 问题 1: `pip install -e .` 编译失败

**症状**: 编译过程报找不到头文件，如 `fatal error: hccl/hccl_types.h: No such file`

**原因**: CANN SDK 路径未正确设置。

**解法**:
```bash
# 检查环境变量
echo $ASCEND_HOME_PATH

# 检查头文件是否存在
find $ASCEND_HOME_PATH -name "hccl_types.h" 2>/dev/null

# 如果环境变量未设置，手动指定
export ASCEND_HOME_PATH=/path/to/your/cann/sdk
pip install -e .
```

### 问题 2: `import torch_npu` 报 `libhccl.so not found`

**解法**:
```bash
source ~/Ascend/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/lib64:$LD_LIBRARY_PATH
```

### 问题 3: `import custom_comm` 成功但 `custom_comm._C` 失败

**症状**: `ModuleNotFoundError: No module named 'custom_comm._C'`

**原因**: C++ 扩展未编译或编译失败（setup.py 中 `try/except ImportError` 会静默跳过）。

**解法**:
```bash
# 检查 .so 文件是否存在
find . -name "_C*.so" 2>/dev/null

# 如果没有，重新编译并查看完整日志
pip install -e . -v 2>&1 | tail -50
```

### 问题 4: `torchrun` 测试 hang 住不动

**可能原因**:
- `nproc_per_node` 超过实际 NPU 数量
- 其他进程占用了 NPU

**解法**:
```bash
# 检查可用 NPU 数量
python -c "import torch, torch_npu; print(torch.npu.device_count())"

# 用实际数量替换
torchrun --nproc_per_node=<实际数量> ...

# 检查是否有残留进程
ps aux | grep torchrun
ps aux | grep python | grep npu
# 如果有，kill 掉再重试
```

### 问题 5: `CUSTOM_COMM_USE_CCU=1` 在 A3 上失败

**这是预期行为。** A3 不支持 CCU 路径。去掉 `CUSTOM_COMM_USE_CCU=1`，使用 Phase 1。

### 问题 6: rsync 同步后 `import custom_comm` 报错

**原因**: rsync 覆盖了 `*.egg-info`，editable install 的符号链接失效。

**解法**: 重新编译
```bash
pip install -e .
```

### 问题 7: Docker 中 build 目录无法删除

**解法**:
```bash
# 在 Docker 容器内执行
chown -R $(id -u):$(id -g) /workspace/custom_comm
```

### 问题 8: 测试显示 SKIPPED 而非 PASSED

**常见原因和处理**:

| Skip 原因 | 说明 | 处理 |
|-----------|------|------|
| `NPU device not available` | 当前机器无 NPU | 正常，换有 NPU 的机器 |
| `custom_comm extension not built` | C 扩展未编译 | 重新 `pip install -e .` |
| `torchair not installed` | 未安装 torchair | `pip install torchair` |
| `dist not initialized` | 未用 `torchrun` 启动 | 用 `torchrun --nproc_per_node=N` |

---

## 测试通过检查清单

完成教程后，对照此清单确认:

### 编译
- [ ] `pip install -e .` 成功
- [ ] `import custom_comm` 无报错
- [ ] `custom_comm._C` 加载成功
- [ ] Meta kernel shape 推导正确

### A3 测试（Phase 1 only）
- [ ] 2 卡单 dtype 通过
- [ ] 2 卡异构 dtype 通过
- [ ] 2 卡三 tensor 打包通过
- [ ] 8 卡全量通过
- [ ] 100 次重复调用无崩溃

### A5 测试（Phase 1 + Phase 2）
- [ ] Phase 1 全部通过（同 A3）
- [ ] Phase 2 CCU vs Phase 1 bit-exact 对比通过
- [ ] 8 卡 CCU 正确性通过
- [ ] CCU 100 次重复调用无崩溃
- [ ] Phase 1 性能基准数据已记录
- [ ] Phase 2 CCU 性能基准数据已记录
- [ ] Phase 2 性能优于 Phase 1

### 图模式（可选）
- [ ] GE 图模式编译通过
- [ ] GE 图模式 vs eager 正确性对比通过
- [ ] aclGraph capture + replay 通过
