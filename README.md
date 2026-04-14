# custom_comm

面向昇腾 NPU 的高性能自定义通信算子库。在 HCCL 之上提供融合通信原语，
通过 `torch_npu` 注册为 PyTorch 自定义算子，支持 eager mode 和 graph mode (torchair)。

## 前置依赖

- CANN 9.0+ (Atlas A5 / Ascend 910_95)
- PyTorch 2.6+，torch_npu 2.6+
- Python 3.10+

## 安装

custom_comm 可以从源码安装。安装前需要确保 CANN toolkit 已就绪。

```bash
source ~/Ascend/set_env.sh
pip install -e .
```

仅编译 C++ 库（不含 Python 绑定）：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 快速开始

下面的例子展示了如何在多卡环境下使用 `allgather_batch` 将不同 dtype 的 tensor
一次性聚合：

```python
import torch
import torch_npu
import custom_comm

torch.distributed.init_process_group(backend="hccl")
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.npu.set_device(rank)

# 获取 HCCL 通信组名
pg = torch.distributed.group.WORLD
hcom = pg._get_backend(torch.device(f"npu:{rank}")).get_hccl_comm_name(rank)

# INT8 数据 + FP32 scale, 一次调用完成聚合
data = torch.randn(2048, device=f"npu:{rank}").to(torch.int8)
scale = torch.randn(4, device=f"npu:{rank}", dtype=torch.float32)

gathered = custom_comm.allgather_batch([data, scale], hcom, world_size)
# gathered[0].shape == (2048 * world_size,), dtype=int8
# gathered[1].shape == (4 * world_size,),    dtype=float32
```

### 运行

```bash
torchrun --nproc_per_node=8 example.py
```

## 执行模式

`allgather_batch` 支持两种执行路径，通过环境变量切换：

| 环境变量 `CUSTOM_COMM_USE_CCU` | 策略 | 说明 |
|---|---|---|
| 未设置（默认） | Phase 1: Decomposed | pack 到连续 buffer → 单次 HcclAllGather → unpack |
| `1` | Phase 2: CCU Kernel | 单次 CCU launch，zero-copy RDMA，无 pack/unpack 开销 |

## Graph Mode

`custom_comm` 注册了 torchair GE converter，可以在 `torch.compile` 图模式下使用：

```python
@torch.compile(backend="npu")
def fused_allgather(acts, scales, hcom, ws):
    return custom_comm.allgather_batch([acts, scales], hcom, ws)
```

当前 converter 将 `allgather_batch` 分解为多个独立的 `HcomAllGather` 算子。
如果未来 GE IR 支持原生 batched AllGather，可以进一步优化。

## 测试

```bash
# Meta-device 测试（无需 NPU）
pytest tests/ -k "meta or Meta"

# NPU 功能测试
torchrun --nproc_per_node=8 -m pytest tests/test_allgather_batch.py -v

# 性能基准
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
```

## License

Apache-2.0
