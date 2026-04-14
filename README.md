# custom_comm

面向昇腾 NPU 的高性能自定义通信算子库，基于 CANN HCCL/HComm 构建，
以 PyTorch custom op 形式提供，支持 eager mode 和 torchair graph mode。

## 算子

- `allgather_batch` — 单次调用完成多个异构 dtype tensor 的 AllGather，
  避免多次独立调用的启动开销。典型场景：INT8 量化激活 + FP32 scale 同时 gather。

## 前置条件

- CANN 9.0+ (Atlas A5 / Ascend 950)
- Python >= 3.10
- PyTorch >= 2.6 + torch_npu >= 2.6

## 安装

预编译 wheel（推荐）：

```bash
pip install custom_comm
```

从源码安装（需要 CANN toolkit）：

```bash
source ~/Ascend/set_env.sh
pip install -e .
```

## 快速开始

```python
import torch, torch_npu, custom_comm

torch.distributed.init_process_group(backend="hccl")
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.npu.set_device(rank)

pg = torch.distributed.group.WORLD
backend = pg._get_backend(torch.device(f"npu:{rank}"))
hcom = backend.get_hccl_comm_name(rank)

# 单次调用同时 AllGather INT8 数据和 FP32 scale
activations = torch.randn(2048, device="npu").to(torch.int8)
scales = torch.randn(4, device="npu", dtype=torch.float32)
outputs = custom_comm.allgather_batch([activations, scales], hcom, world_size)
```

## 执行路径

| `CUSTOM_COMM_USE_CCU` | 策略 |
|---|---|
| 未设置 | Decomposed: pack → HcclAllGather → unpack |
| `1` | CCU Kernel: 单次 CCU launch, zero-copy RDMA gather |

## Graph Mode

```python
@torch.compile(backend="npu")
def fn(x, s, hcom, ws):
    return custom_comm.allgather_batch([x, s], hcom, ws)
```

Converter 将 `allgather_batch` 分解为多个 `HcomAllGather` GE 算子。

## 测试

```bash
pytest tests/ -k "meta"                       # shape inference (无需 NPU)
torchrun --nproc_per_node=N pytest tests/      # NPU 功能测试
torchrun --nproc_per_node=N tests/bench_allgather_batch.py  # 性能基准
```

## License

Apache-2.0
