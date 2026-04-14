# custom_comm

面向昇腾 NPU 的自定义通信算子库，基于 HCCL/HComm 构建，以 PyTorch custom op
形式提供，支持 eager mode、torch.compile 和 torchair graph mode。

## 算子

- allgather_batch: 单次调用完成至多 8 个异构 dtype tensor 的 AllGather，
  避免多次独立调用的 launch 开销。典型场景：INT8 量化数据 + FP32 scale 的聚合。

## 前置条件

- CANN 9.0+ toolkit（Atlas A5 / Ascend 950）
- PyTorch >= 2.6 + torch_npu >= 2.6
- Python >= 3.10

## 安装

预编译 wheel（推荐）:

    pip install custom_comm

从源码安装（需要 CANN toolkit）:

    source ~/Ascend/set_env.sh
    pip install -e .

## 快速上手

    import torch, torch_npu, custom_comm

    torch.distributed.init_process_group(backend="hccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    pg = torch.distributed.group.WORLD
    hcom = pg._get_backend(torch.device(f"npu:{rank}")).get_hccl_comm_name(rank)

    activations = torch.randn(2048, dtype=torch.int8, device="npu")
    scales = torch.randn(4, dtype=torch.float32, device="npu")
    outputs = custom_comm.allgather_batch([activations, scales], hcom, world_size)

graph mode 同样支持：

    @torch.compile(backend="npu")
    def step(x, s, hcom, ws):
        return custom_comm.allgather_batch([x, s], hcom, ws)

## 执行路径

  CUSTOM_COMM_USE_CCU  策略
  (unset)              Decomposed: 打包为连续 buffer, 调一次 HcclAllGather, 解包
  1                    CCU batched: 单个 CCU kernel 直接 RDMA gather, 无 pack/unpack

## 测试

    pytest tests/ -k "meta"                    # shape inference (无需 NPU)
    torchrun --nproc_per_node=N pytest tests/  # NPU 功能测试
    torchrun --nproc_per_node=N tests/bench_allgather_batch.py  # 性能基准

## 许可证

Apache-2.0
