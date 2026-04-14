================================================================================
  custom_comm -- CANN 自定义通信算子库
================================================================================

概述
----
custom_comm 提供面向昇腾 NPU 的高性能融合通信算子, 作为 PyTorch custom op
通过 torch_npu 集成, 支持 eager mode 和 torch.compile graph mode.

当前算子:
  allgather_batch  将至多 8 个异构 dtype 的 tensor 在一次调用中完成 AllGather,
                   避免多次独立 AllGather 的启动开销.

目标平台: Atlas A5 (Ascend 950) / CANN 9.0


前置依赖
--------
  Python        >= 3.10
  PyTorch       >= 2.6
  torch_npu     >= 2.6     (需与 CANN 版本匹配)
  CANN toolkit  9.0        (标准安装即可, 无需额外 SDK)


安装
----
1) 预编译 wheel (推荐):

    pip install custom_comm

2) 从源码安装 (需要 CANN toolkit):

    source /path/to/Ascend/set_env.sh
    pip install -e .

3) 仅编译 C++ 库 (不含 Python 绑定):

    cmake -B build && cmake --build build


快速上手
--------
    import torch, torch_npu, custom_comm

    # 初始化分布式
    torch.distributed.init_process_group(backend="hccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.npu.set_device(rank)

    # 获取 HCCL comm group 名称
    pg = torch.distributed.group.WORLD
    hcom = pg._get_backend(torch.device(f"npu:{rank}")).get_hccl_comm_name(rank)

    # 单次调用同时 AllGather INT8 数据和 FP32 scale
    data   = torch.randint(0, 127, (2048,), dtype=torch.int8, device="npu")
    scales = torch.randn(4, dtype=torch.float32, device="npu")
    gathered = custom_comm.allgather_batch([data, scales], hcom, world_size)
    # gathered[0].shape == (2048 * world_size,)
    # gathered[1].shape == (4 * world_size,)


执行路径
--------
环境变量 CUSTOM_COMM_USE_CCU 控制执行路径:
- 默认 (Phase 1):  将多个 tensor pack 成连续 buffer, 调一次 HcclAllGather, 再 unpack
- =1   (Phase 2):  注册 CCU kernel, 通过 HComm CCU 指令直接在 NPU 上做 zero-copy gather


Graph Mode (torchair)
---------------------
custom_comm 注册了 GE converter, 在 torch.compile 图模式下可用:

    @torch.compile(backend="npu")
    def fn(x, s, hcom, ws):
        return custom_comm.allgather_batch([x, s], hcom, ws)

当前 converter 将 batched op 拆分为多次 HcomAllGather GE 算子调用
(GE 无原生 batched AllGather 支持).


测试
----
    # shape/dtype 元信息测试 (无需 NPU)
    pytest tests/ -k "meta or Meta"

    # NPU 功能测试 (需要多卡环境)
    torchrun --nproc_per_node=N pytest tests/

    # 性能基准
    torchrun --nproc_per_node=8 tests/bench_allgather_batch.py


目录结构
────────
    custom_comm/
      CMakeLists.txt              构建配置
      setup.py                    Python包安装
      ops/allgather_batch/
        inc/                      C头文件 (公开C API + 内部定义)
        src/                      C++实现 (dispatch, decomposed, CCU kernel)
      torch_ext/csrc/             PyTorch op注册 + Meta/NPU dispatch
      python/custom_comm/         Python包
        ops.py                    用户API
        converters/               torchair GE graph-mode converter
      tests/                      pytest测试套件 + 性能基准
      docs/                       设计文档, 架构图, FAQ


许可证
------
Apache-2.0
