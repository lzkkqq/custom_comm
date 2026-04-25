# custom_comm HcclAllGatherBatch 上板测试说明

该目录提供一个独立 C++ testcase，只测试 `HcclAllGatherBatch` 对外接口，并支持和标准 `HcclAllGather` 做总吞吐基线对比。整体使用方式对齐 `hccl/allgatherbatch/testcase`，但入口保持 `custom_comm` 当前 C API，不依赖 Python、Torch 或 pytest。

## 1. 测什么

这个 testcase 只覆盖 device 侧数据路径：

- `custom` 模式：直接调用 `HcclAllGatherBatch`
- `baseline` 模式：把所有 item 的输入按字节拼成一个大 `INT8` buffer，再调用一次标准 `HcclAllGather`
- `both` 模式：同一组输入先跑 `custom`，再跑 `baseline`，最后输出 `delta(us)` 和 `speedup`

这里的 baseline 不是逐 item 单独调多次 `HcclAllGather`，而是“总字节量等价”的单次大 AllGather，对比口径更稳定，也更适合做性能基线。

## 2. 目录和产物

| 文件 | 作用 |
| --- | --- |
| `main.cc` | 参数解析、ACL/HCCL 初始化、device/host buffer 分配、benchmark 计时、结果校验 |
| `Makefile` | 独立构建脚本，链接 `libcustom_comm_impl.so`、`libhccl.so`、`libhcomm.so`、`libascendcl.so` |
| `run.sh` | 运行封装，自动设置 `LD_LIBRARY_PATH`，并切换 `decomposed` / `ccu` / `ccu-ms` |
| `README.md` | 当前使用说明 |

构建后生成：

```text
custom_comm/testcase/custom_comm_allgather_batch_testcase
```

## 3. 环境变量含义

| 变量 | 是否必需 | 含义 |
| --- | --- | --- |
| `ASCEND_CANN_PACKAGE_PATH` | 建议设置 | CANN SDK 根目录，`Makefile` 用它查找 ACL/HCCL/HCOMM 头文件和库 |
| `ASCEND_HOME_PATH` | 可选 | `ASCEND_CANN_PACKAGE_PATH` 的兼容兜底变量 |
| `CUSTOM_COMM_LIB_DIR` | 可选 | `libcustom_comm_impl.so` 所在目录，默认是 `custom_comm/python/custom_comm` |
| `LD_LIBRARY_PATH` | 运行时需要 | 动态库搜索路径，`run.sh` 会自动补齐 |
| `CUSTOM_COMM_USE_CCU` | 运行时可选 | `1` 表示走 CCU 路径，不设置表示走 decomposed |
| `CUSTOM_COMM_CCU_MODE` | 运行时可选 | `ms` 表示走 CCU MS 后端，不设置表示 CCU sched |
| `HCCL_OP_EXPANSION_MODE` | 运行时可选 | communicator 的 op expansion mode，`run.sh ccu` 和 `run.sh ccu-ms` 会自动设置 |

建议先明确 SDK 根目录：

```bash
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

如果 `libcustom_comm_impl.so` 不在默认目录，再补：

```bash
export CUSTOM_COMM_LIB_DIR=/path/to/custom_comm/python/custom_comm
```

## 4. 第一步：构建 custom_comm 动态库

testcase 依赖 `custom_comm/python/custom_comm/libcustom_comm_impl.so`，所以要先在 `custom_comm` 根目录构建这个库：

```bash
cd custom_comm
python3 setup.py build_ext --inplace
```

构建成功后确认产物存在：

```bash
ls -l python/custom_comm/libcustom_comm_impl.so
```

如果当前环境没有 `torch_npu`，现在的 `setup.py` 也会继续构建 `libcustom_comm_impl.so`，只是跳过 Python 扩展 `_C*.so`。对这个 testcase 来说，关键产物就是 `libcustom_comm_impl.so`。

## 5. 第二步：编译 testcase

进入 testcase 目录：

```bash
cd custom_comm/testcase
make
```

如果想显式指定 SDK 和 custom_comm 库目录：

```bash
make clean
make ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest \
     CUSTOM_COMM_LIB_DIR=/path/to/custom_comm/python/custom_comm
```

`make` 主要做三件事：

1. 检查 `$(CUSTOM_COMM_LIB_DIR)/libcustom_comm_impl.so` 是否存在
2. 用 `g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0` 编译 `main.cc`
3. 链接 `-lcustom_comm_impl -lhccl -lhcomm -lascendcl -lpthread`

清理产物：

```bash
make clean
```

## 6. 第三步：运行 testcase

推荐统一用 `run.sh`，它会自动补运行时库路径，并根据模式设置 communicator 相关环境变量。

默认运行 decomposed 路径：

```bash
bash run.sh
```

只跑 4-7 卡：

```bash
bash run.sh --device-list 4,5,6,7
```

跑 `custom` 与标准 `AllGather` 基线对比：

```bash
bash run.sh --device-list 4,5,6,7 --mode both
```

走 CCU sched：

```bash
bash run.sh ccu --device-list 4,5,6,7 --mode both
```

走 CCU MS：

```bash
bash run.sh ccu-ms --device-list 4,5,6,7 --mode both
```

## 7. 新参数模型

这个 testcase 已经完全切到重复 `--item` 模式，不再使用旧的 `--desc-count`、`--bytes`、`--scale-count`。

### 7.1 `--item`

`--item` 可以重复出现，每个 item 的格式是：

```text
dtype:count
```

说明：

- `dtype` 是 HCCL 数据类型名字
- `count` 是这个 item 的元素个数，不是字节数
- 多个 `--item` 的顺序，就是传给 `HcclAllGatherBatch` 的 `descs[]` 顺序
- item 数上限跟当前实现保持一致，最多 `6` 个

支持的 dtype 名字：

```text
int8 uint8 int16 uint16 fp16 bf16 int32 uint32 fp32 int64 uint64 fp64 int128 hif8 fp8e4m3 fp8e5m2 fp8e8m0
```

默认 item 是两项：

```text
--item int8:327680 --item fp32:128
```

### 7.2 其他参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device-list` | `0,1` | 物理卡号列表，逻辑 rank 是列表下标 |
| `--warmup` | `1` | warmup 次数 |
| `--iters` | `1` | 计时次数 |
| `--mode` | `custom` | `custom` / `baseline` / `both` |
| `--timing-mode` | `device` | `host` / `device` |
| `--no-verify` | 关闭 | 跳过 host 侧结果校验 |

`--timing-mode` 说明：

- `device`：默认模式，参考 `hccl/allgatherbatch/testcase`，使用额外 stream/event 做 gate，尽量把计时收敛到 device 执行段
- `host`：更接近整条提交流程，直接以业务 stream 上的 event 计时

## 8. 常用示例

默认两 item，跑 decomposed：

```bash
bash run.sh --device-list 4,5,6,7
```

显式指定两 item，做 custom 与 baseline 对比：

```bash
bash run.sh --device-list 4,5,6,7 \
    --item int8:327680 \
    --item fp32:128 \
    --mode both
```

三 item，小数据量功能验证：

```bash
bash run.sh --device-list 0,1 \
    --item int8:4096 \
    --item fp16:256 \
    --item bf16:256 \
    --iters 1
```

只做 device 侧 benchmark，不回读校验：

```bash
bash run.sh --device-list 4,5,6,7 \
    --item int8:327680 \
    --item fp32:128 \
    --mode both \
    --timing-mode device \
    --no-verify
```

CCU MS 路径：

```bash
bash run.sh ccu-ms --device-list 4,5,6,7 \
    --item int8:327680 \
    --item fp32:128 \
    --mode both
```

## 9. `run.sh` 的模式语义

`run.sh` 第一个参数如果是 `decomposed`、`ccu`、`ccu-ms`，会被当作执行路径；其余参数原样透传给 testcase。

| 命令 | 自动设置的环境变量 | 实际路径 |
| --- | --- | --- |
| `bash run.sh ...` | 清理 `CUSTOM_COMM_USE_CCU`、`CUSTOM_COMM_CCU_MODE`、`HCCL_OP_EXPANSION_MODE` | decomposed |
| `bash run.sh decomposed ...` | 同上 | decomposed |
| `bash run.sh ccu ...` | `CUSTOM_COMM_USE_CCU=1`，`HCCL_OP_EXPANSION_MODE=CCU_SCHED` | CCU sched |
| `bash run.sh ccu-ms ...` | `CUSTOM_COMM_USE_CCU=1`，`CUSTOM_COMM_CCU_MODE=ms`，`HCCL_OP_EXPANSION_MODE=CCU_MS` | CCU MS |

同时，testcase 本身也会根据 `CUSTOM_COMM_USE_CCU` / `CUSTOM_COMM_CCU_MODE` 在 communicator 初始化时切换：

- decomposed：`HcclCommInitRootInfo`
- ccu：`HcclCommInitRootInfoConfig + hcclOpExpansionMode=6`
- ccu-ms：`HcclCommInitRootInfoConfig + hcclOpExpansionMode=5`

## 10. 输出怎么看

启动时会先打印本次配置，例如：

```text
custom_comm HcclAllGatherBatch testcase starts, deviceList=4,5,6,7, rankSize=4, items=int8:327680,fp32:128, warmup=1, iters=10, mode=both, timingMode=device, verify=on
mode       | dataSize(B)        | avgTime(us)    | algoBandwidth(GB/s)  | status
```

如果 `mode=both`，rank0 会打印两行 benchmark 结果：

```text
custom     | ...
baseline   | ...
compare    | delta(us)=..., speedup=...x
```

其中：

- `dataSize(B)`：按“单 rank 总发送字节数 × rankSize”统计
- `avgTime(us)`：平均耗时
- `algoBandwidth(GB/s)`：按同一口径估算的算法带宽
- `speedup`：`baseline / custom`

如果开启校验，每个 rank 还会打印：

```text
[rank 0, device 4] verify success
```

最终成功结束时打印：

```text
custom_comm HcclAllGatherBatch testcase finished successfully
```

## 11. 校验逻辑

校验使用的是“按字节的确定性模式”，不依赖某个 dtype 的浮点格式解释，因此：

- `custom` 路径会逐 item 校验每个源 rank 的输出片段
- `baseline` 路径会校验大 `INT8` 输出是否等于所有 item 输入按顺序拼接后的结果

这意味着：

- 可以同时覆盖多 item 拆分/拼接顺序是否正确
- 不会因为 host 侧 float/bfloat16 转换实现差异带来额外噪声

## 12. 常见问题

### 12.1 提示找不到 `libcustom_comm_impl.so`

先回到 `custom_comm` 根目录执行：

```bash
python3 setup.py build_ext --inplace
```

然后确认：

```bash
ls -l python/custom_comm/libcustom_comm_impl.so
```

### 12.2 动态库加载失败

优先直接使用：

```bash
bash run.sh ...
```

如果要手工运行，先补：

```bash
export LD_LIBRARY_PATH="${CUSTOM_COMM_LIB_DIR}:${ASCEND_CANN_PACKAGE_PATH}/lib64:${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64:${ASCEND_CANN_PACKAGE_PATH}/hcomm/hcomm/lib64:${LD_LIBRARY_PATH}"
```

### 12.3 `invalid item spec`

说明 `--item` 格式不对。合法示例：

```bash
--item int8:327680
--item fp32:128
--item bf16:256
```

不合法示例：

```bash
--item int8
--item fp32:
--item :128
--item fp32:0
```

### 12.4 校验失败

如果日志里出现：

```text
custom verify failed: ...
baseline verify failed: ...
```

说明接口返回成功，但输出内容与预期不一致。建议同时记录：

- 路径：decomposed / ccu / ccu-ms
- `--device-list`
- 所有 `--item`
- `--mode`
- `--timing-mode`
- 第一个 mismatch 的 `item`、`srcRank`、`byteOffset`

### 12.5 只想测 device 侧性能，不想做 host 回读

直接加：

```bash
--timing-mode device --no-verify
```

这样仍会分配并使用 device buffer，但不会把输出拷回 host 校验，更适合纯性能回归。
