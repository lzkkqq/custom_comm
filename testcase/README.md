# custom_comm HcclAllGatherBatch 单板测试用例

该目录是一个独立 C++ 上板测试程序，只测试 `HcclAllGatherBatch` C API。它不走 Python、Torch、pytest，也不需要改 `custom_comm` 其他目录，适合在 A5 单机环境直接验证接口功能和基础性能。

## 1. 测试程序做什么

测试程序会在 `--device-list` 指定的每张物理卡上启动一个线程，每个线程初始化自己的 HCCL rank，然后共同调用一次或多次 `HcclAllGatherBatch`。

默认测试两个 item：

| item | dataType | 默认规模 | sendCount 含义 | 校验方式 |
| --- | --- | --- | --- | --- |
| token | `HCCL_DATA_TYPE_INT8` | `327680` bytes | INT8 元素个数，也等于字节数 | 每个 rank 写不同 byte pattern，AllGather 后检查 `[rank0, rank1, ...]` |
| scale | `HCCL_DATA_TYPE_FP32` | `128` 个 float | FP32 元素个数 | 每个 rank 写 `rank * 1000 + index`，AllGather 后逐元素检查 |

默认 `--desc-count=2`，所以会同时覆盖 `INT8 token` 和 `FP32 scale`。如果指定 `--desc-count=1`，只测 `INT8 token`。

## 2. 目录和产物

| 文件 | 作用 |
| --- | --- |
| `main.cc` | 测试程序源码，负责解析参数、初始化 ACL/HCCL、分配 device/host 内存、调用 `HcclAllGatherBatch`、同步并校验结果 |
| `Makefile` | 独立构建脚本，编译 `main.cc` 并链接 `libcustom_comm_impl.so`、`libhccl.so`、`libhcomm.so`、`libascendcl.so` |
| `run.sh` | 运行封装脚本，自动补 `LD_LIBRARY_PATH`，并通过第一个参数选择 decomposed/CCU/CCU MS 路径 |
| `README.md` | 当前说明文档 |

构建完成后会生成：

```text
custom_comm/testcase/custom_comm_allgather_batch_testcase
```

## 3. 环境变量含义

| 变量 | 是否必需 | 含义 | 默认值 |
| --- | --- | --- | --- |
| `ASCEND_CANN_PACKAGE_PATH` | 建议设置 | CANN SDK 根目录，Makefile 会从这里找 ACL/HCCL/HCOMM 头文件和库 | 如果未设置，使用 `ASCEND_HOME_PATH`；两者都没有则使用 `/usr/local/Ascend/ascend-toolkit/latest` |
| `ASCEND_HOME_PATH` | 可选 | CANN SDK 根目录的兼容变量，低优先级备用 | 无 |
| `CUSTOM_COMM_LIB_DIR` | 可选 | `libcustom_comm_impl.so` 所在目录 | `custom_comm/python/custom_comm` |
| `LD_LIBRARY_PATH` | 运行时需要 | 动态库搜索路径；`run.sh` 会自动追加 custom_comm 和 CANN lib 目录 | 继承当前环境 |
| `CUSTOM_COMM_USE_CCU` | 运行时可选 | `1` 表示走 CCU 路径；不设置表示走 decomposed 路径 | 不设置 |
| `CUSTOM_COMM_CCU_MODE` | 运行时可选 | `ms` 表示 CCU MS 后端；不设置时 CCU 走默认 sched 后端 | 不设置 |

建议上板前先设置 SDK 路径：

```bash
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

如果 `libcustom_comm_impl.so` 不在默认目录，需要指定：

```bash
export CUSTOM_COMM_LIB_DIR=/path/to/custom_comm/python/custom_comm
```

## 4. 第一步：构建 custom_comm 动态库

测试程序链接的是 `custom_comm/python/custom_comm/libcustom_comm_impl.so`，所以要先在 `custom_comm` 根目录构建这个库。

从仓库根目录进入 `custom_comm`：

```bash
cd custom_comm
python3 setup.py build_ext --inplace
```

构建成功后确认动态库存在：

```bash
ls -l python/custom_comm/libcustom_comm_impl.so
```

如果这里不存在，后续 `make` 会报：

```text
missing .../custom_comm/python/custom_comm/libcustom_comm_impl.so
please build custom_comm first, e.g. cd .. && python3 setup.py build_ext --inplace
```

## 5. 第二步：编译 testcase

进入 testcase 目录：

```bash
cd testcase
make
```

也可以从仓库根目录直接执行：

```bash
cd custom_comm/testcase
make
```

`make` 实际做了三件事：

1. 检查 `$(CUSTOM_COMM_LIB_DIR)/libcustom_comm_impl.so` 是否存在。
2. 使用 `g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0` 编译 `main.cc`。
3. 链接 `-lcustom_comm_impl -lhccl -lhcomm -lascendcl -lpthread` 生成可执行文件。

如果需要看完整编译命令，可以执行：

```bash
make clean
make V=1
```

如果要手动指定 SDK 或库目录：

```bash
make clean
make ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest \
     CUSTOM_COMM_LIB_DIR=/path/to/custom_comm/python/custom_comm
```

清理 testcase 产物：

```bash
make clean
```

## 6. 第三步：运行 testcase

推荐使用 `bash run.sh`，因为它会自动设置运行时库路径。

默认运行 decomposed 路径，跑 0、1 两张卡，`desc-count=2`：

```bash
bash run.sh
```

只跑 4-7 卡：

```bash
bash run.sh --device-list 4,5,6,7
```

跑 8 卡，计时 10 次：

```bash
bash run.sh --device-list 0,1,2,3,4,5,6,7 --iters 10
```

走 CCU sched 路径：

```bash
bash run.sh ccu --device-list 0,1,2,3 --iters 10
```

走 CCU MS 路径：

```bash
bash run.sh ccu-ms --device-list 0,1,2,3 --iters 10
```

只做功能冒烟，缩小 token 输入：

```bash
bash run.sh --device-list 0,1 --bytes 4096 --scale-count 128 --iters 1
```

跳过 host 侧结果校验，只看接口是否能正常提交和同步：

```bash
bash run.sh --device-list 0,1 --no-verify
```

## 7. run.sh 的模式参数

`run.sh` 的第一个参数如果是 `decomposed`、`ccu`、`ccu-ms`，会被当作路径选择模式；其余参数原样传给测试程序。

| 命令 | 设置的环境变量 | 实际路径 |
| --- | --- | --- |
| `bash run.sh ...` | 清理 `CUSTOM_COMM_USE_CCU` 和 `CUSTOM_COMM_CCU_MODE` | decomposed |
| `bash run.sh decomposed ...` | 清理 `CUSTOM_COMM_USE_CCU` 和 `CUSTOM_COMM_CCU_MODE` | decomposed |
| `bash run.sh ccu ...` | `CUSTOM_COMM_USE_CCU=1` | CCU sched |
| `bash run.sh ccu-ms ...` | `CUSTOM_COMM_USE_CCU=1 CUSTOM_COMM_CCU_MODE=ms` | CCU MS |

如果不使用 `run.sh`，也可以手动运行：

```bash
export LD_LIBRARY_PATH="${CUSTOM_COMM_LIB_DIR}:${ASCEND_CANN_PACKAGE_PATH}/lib64:${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64:${ASCEND_CANN_PACKAGE_PATH}/hcomm/hcomm/lib64:${LD_LIBRARY_PATH}"
./custom_comm_allgather_batch_testcase --device-list 0,1
```

## 8. 测试程序参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device-list` | `0,1` | 物理卡列表，最多 8 张；逻辑 rank 为列表下标，例如 `4,5,6,7` 对应逻辑 rank `0..3` |
| `--desc-count` | `2` | `1` 表示只测 `INT8 token`；`2` 表示同时测 `INT8 token` 和 `FP32 scale` |
| `--bytes` | `327680` | 每个 rank 的 INT8 token 字节数；因为 dtype 是 INT8，所以也等于 sendCount |
| `--scale-count` | `128` | 每个 rank 的 FP32 scale 元素个数；实际字节数是 `scale-count * 4` |
| `--warmup` | `1` | 计时前 warmup 次数；warmup 会调用接口并同步，但不计入平均耗时 |
| `--iters` | `1` | 计时迭代次数；平均耗时为总耗时除以该值 |
| `--no-verify` | 关闭 | 跳过 host 侧结果校验，适合只测 launch 或排查同步问题 |
| `--help` | 无 | 打印参数说明 |

`--device-list` 指的是物理卡号，不是 HCCL rank。比如：

```bash
bash run.sh --device-list 4,5,6,7
```

程序会使用物理卡 `4,5,6,7`，但逻辑 rank 分别是：

| 物理卡 | 逻辑 rank |
| --- | --- |
| 4 | 0 |
| 5 | 1 |
| 6 | 2 |
| 7 | 3 |

## 9. 预期输出

成功时会先打印本次配置：

```text
custom_comm HcclAllGatherBatch testcase starts, deviceList=0,1,2,3, rankSize=4, descCount=2, tokenBytes=327680, scaleCount=128, warmup=1, iters=10, verify=on
```

rank0 会打印平均耗时和估算带宽：

```text
avgTime(us)=..., dataSize(B)=..., algoBandwidth(GB/s)=...
```

每个 rank 校验成功后会打印：

```text
[rank 0, device 0] verify success
[rank 1, device 1] verify success
```

全部成功时最后打印：

```text
custom_comm HcclAllGatherBatch testcase finished successfully
```

## 10. 常见问题

### 10.1 找不到 `libcustom_comm_impl.so`

现象：

```text
missing .../custom_comm/python/custom_comm/libcustom_comm_impl.so
```

处理：

```bash
cd custom_comm
python3 setup.py build_ext --inplace
cd testcase
make
```

如果库在其他目录，设置：

```bash
export CUSTOM_COMM_LIB_DIR=/actual/path/to/libcustom_comm_impl_dir
```

### 10.2 运行时报 `error while loading shared libraries`

一般是 `LD_LIBRARY_PATH` 没有包含 custom_comm 或 CANN lib 目录。优先用：

```bash
bash run.sh --device-list 0,1
```

如果手动运行，先设置：

```bash
export LD_LIBRARY_PATH="${CUSTOM_COMM_LIB_DIR}:${ASCEND_CANN_PACKAGE_PATH}/lib64:${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64:${ASCEND_CANN_PACKAGE_PATH}/hcomm/hcomm/lib64:${LD_LIBRARY_PATH}"
```

### 10.3 `device is out of range`

说明 `--device-list` 里的物理卡号超过当前机器 ACL 能看到的设备数量。先确认设备：

```bash
npu-smi info
```

然后改成有效卡号，例如：

```bash
bash run.sh --device-list 0,1,2,3
```

### 10.4 校验失败

如果出现：

```text
token verify failed: srcRank=..., index=..., actual=..., expected=...
scale verify failed: srcRank=..., index=..., actual=..., expected=...
```

说明 `HcclAllGatherBatch` 返回成功，但输出内容和预期不一致。此时优先记录：

- 当前运行路径：decomposed、ccu 还是 ccu-ms。
- `--device-list`、`--desc-count`、`--bytes`、`--scale-count`、`--iters`。
- 第一个 mismatch 的 `srcRank` 和 `index`。

### 10.5 接口调用失败

如果出现：

```text
acl call failed: ...
hccl call failed: ...
```

说明 ACL/HCCL 接口返回非成功码。日志会打印失败调用、源码行号和返回码，可先根据失败点区分是设备初始化、comm 初始化、内存分配、接口调用还是同步阶段失败。
