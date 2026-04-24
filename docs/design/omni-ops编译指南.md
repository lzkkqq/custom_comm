# omni-ops 编译指南（蓝区）

记录在蓝区把 `omni-ops/training/ascendc` 编译成 CANN 安装包，并安装到 CANN SDK、让 Python `import omni_training_custom_ops` 工作的完整流程。`training/pypto` 和 `training/triton` 不在本文范围。

## 0. 速查

```bash
# 一次性
ssh bluezone
source ~/venv/bin/activate
export https_proxy=http://127.0.0.1:10077 http_proxy=http://127.0.0.1:10077
pip install decorator attrs psutil cloudpickle sympy scipy

# 每次编译
rsync -av --delete --exclude='.git/' --exclude='build/' --exclude='output/' \
  ~/repo/vllm-project/omniai/omni-ops/ bluezone:~/code/omni-ops/

ssh bluezone
source ~/venv/bin/activate
source ~/Ascend/cann-9.0.0/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/lib64:$LD_LIBRARY_PATH

cd ~/code/omni-ops/training/ascendc && bash build.sh
./output/CANN-omni_training_custom_ops-*.run
cd ../../training/ascendc/torch_ops_extension && bash build_and_install.sh
```

## 1. 机器与路径约定

| 项         | 位置                                          |
| ---------- | --------------------------------------------- |
| CANN SDK   | `~/Ascend/cann-9.0.0`（完整 toolkit）         |
| Python     | `~/venv`（Python 3.12，torch/torch_npu 预装） |
| bisheng    | `$ASCEND_HOME_PATH/bin/bisheng`               |
| 本机代理   | `127.0.0.1:10808`（供蓝区反向映射到 10077）   |
| 源码同步位 | `bluezone:~/code/omni-ops/`                  |

源代码以本机为权威：`~/repo/vllm-project/omniai/omni-ops/`。蓝区是纯编译机。

## 2. 一次性准备

### 2.1 蓝区 Python 依赖

bisheng 代码生成阶段会起 Python 子进程。蓝区 `~/venv` 默认缺 scipy 等，要先补齐：

    ssh bluezone
    source ~/venv/bin/activate
    export https_proxy=http://127.0.0.1:10077 https_proxy=http://127.0.0.1:10077
    pip install numpy scipy decorator sympy attrs psutil cloudpickle

这里 `10077` 是 ssh config 里 `bluezone` 的 RemoteForward 端口，把本机代理反向映射到蓝区。

### 2.2 CANN 环境

每个新 shell 都要 source：

    source ~/venv/bin/activate
    source ~/Ascend/cann-9.0.0/set_env.sh
    export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/lib64:$LD_LIBRARY_PATH

`set_env.sh` 会设 `$ASCEND_HOME_PATH`、把 bisheng/ccec 加到 PATH。`LD_LIBRARY_PATH` 要手动补 `x86_64-linux/lib64`，里面是 `libascendcl.so`、`libopapi.so` 等。

## 3. 同步源代码到蓝区

```bash
rsync -av --delete \
    --exclude='.git/' --exclude='build/' --exclude='output/' \
    --exclude='__pycache__/' --exclude='*.egg-info/' \
    ~/repo/vllm-project/omniai/omni-ops/ \
    bluezone:~/code/omni-ops/
```

## 4. 编译生成 .run 包

```bash
ssh bluezone
source ~/venv/bin/activate
source ~/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/lib64:$LD_LIBRARY_PATH

cd ~/code/omni-ops/training/ascendc
rm -rf build output
bash build.sh 2>&1 | tee /tmp/build.log
```

成功后生成：
- `build/CANN-omni_training_custom_ops-*.run`（约 195 MB）
- `output/CANN-omni_training_custom_ops-*.run`（同名副本）
- `build/libcust_opapi.so`、`libcust_opsproto_rt2.0.so`、`libcust_opmaster_rt2.0.so`

## 5. 安装 .run 到 CANN vendors

```bash
cd ~/code/omni-ops/training/ascendc/output
./CANN-omni_training_custom_ops--linux.x86_64.run --install-path=$ASCEND_HOME_PATH
```

这一步把算子装到 `$ASCEND_HOME_PATH/vendors/omni_training_custom_ops/` 下，包含 `op_api/`、`op_proto/`、`op_impl/`、`op_tiling/`。

## 6. 构建 Python 扩展（torch_ops_extension）

CANN 安装完后，编译 Python 绑定：

    cd ~/code/omni-ops/training/ascendc/torch_ops_extension
    source ~/venv/bin/activate
    source ~/Ascend/cann-9.0.0/set_env.sh
    export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/lib64:$LD_LIBRARY_PATH
    bash build_and_install.sh   # 内部跑 python setup.py install

完成后：

    python -c "import omni_training_custom_ops"

会从 `~/venv/lib/python3.12/site-packages/omni_training_custom_ops/` 加载。

### 6.1 运行期 LD_LIBRARY_PATH

import 时要拉到三类 `.so`。缺任何一类都会 fail：

| 符号出处                        | `.so` 所在                                               |
| ------------------------------- | -------------------------------------------------------- |
| libtorch / libc10               | `$(python -c "import torch,os;print(os.path.dirname(torch.__file__)+\"/lib\")")` |
| libtorch_npu                    | `$(python -c "import torch_npu,os;print(os.path.dirname(torch_npu.__file__)+\"/lib\")")` |
| libhccl、libascendcl 等         | `$ASCEND_HOME_PATH/x86_64-linux/lib64`       |

推荐 shell rc 里固化一份：

```bash
source ~/venv/bin/activate
source ~/Ascend/cann-9.0.0/set_env.sh
TORCH_LIB=$(python3 -c "import os,torch;print(os.path.dirname(torch.__file__)+'/lib')")
TORCH_NPU_LIB=$(python3 -c "import os,torch_npu;print(os.path.dirname(torch_npu.__file__)+'/lib')")
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/lib64:$TORCH_LIB:$TORCH_NPU_LIB:$LD_LIBRARY_PATH
```

## 7. 验证

```bash
python -c "import omni_training_custom_ops; print(omni_training_custom_ops.__file__)"
```

预期输出：
```
/home/fan33/venv/lib/python3.12/site-packages/omni_training_custom_ops/__init__.py
```

随后 `torch.ops.omni_training_custom_ops.*` 或 `torch_npu.*` 即可调用算子（具体调用接口看 `torch_ops_extension/omni_training_custom_ops/` 下每个子包的 `__init__.py`）。

> 注意：蓝区没有 NPU 硬件，能 `import` 但 op 实际执行会抛 `aclError`。真正跑 op 需要切到有卡环境。

## 8. 产物清单

| 产物                                                       | 位置                                                                | 作用                        |
| ----                                                       | ----                                                                | ----                        |
| `CANN-omni_training_custom_ops-*.run`                      | `training/ascendc/{build,output}/`                                  | 自解压安装包（~195 MB）     |
| `libcust_opapi.so`, `libcust_opsproto_rt2.0.so`, `libcust_opmaster_rt2.0.so` | `training/ascendc/build/`                                           | ops 的 host-side 动态库     |
| vendor 安装目录                                            | `$ASCEND_HOME_PATH/vendors/omni_training_custom_ops/`           | `.run --install` 后的落盘   |
| Python wheel                                               | `training/ascendc/torch_ops_extension/dist/omni_training_custom_ops-*.whl` | Python 包                 |
| 已装 Python 包                                             | `site-packages/omni_training_custom_ops/`                           | `import` 入口               |

## 9. 踩坑速查

| 现象 | 位置 | 原因 / 解决 |
|------|------|-------------|
| `ModuleNotFoundError: numpy`（build 阶段） | bisheng 代码生成 | venv 里缺依赖，跑 §2.1 |
| `.json not generated!` / `generate_opinfo_config.py` 报错 | cmake 多处子命令级联失败 | 上游是缺 numpy/scipy，日志里找最早的 ImportError |
| `ImportError: libhccl.so` | import torch_npu 或 import omni_training_custom_ops | `LD_LIBRARY_PATH` 没补 `$ASCEND_HOME_PATH/x86_64-linux/lib64` |
| `ImportError: libc10.so` | import omni_training_custom_ops | `LD_LIBRARY_PATH` 没补 torch 的 `lib/` |
| `ImportError: libtorch_npu.so` / `libtorch_python.so` | 同上 | 补 torch_npu 的 `lib/` 到 `LD_LIBRARY_PATH` |
| `pip install` 403 / connection refused | venv 依赖 | 本机 ssh 的 RemoteForward 没开，或者没 `export https_proxy=http://127.0.0.1:10077` |
| `omni_training_custom_ops has no attribute xxx` | import 后调不到新算子 | 算子没注册到 `torch.ops.custom`，检查 setup.py 是否漏挂新算子的 `.cpp` |

## 10. 参考

- `build.sh` 参数：`./build.sh --help`（支持 `-n` 限定算子名、`-c` 限定芯片、`--opapi` 启 UT 等）
- 算子源码：`training/ascendc/src/`
- vendor 目录结构：`$ASCEND_HOME_PATH/vendors/omni_training_custom_ops/`（含 `op_api/`、`op_impl/`、`op_proto/`）
- Python 绑定源码：`training/ascendc/torch_ops_extension/omni_training_custom_ops/`
- 本机 SSH 代理约定：`~/.ssh/config` 里 `bluezone` 段配置 `RemoteForward 10077 127.0.0.1:<本机代理端口>`