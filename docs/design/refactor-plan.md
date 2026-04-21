# custom_comm 模块化重构计划

## 背景

custom_comm 当前仅承载 `allgather_batch` 一个算子，采用 dual-ABI 架构（shim ABI=0 + torch extension ABI=1，见 `CLAUDE.md` "Build/Testing" 段）。单算子阶段这一架构已能正常工作；但面向 reduce_scatter_batch、alltoall 等后续算子扩展时，现有布局会放大三类问题：

1. shim 层缺少跨算子可复用的抽象（registry、backend interface、共享 dtype / comm 查询）。
2. `ops/allgather_batch/src/*` 里 `HCCL vs CCU` 两条执行路径靠 `CUSTOM_COMM_USE_CCU` 环境变量 `if/else` 硬路由，违反 OCP。
3. torch_ext 层 `torch_ext/csrc/allgather_batch.cpp` 把 schema 注册、meta kernel、eager 路径、tensor → desc 转换全部混在一个 300+ 行文件里，无法按算子线性扩展。

本计划的目标：在不破坏现有 ABI firewall 的前提下，把 custom_comm 重构为"op × backend"正交扩展的结构，使后续每新增一个算子或一种 backend 实现，边际工作量集中在单个文件或目录。

## 参考资料

三个外部参考仓库和一个原则集：

- `docs/design/refactor-references/` 暂缺（如果要落地可放对标快照）
- torchcomms (`/Users/shanshan/repo/torch/torchcomms/`)：backend abstraction + dlopen-registered 动态后端；`TorchCommBackend` 基类 + `CommFactory::Register()` 模式
- TransformerEngine (`/Users/shanshan/repo/nvidia/transformerengine/`)：`transformer_engine/common/` 纯 C 核 + `transformer_engine/pytorch/` framework binding 的双层结构
- omni-ops (`/Users/shanshan/repo/vllm-project/omniai/omni-ops/`)：`inference/` × `op_host/op_kernel/` 的目录拆法
- 设计原则（本机路径，非仓库内）：`~/repo/_me/skills/vibe-cod/references/design-code-principles/{software-design,cpp-language}/`。评审者可暂忽略，后续可把用到的原则段落摘录到 `docs/design/principles-excerpt.md`。

## 设计原则映射

把即将做的改动和引用的原则对齐——每条都有依据：

| 改动 | 对应原则 | 来源文件 |
|------|---------|---------|
| Backend Registry + `IBackend` 抽象 | OCP、Strategy、Factory | `round1_solid.md`、`round2_creational_structural.md` |
| Op facade（`all_gather_batch.cc` 瘦身） | SRP、Facade | `round1_solid.md`、`round2_creational_structural.md` |
| shim/extension 双 .so 保留 | Pimpl、ABI stability | `round4_modularity_contracts.md`、`round2_value_memory.md` |
| C-struct + opaque handle 跨 ABI | DIP 的 C 投影、No value semantics across ABI | `cpp-language/round1_polymorphism.md`、`round3_behavioral.md` |
| `ops/<name>/backends/<impl>/` 目录层级 | Parnas 信息隐藏 | `round4_modularity_contracts.md` |
| 表驱动 dispatch（替代 if-ccu-else） | 查表 > 分支 | `round4_modularity_contracts.md` |
| 保留 meta-kernel 单文件 | YAGNI、不做过度抽象 | `round4_modularity_contracts.md` |

## 目标目录结构

对标 TransformerEngine（common/ + pytorch/ + jax/ 分层）、torchcomms（backend 目录平铺 + registry）、omni-ops（`ops/<name>/{op_host,op_impl}` 三段式）三方后融合：

```
custom_comm/
├── include/custom_comm/              # 公开 C ABI (ABI=0 shim 对外)
│   ├── custom_comm.h                 # 聚合包含
│   ├── types.h                       # POD: AgbDesc, AgbConfig, enum
│   ├── allgather_batch.h             # 现有公开 API (不动)
│   └── registry.h                    # 新: C 风格 backend 注册表
├── ops/                              # C++ 实现层 (ABI=0)
│   ├── _core/                        # 跨 op 共享基础设施
│   │   ├── backend.h                 # IBackend 纯虚接口 (internal)
│   │   ├── registry.{h,cc}           # backend 注册与查表 (internal)
│   │   ├── dtype.{h,cc}              # 从散落各处搬来的 HcclDataType <-> size
│   │   ├── limits.h                  # MAX_DESC_COUNT 等常量唯一定义点
│   │   ├── status.h                  # custom_comm_status_t enum
│   │   ├── logging.h                 # CUSTOM_COMM_LOG_* 宏
│   │   └── hccl_utils.{h,cc}         # HCCL 公共包装 (group 查询、status 检查)
│   └── allgather_batch/
│       ├── api.cc                    # C ABI 入口: HcclAllGatherBatch -> dispatch
│       ├── op.{h,cc}                 # C++ Op facade (desc validation + backend select)
│       └── backends/
│           ├── decomposed.cc         # IBackend impl，自注册
│           └── ccu_mesh1d.cc         # IBackend impl，自注册
├── kernels/                           # CCU kernel 源文件独立目录（便于 NVCC-equivalent 编译规则）
│   └── ccu/
│       └── ccu_allgather_batch_mesh1d.cc   # 原 ccu_kernel_*.cc 内容
torch_ext/
└── csrc/
    ├── registration.cc               # 单一 PYBIND11_MODULE + TORCH_LIBRARY
    ├── ops/
    │   └── allgather_batch.cpp       # Tensor -> desc 转换 + 调用 C API
    └── common/
        ├── comm_resolver.{h,cc}      # ProcessGroup → hcclComm_t
        ├── dtype_mapper.{h,cc}       # torch dtype → HcclDataType
        └── stream_ctx.{h,cc}         # 流作用域管理（原 DispatchGuard）
python/custom_comm/
└── ops/                              # 每算子一个 Python 模块
    └── allgather_batch.py
```

三条关键的边界：
1. `include/custom_comm/` 以下仅 C 类型 + `extern "C"` 函数。跨 ABI 边界的唯一合法接口层。
2. `ops/_core/` 不对外可见（internal）；所有跨 op 的 C++ 基础设施住在这里。
3. `torch_ext/` 只与 C-ABI 头打交道，不得 #include `ops/_core/`。

## 参考项目对照（决策依据）

| 问题 | 借鉴来源 | 决策 |
|------|---------|------|
| 如何加第二、第三个 op 不碰分发代码 | OCP + registry table (`round1_solid.md` §2, `round4_modularity_contracts.md` §5) | `BackendRegistry<OpId>` 表驱动 |
| 多 backend 并存 + 选择 | Strategy + Factory（torchcomms 的 `TorchCommBackend` 派生） | 每个 backend 一个类 + 宏自注册 |
| 多框架绑定扩展（暂不需要但不阻断） | TransformerEngine `common/pytorch/jax` 三分 | `torch_ext/` 独立目录；C-ABI 为唯一 IPC |
| Op 内部分层 | omni-ops `op_host` / `op_kernel` | `op.cc`（host 逻辑）× `backend/*.cc`（kernel 调度）分离 |
| ABI firewall | `round4_modularity_contracts.md §Pimpl`、TE `common/include/` 全 C | 保持现状，严格化：公开头禁 `std::` |
| 常量散落 | SSOT | `include/custom_comm/types.h` 集中 enum + POD；内部 `ops/_core/limits.h` 集中 MAX_DESC_COUNT 等 |

## 分阶段路线图

所有阶段单独成 PR，每阶段结束产物 `pytest tests/ -k "meta or abi_firewall"` 通过，且 bluezone smoke 通过。

### Phase 0: 准备（不改代码，0.5 天）

- 在 `docs/design/agb/` 目录下新增 `architecture.md`，固化当前 `allgather_batch.h` 作为 FROZEN C ABI 版本号 v1
- 写 `CHANGELOG.md` 开始跟踪 shim/ext 符号表
- blueprint: 用 `nm -CD --defined-only libcustom_comm_impl.so` 导出 baseline，后续每阶段比对确保 public symbol 集合不扩大

### Phase 1: 基础设施抽离（1 天）

目标：把当前散落的公共代码归位到 `ops/_core/` 下。不改对外 API，不改行为。

改动：
- 新建 `ops/_core/limits.h`：收入 `MAX_DESC_COUNT = 6`（现在的值，参考 `ops/allgather_batch/inc/allgather_batch.h`）、`DEFAULT_TIMEOUT_MS` 等
- 新建 `ops/_core/dtype.{h,cc}`：`HcclDataType ToHcclDtype(aclDataType)` 从 `torch_ext/csrc/allgather_batch.cpp` 抽出
- 新建 `ops/_core/logging.h`：CUSTOM_COMM_LOG(tag, fmt, ...) 宏（搬 `ccu_kernel_ag_mesh1d.cc` 里散落的 `printf`）
- 新建 `ops/_core/status.h`：统一 `custom_comm_status_t` enum

验收：`ripgrep --count 'ScalarTypeToHcclDataType'` 从 3+ 处降到 1 处；`pytest tests/ -v` 全绿。

### Phase 2: Backend 抽象与 registry（中风险，约 2 天）

目标：让新增 backend 不需要改动 `all_gather_batch.cc`。

改动：
1. 定义 C-ABI 安全的 backend 接口（避免 `std::function`、`std::string` 等 ABI 敏感类型）：
   ```c
   // ops/_core/backend.h
   typedef struct custom_comm_backend_v1 {
       const char* name;                                 // 例 "ccu_mesh1d"、"decomposed"
       uint32_t abi_version;                              // == CUSTOM_COMM_BACKEND_ABI_V1
       custom_comm_status_t (*prepare)(const AgbConfig*, void** state);
       custom_comm_status_t (*execute)(void* state, AgbDescArray*, HcclComm, aclrtStream);
       void (*destroy)(void* state);
       bool (*probe)(const AgbConfig*);                   // 可选；NULL 表示"总是可用"
   } custom_comm_backend_v1;
   ```
2. 实现 `BackendRegistry`（内部 C++）：
   - 接口：`Register(op_id, backend_v1*)`、`Find(op_id, name)`、`List(op_id)`
   - 内部用 `std::map<std::string, std::vector<custom_comm_backend_v1*>>`，但不出现在 header
3. `ops/allgather_batch/backends/decomposed.cc` 和 `ccu.cc` 各自在文件末尾通过 `__attribute__((constructor))` 自注册
4. `api.cc` 里的 dispatch 变成：
   ```c
   auto backends = BackendRegistry::Instance().List(kOpAllgatherBatch);
   auto* be = SelectBackend(backends, cfg);  // 按 name/env/capability 选
   return be->execute(be->state, descs, comm, stream);
   ```

### Backend 接口字段详解

| 字段 | 类型 | 调用时机 | 所有权与不变式 |
|---|---|---|---|
| `name` | `const char*` | 注册时填入 | 静态字符串字面量；不拷贝、不释放；作 `SelectBackend` 的匹配 key |
| `abi_version` | `uint32_t` | 注册时比对 | `== CUSTOM_COMM_BACKEND_ABI_V1`；不匹配则拒绝注册 |
| `prepare(cfg, &state)` | 函数指针 | 一次性；首次 `execute` 前 | lazy 构造缓存；`*state` 由 backend 分配，registry 仅透传 |
| `execute(state, desc[], n, comm, stream)` | 函数指针 | 每次 op 调用 | 线程安全；不得阻塞；不得在热路径分配 host 内存 |
| `destroy(state)` | 函数指针 | engine 卸载 | 幂等；释放 `prepare` 分配的全部资源 |
| `probe(cfg)` | 函数指针（可选） | selector 决策时 | 返回 bool，backend 是否支持此 cfg；缺省即总是支持 |

### 错误码（`custom_comm_status_t`）

| 值 | 语义 | 调用者应对 |
|----|------|-----------|
| `CC_OK` = 0 | 成功 | 继续 |
| `CC_ERR_INVALID_ARG` | descs 为空 / rank 越界 / dtype 不支持 | 立即 return，不切换 backend |
| `CC_ERR_UNSUPPORTED` | 当前 backend 不支持该 desc 组合（例如 CCU 不支持某 dtype） | dispatcher 回退到下一个候选 backend |
| `CC_ERR_RESOURCE` | HCCL / aclrt 资源错误（OOM、stream invalid） | 立即 return，不切换 |
| `CC_ERR_INTERNAL` | backend 内部 bug（断言失败等） | 立即 return，记日志 |

严格约束：`execute` 失败时 `state` 必须保持 `prepare` 后的有效状态，即失败可重试；若破坏了 `state` 必须返回 `CC_ERR_INTERNAL` 并期望 caller 重建 backend。

生命周期（时序）：

```
prepare(cfg, &state)  →  execute() × N  →  destroy(state)
                              ↑                  ↑
                              │                  └─ 只在 dispatcher 销毁或换 backend 时触发
                              └─ 并发安全（不同 stream 并发调用允许）
```

选择策略（dispatcher 内）：

- 读 `CUSTOM_COMM_BACKEND` 环境变量，存在则硬选
- 否则按 registry 注册顺序 probe：调 backend 的 `probe(cfg)` 查可用性（可选字段，默认返回 true）
- 第一个 `probe` 通过的作为主；其余按顺序 fallback（当 `execute` 返回 `CC_ERR_UNSUPPORTED` 时切换）

Phase 2 退出标准：

- 现有 2 个 backend 通过 registry 调用，测试全绿
- 去掉 `HcclAllGatherBatchWithImpl` 这种内嵌分支函数，dispatch 纯表驱动
- `nm libcustom_comm_impl.so | grep BackendRegistry` 只暴露一个 factory 符号

### Phase 3: 解耦 torch_ext（2-3 天）

目标：`torch_ext/csrc/allgather_batch.cpp` 从 ~350 行降到 <80 行，业务逻辑下沉到 `ops/`。

动作：
1. 把 schema 宏（`TORCH_LIBRARY`, `TORCH_LIBRARY_IMPL`）集中到 `torch_ext/csrc/registration.cc`，一处注册所有 op
2. Tensor → DescArray 的转换搬到 `torch_ext/csrc/common/tensor_bridge.{h,cc}`，meta 和 NPU 两个后端复用
3. 让 `allgather_batch.cpp` 只剩：参数校验 + 调 C API + 结果转 Tensor。CCU / 选 backend 的逻辑一条都不剩
4. aten 2.5+ 的 `TORCH_LIBRARY_IMPL(..., Meta, m)` 单独放一个 `meta_impls.cc`

Phase 3 退出标准：
- `torch_ext/csrc/ops/allgather_batch.cpp` ≤ 100 行
- 添加任意新 op 时，torch_ext 侧修改面积 = 新 op 一个文件 + registry 一行

### Phase 4: 可选增强（暂缓，视后续需求）

- 共享 `.so`：对 cann_custom_comm.so 做动态加载（`dlopen` + `dlsym`）的 backend plugin 化。仅当有第三方后端需求时再做
- op 自动发现：基于宏 + static initialization 或 linker section (`__attribute__((section("custom_comm_ops")))`) 构建 op list。Phase 3 的手动 registry 足够用了，暂不做
- bench 框架化：对标 torchcomms 的 benchmarks/，把 `tests/test_allgather_batch.py` 的 bench 部分抽出来。只在需要跨 backend 对比时做

## 关键设计决策

### D1：Registry 用 C 结构，不用 `std::vector<std::function>`

跨 .so 传递带 vtable 的 C++ 对象有 ABI 风险。参考 torchcomms `TorchCommRegistry`，用 POD struct of function pointers。

### D2：保留 ABI firewall

不改 public header。`include/custom_comm/allgather_batch.h` 继续只暴露 C 函数 + opaque `CustomCommContext*`。Python 侧的 `_C.so` 通过 C ABI 调用 `libcustom_comm_impl.so`。参考 TransformerEngine `nvte_*` 接口风格。

### D3：Backend 选择靠显式配置 + fallback，不做自动推断

Phase 2 提供 `CustomCommConfig{ .backend_name = "ccu_mesh1d" }`；不存在时按顺序回退（ccu_mesh1d → decomposed → error）。避免魔法行为（昇腾 ABI 层面 HCCL 已经有隐式 topology 检测，再加一层会难调）。

### D4：不做的事

详见文档末尾"不做的事"小节。

### D5：backend vtable 带 `abi_version` 字段

vtable 是跨编译单元（shim 内部静态注册 + dispatcher 查表调用）的契约。字段增删会破坏二进制兼容。`abi_version` 写入后，dispatcher 可以在注册时校验版本一致，拒绝不兼容的 backend。Why：后续新增 `query_capability`、`prefetch` 等字段时，旧 backend `.o` 不需要重编也能在加载期被 dispatcher 识别并正确拒绝/降级，避免静默越界读写 vtable。

### D6：不引入 C++20 coroutine / `std::expected` / 跨 DSO coroutine

Why：
1. toolchain 现状是 gcc 10 + CANN 9.0 的 libstdc++，对 C++20 feature 支持不完整；引入就要绑定工具链版本
2. 本项目 hot path 是 "prepare kernel → submit to HCCL stream → return"，主机侧不做异步等待——没有真正的 coroutine suspend 点；`std::future` + stream sync 已经足够
3. coroutine frame 分配与 ABI boundary（shim.so ↔ torch_ext.so）交互有坑：promise_type 的 ODR、exception propagation 都跨不过 ABI 边界
4. 当前属于 YAGNI；满足以下任一信号时再重新评估：
   - 出现跨算子 host-side 编排（如 MoE all-to-all 的 dispatch/combine，或 AG+RS+matmul 重叠需要 host 侧多个 event 等待）
   - MoE dispatch / combine 这类多级派发的 op，代码里 state machine 已经复杂到 callback-hell
   - CANN + gcc 升级到 C++20 默认且 `std::coroutine` 在 ABI=0 环境验证过稳定性
   - bench 数据显示 host 侧 `HcclAllGatherBatch` 提交延迟成为 TTFT / step 瓶颈

（"本项目 hot path" 结论需要 profile 数据支持；如果 bench 显示 CPU 开销成为瓶颈，则这个决策应重新审视。）

## 验证方式

每个 Phase 完成后：

1. `pytest tests/` 必须全绿（现有 meta + NPU 用例）
2. `nm python/*.so | c++filt | grep custom_comm` 对比符号数：Phase 1-2 应保持不变，Phase 3 可能略有变化但核心 API 符号不变
3. `ldd python/*.so` 依赖无新增
4. `pytest tests/test_abi_firewall.py` 专项通过（这是之前 PR 引入的守护，见 `faq.md`）
5. Phase 3 后新加一个 dummy op（例如 `reduce_scatter_batch`，只实现 decomposed）走一遍流程，证明新框架 works

## 风险

- 符号冲突：`ops/_core/` 里的符号如果没有 namespace / visibility=hidden，会和消费者的符号冲突。缓解：`ops/_core/` 全部用 `-fvisibility=hidden`，只有 `extern "C"` 的 C API 用 `__attribute__((visibility("default")))`。
- CANN / HCCL 版本兼容：Phase 2 的 registry 可能遇到 C++ 静态初始化顺序问题（SIOF）。缓解：registry 用 Meyers singleton（函数内 static），不用 namespace-level global。
- CCU kernel 编译依赖：`ccu_kernel_ag_batch.cc` 当前直接 include CANN kernel 头，移动位置时需确认 include path 跟随更新。

## 不做的事

- 不改 `allgather_batch.h` 公共 C API 的函数签名或 struct 布局（FAQ 已明确 ABI 冻结）
- 不引入 dynamic dlopen / plugin 机制（YAGNI；等真有外部 backend 需求再做）
- 不抽象到"可替换 comm library"层级（HCCL 是唯一 target，过度抽象是负担）
- 不重写测试用例；只在 fixtures 里补"强制某 backend"的开关以便新分支测试
- 不引入 C++20 coroutines / `std::expected`：见 D6
- 不统一 Meta 与 NPU dispatch（Meta 只需 shape 推导，语义不同）
- 不提前加 observer / telemetry hook（YAGNI，等真有多 backend 再考虑）

## 开放问题

1. CCU backend 目前依赖 `HCCL_COMM_HCOMM_QOS_CONFIG_NOT_SET` 这类宏（来自 `common.h`）。重构后这些常量放哪个头？倾向放 `ops/_core/hccl_quirks.h`，仅 impl 可见。
2. Phase 2 开始时要先删 `src/ccu_v2/` 残留（已在 `343221c` commit 中处理一部分）还是保留作为下一代 backend 的 scaffold？倾向先删，新 backend 在 Phase 2 之后新起。
3. 是否要把 `python/` 目录提上日程用于放置 pytest helper 和 typing stub？目前 stub 散落在 `tests/`，下游消费者拿不到。

## 评审与后续

- 先对本计划征求 owner 评审（预期 revision 点：backend 抽象的 C-struct vs C++ virtual 选择、\_core\_ 目录命名、目录分层的颗粒度）
- 评审通过后，Phase 1 作为独立 PR 先落地；Phase 2/3 各自独立 PR
- 每个 PR 的描述需明确引用本计划中的对应章节，便于追溯
