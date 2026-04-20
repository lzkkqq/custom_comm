# FAQ

## ABI 防火墙：为什么 `custom_comm` 拆成两个 `.so`?

### 1. 症状

`TestCcuPath::test_ccu_only` 稳定 SIGSEGV 或 `free(): invalid pointer`，堆栈落在
`HcclCcuKernelRegisterFinish` 之后 `std::string` / `std::ostringstream` 的析构或赋值里。
而 `TestCcuPath::test_decomposed` 路径永远正常 —— 因为 decomposed 只碰 HCCL 的 C API，不吃 C++ 对象。

### 2. 根因：CANN 与 torch_npu 用了两套 libstdc++ ABI

两条关键证据，来自 `nm -D` 直接在编译产物上看符号：

| 对象 | `__cxx11` 符号 | 含义 |
|---|---|---|
| `libhcomm.so`（来自 CANN 9.0 SDK） | 0 | 被编译时显式 `-D_GLIBCXX_USE_CXX11_ABI=0`，使用 pre-C++11 ABI |
| `libtorch_cpu.so` / PyTorch wheel | 大量 `__cxx11::` | 默认 CXX11 ABI（=1）|

用户的 PyTorch 扩展（`custom_comm._C.so`）必须匹配 torch 的 ABI=1，但它又要调用 `libhcomm` 的 C++ 接口（`hcomm::CcuKernelSignature::Append(std::string)` 这类）。`std::basic_string` 在 ABI=0 和 ABI=1 下是两套完全不同的类型（SSO buffer、shared pointer、引用计数字段都不一样）：一边把对象里嵌入的 `std::string` 当成旧 ABI 布局构造，另一边当成新 ABI 布局析构，堆损坏 → SIGSEGV。

关键证据：崩溃总是发生在 `HcclCcuKernelRegisterFinish` 回来之后的析构，而不是 CCU 内部算法里；不管 CCU kernel 做什么，只要构造/析构路径经过一个 mismatched std::string 就触发。

### 3. 修复：split 成两个 .so，用 extern "C" 做 ABI 边界

```
┌──────────────── Python process ────────────────┐
│                                                │
│  _C.cpython-*.so          libcustom_comm_impl.so│
│  ───────────              ──────────            │
│  ABI=1 (与 torch 同)      ABI=0 (与 libhcomm 同)│
│  只包含 torch_ext/csrc/   只包含 ops/*/src/**/ │
│                                                │
│  包含 torch headers       包含 hcomm C++ 类     │
│  (std::string, at::Tensor)(std::string, CcuXxx)│
│                                                │
│       │ extern "C" + 基本类型 / 指针 │          │
│       └───────────────►──────────────┘         │
│         HcclAllGatherBatch 等 C API            │
└────────────────────────────────────────────────┘
```

`_C.so` 通过 `RPATH=$ORIGIN` 在同一目录 dlopen `libcustom_comm_impl.so`，两个 `.so` 之间只有 `extern "C"` 函数调用，任何 `std::` 类型都不跨 `.so` 边界。

### 3.1 怎么让 `libcustom_comm_impl.so` 真的是 ABI=0

setup.py 里 `build_shim()` 直接用 `g++ -c ... -D_GLIBCXX_USE_CXX11_ABI=0 ...` 并行编 `.o`，然后手动链接。不走 `NpuExtension`，因为后者会继承 torch 的 CXX11 ABI。

### 3.2 构建期断言

`_verify_shim_abi()` 在编译完立刻跑 `nm -D libcustom_comm_impl.so` 确认：

- 没有任何 `__cxx11` 符号（证明 ABI=0 生效）
- `HcclAllGatherBatch` 作为 `T` 符号导出（证明 extern-C 层正确）

如果哪一条不满足 `pip install` 直接失败，防止默默编译出一个会 segfault 的 shim。

### 4. `tests/test_abi_firewall.py` 的 5 个用例

每个测试都是 op-agnostic（不依赖具体算子名）：

| 测试 | 检查 |
|------|------|
| `test_shim_is_abi_zero` | shim 里 `__cxx11` 符号数 = 0 |
| `test_shim_exports_at_least_one_c_api` | shim 里有至少一个非 mangled 的 `T` 符号 |
| `test_binding_is_cxx11_abi` | `_C.so` 里有 `__cxx11` 符号（ABI=1 生效） |
| `test_binding_links_to_shim` | `_C.so` 的 `NEEDED` 包含 `libcustom_comm_impl.so`，`RPATH` 含 `$ORIGIN` |
| `test_binding_imports_from_shim` | binding 的 `U`（未解析）符号和 shim 的 `T`（已定义）符号有非空交集 |

### 5. 延伸问题

- **为什么不用一个 .so 搞定？** torch 2.x 的 Python wheel 用 ABI=1，CANN 9.0 的 libhcomm 用 ABI=0。如果只编一个 .so，要么选 ABI=0 → 无法链接 torch；要么选 ABI=1 → 无法安全调用 libhcomm 内部 C++ 符号。
- **为什么需要 `-Wl,-rpath,'$ORIGIN'`？** `_C.so` 在 Python 包目录里，`libcustom_comm_impl.so` 是它的平级邻居。`$ORIGIN` 让动态链接器在加载 `_C.so` 时到同目录里找 shim，不需要调用方设置 `LD_LIBRARY_PATH`。
- **HCCL 宏别名 `HCCL_COMM_HCCL_QOS_CONFIG_NOT_SET` 为啥还留着？** 那是 CANN 9.0 改了一个常量名，torch_npu 2.9 的 wheel bundled 了旧的 hccl header 还用旧名。属于 header 名字层面的 mismatch，不是 ABI 层面的，放在 setup.py 里 `-D` alias 解决。和本节主线无关。

### 6. 长期

- torch_npu 升到和 CANN 完全同步的版本后，可以去掉 `-D HCCL_COMM_HCCL_QOS_CONFIG_NOT_SET=...`
- 如果 torch 官方切到 ABI=0（不会），整个 shim 拆分就可以合回去
- 如果 CANN 把 `libhcomm.so` 切到 ABI=1（可能，某天），shim 的 `-D_GLIBCXX_USE_CXX11_ABI=0` 可以去掉，合回一个 .so
