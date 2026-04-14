# 测试矩阵

## 蓝区已通过 (32/44)

| 测试 | 文件 | 数量 |
|------|------|------|
| Meta shape inference | test_allgather_batch.py::TestMetaKernel | 23 |
| torchair converter 注册 | test_graph_mode.py::TestConverterRegistration | 2 |
| GE graph shape inference | test_graph_mode.py::TestMetaShapeInference | 7 |

## 等待 NPU (12/44)

| 测试 | 文件 | 说明 |
|------|------|------|
| 单 desc 正确性 (4 dtype) | TestNpuFunctional::test_single_desc | Phase 1 |
| INT8+FP32 聚合 | TestNpuFunctional::test_heterogeneous | OPT-AG-09 |
| 三合一 (int8+fp32+int32) | TestNpuFunctional::test_three_tensor_pack | OPT-AG-04/09 |
| 100 次重复稳定性 | TestNpuFunctional::test_repeated_calls | Phase 1 |
| CCU vs Decomposed 一致性 | TestCcuPath::test_ccu_matches | Phase 2 |
| GE graph 编译 | TestGraphModeE2E::test_ge_graph_compile | torchair |
| GE graph 正确性 | TestGraphModeE2E::test_ge_graph_correctness | torchair |
| aclGraph capture + replay | TestAclGraphCapture::test_capture_replay | 1 |
| aclGraph 重复 replay | TestAclGraphCapture::test_repeated_replay | 1 |

## Benchmark（需 NPU）

```bash
# 通用矩阵
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py

# OPT-AG-09 场景
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py --ag09

# Phase 2 CCU
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
```
