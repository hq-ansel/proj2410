# 3-bit Dequantization Kernel 优化报告

## 概述

为 TritonV2QuantLinear 添加了 3-bit 量化支持，包括专用的 Triton dequant kernel。

## 3-bit 打包方案

3-bit 量化将 32 个值打包到 3 个 int32 中：
- int32[0]: values[0:10] at shifts [0,3,6,...,27], value[10] 低 2 bits at shift 30
- int32[1]: value[10] 高 1 bit at shift 0, values[11:21] at shifts [1,4,7,...,28], value[21] 低 1 bit at shift 31
- int32[2]: value[21] 高 2 bits at shift 0, values[22:32] at shifts [2,5,8,...,29]

## 实现的 Kernel 版本

### v1 (dequant_kernel_3bit)
- 使用 5 个独立的 `tl.where` 分支处理不同区域
- 配置: X_BLOCK=256, num_warps=2, num_stages=1
- 特点: 代码清晰，在小矩阵上性能好

### v2 (dequant_kernel_3bit_v2)
- 使用嵌套的 `tl.where` 减少分支发散
- 配置: X_BLOCK=[256,512,1024], num_warps=[2,4], num_stages=[1,2] (autotune)
- 特点: 在大矩阵上略有优势

## 性能测试结果 (RTX 4090)

### Dequant Kernel 性能

| Config | 4-bit (ms) | 3-bit v1 (ms) | 3-bit v2 (ms) | v1/4bit | v2/4bit |
|--------|------------|---------------|---------------|---------|---------|
| 4096x4096 g128 | 0.028 | 0.049 | 0.048 | 1.76x | 1.72x |
| 4096x11008 g128 | 0.124 | 0.143 | 0.143 | 1.16x | 1.16x |
| 11008x4096 g128 | 0.124 | 0.141 | 0.141 | 1.14x | 1.14x |
| 896x896 g64 | 0.027 | 0.027 | 0.033 | 0.97x | 1.22x |
| 896x4864 g64 | 0.027 | 0.027 | 0.033 | 0.99x | 1.22x |

### Forward Pass 性能 (batch=1, seq=2048)

| Config | 4-bit (ms) | 3-bit (ms) | Ratio |
|--------|------------|------------|-------|
| 4096x4096 g128 | 0.514 | 0.524 | 1.02x |
| 4096x11008 g128 | 1.422 | 1.487 | 1.05x |
| 11008x4096 g128 | 1.282 | 1.324 | 1.03x |
| 896x896 g64 | 0.105 | 0.074 | 0.71x |
| 896x4864 g64 | 0.152 | 0.158 | 1.03x |

## 性能分析

### 3-bit 比 4-bit 慢的原因

1. **更复杂的位操作**: 3-bit 不是 2 的幂次，导致打包/解包逻辑复杂
2. **跨边界值**: 位置 10 和 21 的值跨越两个 int32，需要额外的位操作
3. **更多内存加载**: 每次需要加载 3 个 int32（vs 4-bit 只需 1 个）
4. **分支发散**: 5 个不同区域的处理逻辑

### 为什么 Forward Pass 差距小

- Matmul 操作主导整体时间
- Dequant 只占 forward pass 的一小部分
- 实际推理场景中 3-bit 只比 4-bit 慢 2-5%

### 小矩阵上 3-bit 更快的原因

- Kernel launch 开销主导
- 3-bit 的 X_BLOCK=256 vs 4-bit 的 X_BLOCK=512
- 更小的 block size 在小矩阵上更高效

## 优化建议

### 已实现的优化

1. 专用的 3-bit kernel 而不是通用 kernel
2. 使用 `eviction_policy="evict_last"` 优化缓存
3. 提供 v1/v2 两个版本供不同场景选择

### 潜在的进一步优化

1. **向量化加载**: 使用 `tl.load` 的向量化版本一次加载多个 int32
2. **共享内存**: 将 qweight 加载到共享内存减少重复加载
3. **Warp-level 优化**: 利用 warp shuffle 在 warp 内共享数据
4. **融合 kernel**: 将 dequant 和 matmul 融合减少内存带宽

## 文件修改

- `EfficientQAT/core/linear/q_linear_triton_kernels.py`: 添加 `dequant_kernel_3bit` 和 `dequant_kernel_3bit_v2`
- `EfficientQAT/core/linear/q_linear_tritonv2.py`: SUPPORTS_BITS 添加 3
- `EfficientQAT/core/linear/q_linear_pack.py`: 移除 3-bit TP 限制
- `VeOmni/tasks/quantize/export_tritonv2_quant.py`: 更新帮助文本

## 测试

- `test/test_3bit_quant.py`: 功能测试（pack/unpack, forward, symmetric）
- `test/benchmark_3bit.py`: 性能测试
- `test/ncu_profile_3bit.py`: NCU profiling 脚本

## 结论

3-bit 量化支持已完整实现，在实际推理场景中性能损失仅 2-5%，是可接受的。对于需要更激进压缩的场景，3-bit 提供了比 4-bit 更好的压缩率（节省 25% 存储）同时保持合理的性能。
