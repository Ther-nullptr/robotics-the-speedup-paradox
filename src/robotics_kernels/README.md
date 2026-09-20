# Robotics GPU operators

本目录是 robotics 独立维护的算子源码，不在运行时导入工作区的 VLM 项目。
默认 CPU 安装不加载 Torch、CUDA、Triton 或编译扩展。

| 入口 | 格式与职责 | 验证边界 |
| --- | --- | --- |
| `fused.py` | RoPE、gated residual、GELU×up、norm affine、AdaLN modulation；保留低精度中间舍入 | RTX 6000 Ada；GPU 数值与当前 stream 测试 |
| `integer.py` / `csrc/integer_gemm.cu` | signed INT8/W8A8、signed INT4/W4A4；S32 Tensor Core 累加，融合 scale/bias，BF16 输出 | SM89 实机验证；Ampere 源码路径仍需目标设备验证 |
| `projections.py` | 加载时 QKV/Gate-Up 权重合并与输出切片 | BF16 合并可能改变 GEMM 舍入；不列入已验证无损配置 |
| `graph.py` | 输入刷新、输出复制、目标设备 capture stream、失败恢复和常量所有权 | 每个实例串行调用；权重在上下文内不可变 |
| `blackwell/` | 保留 FP4、tensor FP8、MXFP8、相关 mixed-input 及 CuTe 实现 | 独立来源记录；Blackwell 编译/模型性能尚待实机验证 |

INT 的公开包装遵循 `from_linear → pack_input → forward_packed/forward` 生命周期。
`forward_gelu` 将 GELU 或 gated GELU 与 activation pack 融合。激活按行、权重按输出
通道使用动态/静态 float32 scale；INT4 字节的低 nibble 对应前一个元素。
逻辑 K/N 与物理 padding 分开：K 补齐 128、N 补齐 8，补齐区域为数学零，输出再裁剪。
这些是当前 kernel 的显式能力，不是模型 hidden-size 硬编码。

两个明确 tactic：0 使用 CTA M=64，1 使用 CTA M=128；N=128，INT4/INT8 的 CTA K
分别为128/64。按真实 shape、设备和完整 Linear 开销比较，不从旧 LLM 阈值继承最优值。
`forward` 的计时须包含 activation prepare、pack 和必要的布局转换；离线权重打包单列。

构建 INT 后端时显式提供兼容 CUTLASS checkout 和 CUDA toolkit：

```bash
export ROBOTICS_CUTLASS_ROOT=/path/to/cutlass
export CUDA_HOME=/path/to/cuda
export TORCH_CUDA_ARCH_LIST=8.9
```

扩展按 Torch 的 CUDA extension 缓存规则构建；不读取某个固定实验室目录。
临时 workspace 由 Torch 管理，CUDA 调用使用输入设备的当前 stream。

Blackwell 构建入口：

```bash
python tools/build_blackwell.py --backend fp8 \
  --cutlass-root /path/to/compatible-cutlass --arch 11.0a
```

`--arch` 必须与目标 GPU、CUDA 和 CUTLASS 版本匹配。FP4/NVFP4/MXFP4、tensor FP8/MXFP8
与 INT4/INT8 的编码和 scale 布局不同，不能共用 packed buffer。
FP 后端的来源与修改边界见 `blackwell/PROVENANCE.json` 和许可证。

GPU 测试在模型环境中显式启用，公共 CPU CI 默认跳过：

```bash
ROBOTICS_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/gpu
```

模型接入位于 `robotics_bench/optimizations/`；完整调用计时和可视化入口见
[推理实验说明](../../benchmarks/inference/README.md)。算子数值通过不等于闭环质量通过，
量化输出偏差和实际任务结果分别记录。
