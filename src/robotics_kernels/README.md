# Robotics GPU operators

本目录是 robotics 独立维护的算子源码，不在运行时导入工作区的 VLM 项目。
默认 CPU 安装不加载 Torch、CUDA、Triton 或编译扩展。

源码按架构组织，架构目录内部再放对应的 CUDA/C++ 源码：

```text
robotics_kernels/
  ampere_ada/     SM80兼容的整数与BF16卷积包装、准备和调优配置
    csrc/        INT4/INT8 GEMM与BF16 CUTLASS卷积实现
    third_party/cutlass/ 固定版本的构建头文件与许可证
  blackwell/     独立FP4/FP8及其CUDA源码
  common/        跨架构Triton融合、CUDA Graph和浮点投影工具
  integer.py     兼容导入入口；实现位于ampere_ada/
  fused.py       兼容导入入口；实现位于common/
  graph.py       兼容导入入口；实现位于common/
  projections.py 兼容导入入口；实现位于common/
```

Ampere和Ada共用同一个整数内核实现，显式tactic负责形状调优，不复制相同CUDA文件。
当前性能/数值实测在SM89上；Ampere与Blackwell分别需要目标设备验证。

| 入口 | 格式与职责 | 验证边界 |
| --- | --- | --- |
| `fused.py` | RoPE、gated residual、GELU×up、norm affine、AdaLN modulation；保留低精度中间舍入 | RTX 6000 Ada；GPU 数值与当前 stream 测试 |
| `ampere_ada/integer.py` / `ampere_ada/csrc/integer_gemm.cu` | signed INT8/W8A8、signed INT4/W4A4；S32 Tensor Core 累加，融合 scale/bias，BF16 输出 | SM89 实机验证；Ampere 源码路径仍需目标设备验证 |
| `projections.py` | 加载时 QKV/Gate-Up 权重合并与输出切片 | BF16 合并可能改变 GEMM 舍入；不列入已验证无损配置 |
| `graph.py` | 输入刷新、输出复制、目标设备 capture stream、失败恢复和常量所有权 | 每个实例串行调用；权重在上下文内不可变 |
| `blackwell/` | 保留 FP4、tensor FP8、MXFP8、相关 mixed-input 及 CuTe 实现 | 独立来源记录；Blackwell 编译/模型性能尚待实机验证 |
| `ampere_ada/convolution.py` | 仓内CUTLASS 2D/3D卷积，8个显式tactic，权重预打包、布局处理与bias epilogue | Ada实测；近似数值，不是无损替换；不支持的模块保持native并记录 |
| `common/vae.py` | 保留原生通道归约，融合后续除法、scale、affine和可选SiLU | BF16逐值GPU检查及LIBERO/RoboCasa固定输入检查；不代表全任务集质量 |

INT 的公开包装遵循 `from_linear → pack_input → forward_packed/forward` 生命周期。
`forward_gelu` 将 GELU 或 gated GELU 与 activation pack 融合。激活按行、权重按输出
通道使用动态/静态 float32 scale；INT4 字节的低 nibble 对应前一个元素。
逻辑 K/N 与物理 padding 分开：K 补齐 128、N 补齐 8，补齐区域为数学零，输出再裁剪。
这些是当前 kernel 的显式能力，不是模型 hidden-size 硬编码。

八个明确 tactic 的M/N/K、warp和流水级数位于 `ampere_ada/tactics.py`；0/1保留
原始配置，2..7扩展窄N、大N、大K和流水深度候选。按真实 shape、设备和完整 Linear
开销比较，不从旧 LLM 阈值继承最优值，也不将一张卡的配置自动套到另一张卡。
`forward` 的计时须包含 activation prepare、pack 和必要的布局转换；离线权重打包单列。

`integer_qkv` 与 `integer_gate_up` 分别合并同输入、同格式的投影；`integer_grouped`
兼容同时开启两者。权重/scale在初始化时拼接，运行时共享GEMM结果。
`integer_group_views` 允许消费者直接读取输出切片，省去显式的连续化拷贝；对应激活
准备支持末维连续、行间有stride的输入。任意不规则布局会明确拒绝，不默默复制。
`integer_pack_reuse` 让INT4量化与nibble打包复用同一行数据。所有开关独立保留，
模型级性能不能由合并后的GEMM微测结果推断。

`ampere_ada/modulation.py` 提供Cosmos的modulation与激活量化融合，保留原BF16
中间舍入；内部packed carrier同时支持按需提供INT4/INT8，避免混合档位误用格式。
模型外接口仍交付CPU动作，不向模拟器暴露packed数据。
`norm_modulation_quant` 将LayerNorm也并入打包，`residual_norm_modulation_quant`
进一步并入前置门控残差；新的FP32归约顺序可改变动作，二者是独立的数值实验开关，
不属于保留原生归约的配方。它们与原调制打包共用当前文件中的格式和舍入边界。
`integer_biasless` 为无bias层提供独立尾部处理实验路径；有bias时继续正常计算bias。

目前量化迭代优先Cosmos，π0.5先限于text/LLM。动作专家/diffusion量化暂缓；
已有可选scope不表示通过了相应任务质量验收。

Cosmos本轮固定/渐进整数实现和共享VAE优化已完成阶段验证；启动配方、测量口径、
默认关闭的实验选项和回退方式见 [Cosmos量化指南](../../docs/cosmos-quantization.md)。

卷积的CUTLASS迭代器、分块、流水线与数值边界见
[CONVOLUTION.md](ampere_ada/CONVOLUTION.md)。单算子调优收益必须回到完整policy复核；
当前卷积与attention替换均保留独立实验开关，不自动替换默认后端。

INT后端使用架构目录内的CUTLASS头文件副本，固定revision为
`982748aa7356fa838c2ea4994ddcb0b2a4b4cefa`，来源及逐文件哈希见
`ampere_ada/third_party/cutlass/PROVENANCE.json`。构建不读取
`ROBOTICS_CUTLASS_ROOT`，也不查找外部VLM、QuaRot或Mini QServe项目。
本机仍需提供CUDA开发工具链：

```bash
export CUDA_HOME=/path/to/cuda
export TORCH_CUDA_ARCH_LIST=8.9
```

扩展按Torch的CUDA extension缓存规则生成编译产物；源码与头文件均来自本仓。
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
