---
name: edge-inference-optimization
description: Use when optimizing edge policy, VLM, or streaming inference, especially when quantized GEMM speedups are lost to activation preparation, fusion gaps, memory copies, cache handling, or runtime overhead.
---

# Edge Inference Optimization

用户要求的模型、设备与任务范围优先；本 skill 不要求先复现论文或扩展到其他模型。把目标设备上的完整推理及任务指标作为目标。GEMM 占比是诊断量，不以提高占比替代降低延迟。

## 建立一个可比较的案例

记录模型/checkpoint、请求形状、精度、设备/功耗/频率/软件栈、当前 backend 和完整调用入口。
固定真实 baseline commit、输入或 seed、warmup、计时范围和任务可接受门槛。
区分 host、GPU、sim 时间；记录 decode/prefill 或 vision/prefix/flow-denoise/action-head 的阶段。
优先采用已有自动计时与 profile 工具。单个调用没同步，不代表 GPU 工作已结束。
对共享设备先检查占用；性能数据要求隔离，或明确标注受干扰且不用于性能结论。

## 找到真正执行的热点

同时保留 kernel、完整 inference 和任务层测量。
识别 GEMM/GEMV、attention、norm、activation quant、layout/copy、cache、allocation 和 host dispatch。
存在重叠时按关键路径解释耗时；不要把多条 stream 的所有 duration 直接求和当作总时延。
对最大热点写出“证据 → 修改 → 预期收益 → 验证与停止条件”，先处理已有证据的高收益路径。

## 按执行路径选择优化

量化：区分 fake quant、W4A16、W8A8 等实际后端；记录 scale/group/zero-point/累加 dtype、校准和 pack 格式。
跟踪真实 kernel 与 fallback；有意 fallback 要记录原因和占比。
加载时完成可复用 pack/concat，避免每轮重建权重；tile 内 unpack 与完整权重反量化分别识别。

外围路径：检查 QKV/gate-up 是否共享 activation prepare；norm+quant、bias/residual、layout/copy 是否值得融合。
先验证编译器或现有后端能否解决，再决定手写 kernel；fusion 需检查 register pressure 和端到端收益。
workspace、输入输出 buffer、KV 生命周期与 shape 固定性允许时再使用复用或图捕获。
多个请求并发时不共享可写 buffer；图捕获失败或不支持 shape 有明确 fallback。

按 workload 分流：
- 自回归：prefill 与 q_len=1 decode 分别测，检查 GEMV、KV append 和 cache reset。
- diffusion/flow policy：vision/prefix 是否可复用、denoise step 的 mask/position/cache 是否等价；减少 step 属于有损方法。
- Jetson/端侧：核对架构支持、统一内存、CPU 线程、功耗和热状态；不从桌面 GPU 外推最优 kernel。

## 保留两种质量证据

算子级：参考实现误差、非对齐 shape、dtype/layout、NaN/Inf 和累加精度。
模型级：固定 observation/noise/seed 比较 action 向量、必要的 hidden state 和多步误差。
闭环级：相同 episode seeds 的 SR、任务时间和失败预算；成功条件时间单列。
LLM 首 token/exact match 只用于适用的生成任务，不能替代连续动作模型的质量验证。
数值等价融合与量化、剪枝、减少 step 等有损变化分开记录，便于定位收益与误差来源。

## 完成和迭代

在同设备和可比环境验证完整入口；microbenchmark 快而端到端不快时不能宣称部署加速。
达到任务门槛且测量收益高于噪声才保留优化；否则回退局部变更或作为实验路径隔离。
输出变更、实际 backend、前后数据、精度/任务证据、复现命令、适用条件与未验证范围。
每次真实案例只沉淀有证据的新规则；单模型技巧留在案例记录，不扩张成所有模型必须遵循的要求。
