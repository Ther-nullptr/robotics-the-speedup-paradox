# 模型与模拟器接入协议

本协议定义首版接入的责任与交付物，不要求先跑完整论文实验。所有模型当前为 planned；路径是后续任务建议。

## 所有 adapter 共享的要求

每次推理的输入是带 observation ID、episode ID、采样时间域的不可变快照；输出是带 request/observation 来源、动作单位、执行间隔和有效区间的 action chunk。模型可以内部同步实现，真实异步完成句柄后续再接，但不能把 GPU enqueue 返回当作动作已可用。

adapter 提供 reset 行为，声明历史状态、cache、随机流和输出 buffer 生命周期。reset 的完成确认、跨任务 cache scope 和加载模式遵循 [后端状态协议](backend-state.md)。输出 action horizon、实际执行动作数、denoise/refinement steps、cache chunk 数使用不同字段。配置序列化为 resolved manifest；公开的选项必须真正进入调用路径。

静态实验默认覆盖模型计时与静态场景闭环指标。先用固定观测 fixture 做 action 对齐即可提交基础接入，完整 rollout 与优化实验作为后续 PR。没有 GPU 的成员可以完成配置、shape、错误路径与 trace 契约工作。

## 首版方向

| 方向 | policy adapter 应保留 | protocol / simulator 负责 | 第一份验收证据 |
| --- | --- | --- | --- |
| Cosmos-Policy 静态 | vision/DiT/action 输出、层名、候选量化站点、dtype/pack manifest | 静态执行周期、动作块、成功与失败统计 | 固定 observation/noise 下 reference action 对齐；实际 backend 映射 |
| LingBot-VA 静态 | visual/action KV 写入及读取窗口、跨轮状态、action 解码 | 环境重置、chunk 执行、时间线 | reset 不泄漏旧 KV；可读窗口与存储/写入量分开报告 |
| π0.5 静态 | vision/language prefix、flow action expert、预处理/normalization | 同步基线、历史观测模式、真实 overlap 分离 | 固定输入/噪声动作对齐；无内部 env.step |
| 动态仿真 | 模型侧只消费快照并返回动作 | 世界在等待中推进、延迟采样、旧动作/hold 规则、terminal | CPU scripted policy 验证运动和延迟；之后接 Kinetix/DOM 等真实环境 |

Cosmos-Policy 和 LingBot-VA 的机器人适配器属于本仓库；后续其他研究项目的实现不因模型类别相同而自动纳入公开范围。

## 量化与融合接口

量化配置独立于 policy 与 kernel。每层记录模块路径、weight/activation/accumulator dtype、scale/group/zero-point 规则、calibration/pack 版本、实际 backend、fallback 原因。fake quant 只用于质量/方法分析，不作为低比特性能证明。

保留 reference、仅替换量化 GEMM、加入外围优化三个可区分的运行点。测试分层：算子误差；固定输入多步 action 误差；闭环 SR/任务时间与失败代价。尚未做的层级写“未验证”，不要将 LLM 的首 token 标准用于连续动作。

π0.5 的开源后端可以作为可选 backend 候选，但不得强迫 Cosmos/LingBot 采用同一张图、同一 KV 格式或同一个量化 kernel。后端抽象共同处理 capability、输入输出和计时，模型内部细节保留在自己的模块。

## 动态仿真接口

`observe()` 返回真实采样时刻的快照；`step(action)` 只推进已声明的 control/physics tick；任务显式给出等待期间 robot control 和世界动力学规则。`render()` 是观测读操作，不能改变状态或阻塞控制。

delay profile 是参数化外部输入，至少标识硬件、模型/配置、计时 scope、原始样本与采样方式。`zero`、虚拟重放、真实计时、additive、target-total 分开；对 target-total 记录 overshoot，对虚拟重放避免叠加宿主计算耗时。完整定义见 [仿真协议](simulation.md)。

## 可视化消费者接口

第一版读取已保存 trace，展示环境帧引用、inference、queue、action、terminal；点击动作能追踪到其来源观测。先用合成样例即可验证交互与状态映射，不需要先生成论文视频。

实时 viewer 之后再增加；消费者慢时丢显示帧并计数，关键事件单独持久化。正式 benchmark 单列 tracing/录像开销。截图可以辅助 review，不能替代时间线和统计数据。
