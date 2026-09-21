# KINETIX参考代码与接入边界

本次审查覆盖调用者提供的 `dynamic-experiments` 源码、RTC/Kinetix工作副本，以及后期native-time实验协议记录。参考仓库顶层README偏早，不能用它替代实际脚本和实验记录。以下外部文件名用于解释来源；本仓运行已经使用迁入的源码，不依赖该参考目录。

## 各部分解决的问题

| 参考部分 | 机制与贡献 | 本次处理 |
| --- | --- | --- |
| `rtc_kinetix.py`、`upstream_runners.py` | 关卡/参数绑定、延迟映射、flow步数和horizon扫描；将实验配置与上游实现分开 | 接入固定关卡、成对随机种子、独立质量/时间参数和可审计输出 |
| RTC `src/model.py` | Flax flow policy；MLP-Mixer生成8步motor/thruster控制，Euler采样步数改变网络调用次数 | 迁入 `flow_model.py`，当前直接使用原checkpoint，不训练或蒸馏 |
| `run_sensitivity_eval.py` | 控制步取整与细粒度物理子步延迟，区分噪声采样节奏并保留物理horizon | 仅保留原生步长下的取整对照；不把物理细分作为默认延迟实现 |
| `run_torch_task_accuracy_eval.py` | 将等价Torch策略置于JAX环境中，支持direct/旧RTC队列、原生eager或CUDA Graph | 整理接口与对照规则；首版实际执行保持JAX/Flax |
| `rtc_torch_policy.py`及低精度runtime | conditioning复用、融合MLP、CUDA Graph、W4/W2等独立执行路径 | 属于后续模型/算子后端；不随本次协议接入宣称已有量化收益 |
| `rtc_student*.py` | 结构化小模型与训练/蒸馏路径 | 当前不迁入训练；结构变化需要独立model配置与checkpoint验收 |
| native-delay协议说明与后期 `protocol.json` | 固定原生物理网格，按执行器指令进行分数槽混合；区分硬件profile与模拟时间 | 经用户确认，在本仓重建native-blend，保留近似边界说明 |
| 后期quality–latency拟合报告 | 分开拟合零延迟质量、固定质量下的延迟响应、硬件完整action-generation时延 | 本次产物保留这些分析需要的独立实验轴；暂不迁入历史拟合结果或自动寻找最优点 |

后期native-blend的协议和结果存在，但在本次检查的本地可执行脚本中没有找到相应实现，因此本仓实现以已记录协议为依据，不能宣称是原代码的逐行移植或完整历史结果复现。

## 两类时间不能混淆

flow采样变量的积分步长为1/N；物理时间步为dt，控制周期为dt×frame_skip。减少N可以改变动作质量和完整推理工作量，但不应改变环境动力学。

早期细分协议用更小dt和更大frame_skip保留控制周期。例如从 `1/60×2` 改到 `1/2400×80`，名义控制周期仍为1/30秒，但每模拟秒的接触/关节求解次数提高40倍。碰撞Baumgarte参数的单项缩放不能自动保持warm-start、位置修正、关节限制、摩擦与终止采样等语义。

参考记录中MuJoCo衍生的walker/half-cheetah零延迟质量对这类变化敏感。因此“零注入延迟”不等于回到了原始任务，也不能据此把质量变化归因给轻量化。这里的行为异常首先应视为数值/协议不可比，需要定位；它不自动意味着语言层面的undefined behavior。

本仓native-blend保留原生网格：完整旧动作槽、一个分数混合槽、后续新动作槽。混合发生在process后的执行器域，原生裁剪和自动电机语义得到保留。它是时间平均近似，不重新求解物理槽内部的接触切换。

## latency–quality实验的拆分

建议保持相同checkpoint、物理配置、噪声、执行horizon及seed列表，分别运行：

1. 固定延迟为0，改变flow步数，得到本协议下的内在质量代理。任务仍然动态，不能称为冻结世界的精度实验。
2. 固定flow步数，改变注入延迟，得到任务对延迟的响应。
3. 使用目标硬件完整action-generation测量，为每个flow步数映射延迟，观察组合结果。

参考拟合使用 `A(t,H,N)≈G_t(N)×H_t(D_H(N))`，分别表示质量、时延响应和硬件时延曲线。这是可检验的可分离模型，不是普适定律；phase-sensitive任务和弱拟合不能通过挑选任务或horizon隐藏。外部profile与当前执行后端不一致时，只能解释为声明的模拟场景。

质量/精度方案应使用相同调度规则与计时范围。fake/dequantized参数误差实验、真实低精度GEMM、CUDA Graph和flow步数减少分别标注，不能把单算子时间或虚拟注入值当作端到端实测加速。

## 源码归属

实际模型、环境、物理核心、关卡与渲染素材迁入 `robotics_bench.kinetix`，内部使用私有包名。JAX、Flax、通用环境接口和渲染库仍是正常Python依赖。原始checkpoint是独立资源。

迁入代码保留MIT版权与来源文件hash，仅包括运行需要的依赖闭包，移除被覆盖的重复类和不参与运行的云端/训练保存函数。默认执行不读取其他仓库的源码，移植后的物理计算仍受原生端点等价验证约束。

当前CLI与输出说明见 [KINETIX case](../benchmarks/dynamic/kinetix/README.md)。进一步研究小时间片的合理性在独立分支开展，不能通过改变主线case的默认物理参数来制造质量收益。
