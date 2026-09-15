# 动态任务实验入口

本目录组织动态任务的case、运行入口和时间协议配置。当前已建立开发入口，时钟驱动、模型与模拟器闭环尚未实现，没有可执行的benchmark命令。

| 规划case目录 | 绑定范围 | 首次接入要明确的内容 |
| --- | --- | --- |
| `dynamicvla_dom/` | DynamicVLA＋DOM动态任务 | 任务匹配权重/processor、Isaac环境与资产、历史观测、原生时延及动作交接语义 |
| [kinetix/](kinetix/README.md) | Kinetix原生JAX策略＋二维任务环境 | 已明确独立policy/env/rollout边界；具体checkpoint/关卡待绑定，观测/动作空间、carry、PRNG与时间协议单独验证 |

这些case均为planned。DOM接入不自动开放静态任务的π0.5/Cosmos权重；新增模型须建立任务匹配且独立验证的case。具体任务/初态、权重、预后处理、环境版本和baseline在运行配置中固定。

## 文件归属

- 后续每个case的启动代码与配置放 `benchmarks/dynamic/<case>/run.py`、`case.yaml`，在实际接入时一起创建并验证，不提前放占位文件。
- DOM等VLA动态任务的ClockDriver、观测历史与动作生效协议放 `src/robotics_bench/protocols/dynamic/`；模型与环境适配分别放 `src/robotics_bench/policies/` 和 `simulators/`。Kinetix先在独立子目录保留原生policy/env/rollout组合，具体归属见其入口；这些源码仍待实际实现。
- case配置分别声明时间/延迟模式、等待期间世界推进、目标保持、动作有效期和任务预算；运行产物放已忽略的 `runs/dynamic/<case>/`。

本路径可比较同步和异步。同步推理不自动冻结世界；对照固定世界推进和时间域。DOM原生补偿、虚拟profile重放和真实时间测量分别命名，禁止重复计入延迟；按块交接与按时间替换也分别验证。详见 [仿真协议](../../docs/protocols/simulation.md) 和 [多后端协议](../../docs/protocols/simulator-backends.md)。

## 接入与验收

先用CPU fake clock/policy验证事件时间线，再用单场景脚本策略验证延迟期间物体运动、目标保持、动作过期和reset；最后接任务匹配模型并与相同case的baseline比较，无需先做完整论文复现。

报告成功率或任务定义的score、任务时间、观测年龄、动作生效/丢弃、欠载和终止原因；真实时间实验另报deadline miss与real-time factor。模拟时间、host时间和设备计算时间分别记录。任务加速比使用可比时间域中的实际事件；复杂动作替换不能直接套固定chunk周期公式，见 [加速比协议](../../docs/protocols/speedup-metrics.md)。

共享事件/指标约定与 [轨迹工具](../../tools/embodied/README.md)，executor/backend按执行能力复用。DOM与Kinetix分别维护自己的控制循环和baseline，不强制共用一个动态Runner。Issue标记 `dynamic/<case>`；共享底层改动标记 `shared`。静态任务见 [独立入口](../static/README.md)。
