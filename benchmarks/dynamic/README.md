# 动态任务实验入口

动态任务按case维护自己的模型、环境与时间协议，共享统计和产物约定。

| Case | 当前状态 | 执行路径 |
| --- | --- | --- |
| [KINETIX](kinetix/README.md) | 已有仓内源码和可执行入口 | RTC JAX flow policy＋KINETIX/Jax2D；native-blend延迟、flow步数/延迟配对矩阵、完整视频与统计 |
| [DynamicVLA＋DOM](dynamicvla_dom/README.md) | 原生单环境入口 | 仓内模型/DOM源码、独立Isaac环境、原生non-streaming/streaming；不含paper_sync |

KINETIX的实际模型、环境、物理核心和关卡代码位于 `src/robotics_bench/kinetix/`，没有外部源码运行依赖。框架环境和checkpoint路径通过case模板配置。DOM不会因KINETIX接入自动获得支持，也不与静态π0.5/Cosmos权重构成任意组合。

KINETIX默认保持原生物理网格，通过执行器指令混合近似分数时间槽内的动作切换；旧控制继续推进世界。它与静态 `paper_async` 的历史观测回取不同，也不声称已实现真实并发。详见 [时间协议](../../docs/protocols/simulation.md)、[参考代码整理](../../docs/kinetix-reference-review.md) 和 [case命令](kinetix/README.md)。

运行产物放 `runs/dynamic/<case>/`。报告分别注明虚拟模拟时间、host调用时间、质量与预算，不能把注入延迟、flow采样步数或单次smoke当作真实硬件加速比或任务集结果。静态入口见 [static](../static/README.md)。
