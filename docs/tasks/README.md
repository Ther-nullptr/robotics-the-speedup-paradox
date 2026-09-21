# 工作路线

这里保留长期方向。具体owner、范围、依赖、验收和接力状态放在Issue/PR；[任务模板](../templates/task.md) 可直接复制使用。阶段任务草稿留本地，不作为发布内容。

从 [静态任务](../../benchmarks/static/README.md) 或 [动态任务](../../benchmarks/dynamic/README.md) 选择case，再定义本次接入或优化任务；共享工具标记为 `shared`。同一case绑定模型、权重和任务环境，不因新增adapter自动扩展组合。

| 归属 | 方向 | 前置条件与主要交付 |
| --- | --- | --- |
| 静态 | π0.5＋LIBERO | 已有object全量同步/论文异步；下一步为完整推理测量与经过质量验证的轻量化 |
| 静态 | Cosmos-Policy＋LIBERO/RoboCasa | 已有独立单环境入口与GPU smoke；扩大任务覆盖，补齐完整环境安装与推理测量 |
| 静态 | LingBot-VA＋RoboTwin | 已有同步单任务GPU smoke；扩展任务覆盖，并定义KV/VAE缓存可见性一致的延迟协议 |
| 动态 | DynamicVLA＋DOM | 单场景时钟、历史观测、延迟与动作生效、终止/reset |
| 动态 | [Kinetix原生策略与rollout](../../benchmarks/dynamic/kinetix/README.md) | 绑定checkpoint/关卡和观测动作类型，再验证carry/PRNG、JAX步进、延迟与终止；独立于VLA执行器 |
| shared | 数据契约与CPU事件语义 | 按真实需求扩展契约；可手算的时间线与错误边界 |
| shared | Trace viewer与分析工具 | 动作来源、时间映射、按case比较指标 |
| 对应case / shared | 端侧优化 | 固定模型任务和设备profile；数值与完整入口证据，共享部分再抽取 |

无需先复现整篇论文。协议与CPU工具可独立推进；实际性能和模型质量结论须等待相应实现与设备证据。

其他模拟器在任务需求明确后加入相应case。两条路径各自验收控制循环和baseline，公共底层变更应说明受影响的case。
