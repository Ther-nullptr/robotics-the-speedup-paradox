# 静态任务实验入口

本目录组织静态任务的case、运行入口和实验配置。[π0.5＋LIBERO](pi05_libero/README.md) 正式入口已完成原精度和单个文本层W8A8替换的同步单episode smoke，同目录还提供W8A8/W4A4 synthetic Linear检查入口；其他case仍在规划。全模型量化质量、真实异步及完整benchmark结果均未验证。

| case入口 | 绑定范围 | 当前状态与接入边界 |
| --- | --- | --- |
| [pi05_libero](pi05_libero/README.md) | 原生 LeRobot π0.5＋LIBERO任务 | 正式入口 `libero_object` 单 episode、157步成功仅为 smoke，不作任务集 SR。使用任务匹配权重及同源processor；不使用VLASH微调权重，pure_async待实现 |
| `cosmos_libero/` | Cosmos-Policy＋LIBERO任务 | 对应checkpoint、统计量、文本编码与相机/动作处理 |
| `cosmos_robocasa/` | Cosmos-Policy＋其适配的RoboCasa任务 | 对应checkpoint、环境fork/资产版本、控制器和预后处理 |
| `lingbot_va/` | LingBot-VA静态任务 | 任务/模拟器/checkpoint组合待明确；cache与历史状态独立定义 |

除上述 π0.5＋LIBERO 单 episode 路径外，表内其他 case 均为 planned。目录名表示接入范围；运行配置必须进一步固定任务/初态清单、checkpoint与processor hash、环境、硬件、控制周期、预算和baseline。一次 smoke 不代表其他任务、权重或设备组合已验证。

## 文件归属

- 每个case的启动代码与配置放 `benchmarks/static/<case>/`；当前 `pi05_libero/` 使用 `run.py`、`case.json`、加载审计模块和独立 `quant_smoke.py`。其他 case 在实际接入时一起创建并验证，不提前放占位文件。
- 模型计算与预后处理放 `src/robotics_bench/policies/`；环境适配放 `src/robotics_bench/simulators/`；静态任务控制协议放 `src/robotics_bench/protocols/static/`。这些源码目录仍待实际实现。
- baseline、量化、同步/异步变体属于同一个case。配置引用固定版本的模型/环境与优化方案，运行产物放仓库已忽略的 `runs/static/<case>/`。

静态任务可以异步运行，也可以发生正常物理演化。任务分类、调度策略和等待期间世界推进分别声明；本路径不依赖DOM的时延补偿或过期动作规则。

## 接入与验收

先用固定输入与噪声对齐原生动作，再验证单任务同步闭环，之后比较同case支持的量化/异步变体。完整论文复现不是前置条件。

每组记录实际backend、完整policy延迟、动作交接、成功率和任务耗时；按该case及硬件的baseline计算加速比。轨迹/jerk分析复用 [现有工具](../../tools/embodied/README.md)。配置和验收遵循 [组合实验](../../docs/protocols/composable-experiments.md)、[模型接入](../../docs/protocols/model-adapters.md) 与 [加速比协议](../../docs/protocols/speedup-metrics.md)。

开发任务在Issue中标记 `static/<case>`；共享底层改动标记 `shared`。动态任务有 [独立入口](../dynamic/README.md)，两条路径不共用任务baseline。
