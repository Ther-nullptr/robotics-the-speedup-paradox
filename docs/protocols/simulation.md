# 仿真时间、异步推理与实验记录协议

版本：设计规范 v0.1。当前提供 manifest/trace 的 schema 与部分事件一致性校验；调度器、模拟器和推理 worker 尚未实现。可执行校验的范围以 schemas、tools/validate_contracts.py 和 tests 为准。

协议的核心是区分 **物理环境时间、程序墙钟时间、动作等待时间和观测年龄**。相同 policy、seed、delay profile 和协议版本应得到可解释、可回放的事件序列。

## 两条独立的配置轴

调度协议与延迟计量分别配置，避免一个 `async=True` 同时改变几个实验因素。

| schedule | 定义 | 用途 |
| --- | --- | --- |
| `sync` | 采样、推理就绪、执行配置数量的动作，再发下一请求 | 同步基线 |
| `history_observation` | 同步 host 控制流，但按论文设置提供过去 n′ 个动作对应的观测 | 复现论文的抽象异步实验；明确不宣称真实并发 |
| `async` | 当前动作执行中发起后续请求；结果就绪后按队列规则接管 | 验证真实 overlap 和状态一致性 |

| clock / delay mode | 推理时环境如何推进 | 延迟来源 |
| --- | --- | --- |
| `virtual` + `zero` | 推理视为无仿真时间成本 | 质量消融；实际计算仍消耗 host 时间 |
| `virtual` + `profile_replay` | 按固定 dt 推进到请求的模拟就绪时刻 | 同模型/形状/配置/设备的实测 profile 或显式合成分布 |
| `realtime` + `measured` | 环境按墙钟 deadline 继续推进，推理 worker 独立运行 | 本次实际 pipeline 延迟 |
| `realtime` + `additive` | 在实际就绪后增加额外等待，世界继续推进 | 实际延迟 + 注入延迟 |
| `realtime` + `target_total` | 仅补足目标总延迟的剩余部分 | `max(0, target - measured_so_far)`；超过目标必须记录 overshoot |

`virtual` 下真实 CPU/GPU 推理耗时只作为 profiling 数据，不再叠加进已经重放的目标延迟，否则会重复计费。若宿主计算比目标 profile 慢，虚拟时钟在最早需要该结果的事件处暂停等待宿主完成，host elapsed 增长但 sim time 不追加此等待；这不是实时能力证明。`zero` 也不是“模型真能零延迟”，报告必须带模式名称。`history_observation` 的 n′ 到 timestamp 映射必须按原实验代码核对，不能自动套用真实异步的等待公式。

## 时钟及事件字段

最小事件 envelope：`schema_version, run_id, episode_id, event_seq, event_type, clock_domain, timestamp_ns`。每个进程另记 `process_id, monotonic_origin`；不同机器的 monotonic 时间不能直接相减。需要跨机器关联时保存时钟对齐方式及误差，否则只报告各域内部 duration。

关键关联字段：`observation_id, request_id, chunk_id, action_index, sim_tick`。配置和 artifact 用独立 manifest 存储并通过 hash 关联，事件中无需重复塞入完整配置。

| 事件 | 时刻的定义 | 最少附加数据 |
| --- | --- | --- |
| `observation_sampled` | 输入实际采样完成 | observation ID、采样 sim tick、输入引用 |
| `inference_submitted` | runner 把 immutable snapshot 交给 policy | request ID、来源 observation ID |
| `inference_started` | 请求实际开始处理 | backend、worker/stream ID |
| `inference_completed` | 动作输出可被消费，含必要同步和后处理 | output/chunk ID、完成状态 |
| `result_released` | 实际或注入的等待已结束，允许入队 | delay mode、sample、profile ID、overshoot |
| `action_started` / `action_finished` | 每个 primitive action 真正开始/完成 | 来源 chunk、action index、来源 observation ID |
| `queue_underrun` / `result_dropped` | 无可执行动作 / 按协议弃结果 | 原因、队列长度、epoch |
| `episode_finished` | 首次 success、timeout 或 task failure | terminal reason、成功标记、budget 类型 |

device kernel 计时单列为 `duration_ns` 及设备时钟域；host trace 与 CUDA event 不能假装共用 epoch。实际异步请求完成应对应动作真正可读，不能把 enqueue 的返回时刻记录成完成。

## 三个常被混淆的量

```text
policy_pipeline = output_ready - request_started
observation_age(action) = action_started - observation_sampled
boundary_wait = max(0, next_action_started - previous_chunk_boundary)
release_wait = max(0, result_released - previous_chunk_boundary)
```

这些差值必须在同一个有效时间域计算。后续 action 的 age 会继续增长，故每个动作都保留来源 observation，不能只在 chunk 首动作测一次。

手算例子：t=0 ms 采样新观测并开始推理，旧 chunk 还会执行到 t=120 ms，新结果在 t=80 ms 就绪，首个新动作在 t=120 ms 开始。此时 `boundary_wait=0`，`observation_age=120 ms`，`policy_pipeline=80 ms`。完全隐藏推理等待也不能消除观测陈旧。若新结果 t=180 ms 才就绪，则等待 60 ms、观测年龄 180 ms。该例应成为规范测试。

历史分析中的剩余等待代理量应单独命名；实际 observation age 始终通过动作与观测时间戳获得。历史观测回取与真实异步是两个独立协议，不能混称为已测得的并发性能。

## 环境推进与队列规则

静态质量基线可在推理期间冻结 simulator；动态实验要在延迟期间推进世界动力学。仅 `time.sleep()` 而不调用环境 step，不能模拟物体在推理时运动。是否保持机器人最后一个控制指令、保持位置或采用专用 idle controller 由 task adapter 定义；位置控制下“全零动作”未必是安全保持，不能作为通用默认值。

新的虚拟协议用整数 tick 或时间累加器推进，使长期累计的控制时刻与目标时间一致；不要每次延迟独立四舍五入造成系统偏差。历史 Kinetix 复现可单独使用 `legacy_round` 映射以保留原论文 round(tau/33.33ms) 语义，禁止把新映射结果混作历史曲线。小于 physics dt 的延迟也必须保留余量。控制 dt、physics dt、实际积分 substeps 和剩余时间都写入配置。

首版 async 限定每个 policy 一个在途请求，队列有界。观察线程只能写入 snapshot/mailbox，policy worker 不能操作可变 env。触发点由“队列还剩多少动作”或明确模拟/墙钟时间决定；`overlap_actions=n′` 是实验参数，其有效上限受实际队列和推理延迟约束。

默认先完成旧 chunk 再接新 chunk，便于对齐同步 baseline。后续 `replace_pending` 或按 action valid-time 丢弃过时前缀作为独立协议增加，声明新旧 chunk 的交接位置、空队列行为及 late-result 行为。禁止静默重复最后动作、无限缓存结果或随意截断输出来改善数字。

episode reset 增加 epoch；旧 epoch 的 Future 即使完成也必须丢弃。episode 在推理/注入等待期间达到 terminal，应立即逻辑结束并拒绝动作；在途 CPU/GPU 请求隔离并等待完成或可靠取消后，才能释放其输入输出和 workspace，结果稍后返回不能继续执行。必须覆盖推理异常、worker 退出、取消/超时、输出非法值、队列耗尽与 late completion。

真实时间模式的 simulator runner 不等待 `Future.result()` 后才判断是否应该 step。worker 与 runner 需避免 GIL、CUDA 同步或渲染导致隐性串行；可用独立进程，但选择须以 trace 验证。runner 记录 deadline miss 和 real-time factor；无法维持控制频率时报告超期，不把 host 调度拖慢误报成可控注入延迟。

## 可复现实验 manifest

每个 run 保存：代码 SHA/dirty patch hash、配置 hash、模型 checkpoint hash、adapter/backend 版本、量化 layer manifest/calibration/pack hash、软件环境、GPU/CPU/驱动/功耗/频率/热状态、warmup 与测量次数、任务与 reset seed 清单、控制/物理 dt、chunk 长度、执行长度、协议版本、delay source 与采样 seed、队列/过期规则、预算及成功判据。

delay profile 附带 scope（仅 forward 或完整 perception→action pipeline）、硬件、输入形状、精度、denoise step、预热方法、原始样本。均值重放、IID 重采样、保序 trace 重放应作为不同实验；发生时间相关抖动时只保存均值无法重现尾部风险。远程网络延迟单列。设备 label 不足以唯一标识 profile。

实验拆成三组：固定零延迟测质量；固定 policy 测延迟响应；使用真实硬件 profile 联合比较。跨方法同时保留原始配置和实测 inference speedup，按相近延迟或质量约束比较，不能将 INT4 层数与 denoise steps 的横坐标直接视作同等强度。

## 任务级指标及失败

每组必须报告 trials、successes、SR 与区间、成功条件下 task time/chunk count、所有 rollout 的终止原因及耗时、timeout budget。manifest 明确 task budget 的 sim/wall clock domain、起点事件以及独立 host watchdog；不同时间域预算不能混比。成功数为 0 时成功条件时间为 null，不能填 0 或以空均值参与排序。

保留两套明确命名的失败处理：

1. `paper_legacy`：为需要兼容的历史实验，按它的动作/chunk 上限和配置相关 `Tmax=Nmax*Tchunk` 计算 penalty，并显示其 budget。
2. `fixed_budget`：新的跨配置主比较固定任务秒数预算 B，`penalized_time=(sum(success completion time)+failures*B)/trials`。同时报告真实终止耗时，避免将 penalty 误称为实测耗时。

失败包含 task failure 与 timeout；运行基础设施错误单独计数，不能无声删除或当成策略失败。若不同配置成功的 seed 集不同，成功条件均值可能有选择偏差；报告共同成功 seed 的配对结果及全部 seed 的 SR/penalty。区间计算以独立 episode 为单位；多任务结果分任务汇报后再定义聚合权重。

静态理论用 `N*Tcycle` 作解释，但真实 trace 允许首尾不完整 chunk、变长推理和异步 pipeline fill/drain，主任务时间取事件测量，不能硬套常数公式。把 simulated task time、wall elapsed 和 CUDA duration 分列。

TISED 的分量拟合、选点与验证使用分开的配置/seed 或 held-out 硬件。明确 O(n) 表述依赖分量可校准和可迁移假设，计入校准成本，并检查非分离的交互项。N 或质量曲线不强行单调拟合；模型不确定性要传递到候选 sweet spot。

## 最小验收场景

| 场景 | 应观察到的结果 |
| --- | --- |
| 同步：2 次推理各 80 ms、每次执行 2×50 ms | 虚拟串行总时间 360 ms，无漏计/重复计延迟 |
| t=0 采样、80 ms 就绪、120 ms 接管 | age=120 ms，boundary wait=0 |
| t=0 采样、180 ms 就绪、120 ms 旧块结束 | age=180 ms，wait=60 ms，欠载事件 |
| 100 ms 注入、世界目标速度 1 m/s | 无其他作用时目标推进 0.1 m；冻结基线为 0 |
| 连续多次非整数 dt 延迟 | 累积推进与总延迟吻合，误差有明确界限 |
| 虚拟 profile=100 ms、host 计算=25 ms | 模拟只计 100 ms；host 25 ms 单列 |
| 虚拟 profile=100 ms、host 计算=250 ms | host 等待但 sim 就绪仍为 100 ms；不宣称满足实时 deadline |
| 墙钟 target=100 ms、实际=120 ms | 不注入负延迟，overshoot=20 ms |
| reset 后旧请求完成 | 没有跨 episode 动作；有 dropped reason |
| 等待中 success/timeout | 恰好一个 terminal，后续动作不执行 |
| 0 个成功 episode | SR=0，successful time=null，penalty=预算 |
| 可视化消费者卡住 | 控制逻辑继续，显示丢帧可计数 |

此表是后续调度器任务的验收规格。当前契约校验只覆盖字段与事件一致性；没有执行这些调度行为测试，也没有运行模型。后续先用 CPU fake clock/policy/simulator 实现时间线，再接真实模型。
