# 数据契约与演进

当前机器可读格式为 **schema_version 1.0.0**。这表示序列化协议的版本，不表示模型、调度器或实时系统已实现。

规范入口：[manifest schema](../../schemas/run-manifest.schema.json)、[event schema](../../schemas/trace-event.schema.json)、[详细字段与校验边界](../../schemas/README.md)。字段枚举以 schema 为准；设计文档里的未来字段不会自动成为合法数据。

## 先验证一个例子

从仓库根目录执行：

```bash
python tools/validate_contracts.py --examples
python tools/validate_contracts.py --manifest examples/contracts/synthetic.manifest.json --trace examples/contracts/synthetic.trace.jsonl
```

两者都应输出 `VALID synthetic-run: 8 events`。这是手工合成的一次观测、请求、完成、释放、动作和终止，不是模型实测。自己的完整 run 用第二条命令替换文件路径；非法数据退出 1，命令用法错误退出 2。工具只读取数据，不执行 manifest 中任何路径或命令。

## v1 已检查的内容

manifest 声明 run、synthetic 标记、调度与时钟/延迟两个配置轴、clock domain、episode seeds 和预算；可选 summary 不能在零成功时给出非空成功条件均值。未知字段拒绝，防止拼错参数被无声忽略。

trace 的 observation、request、chunk 和 action 必须关联同一 episode；时间在同一域中不倒退；动作在结果完成并释放后才可执行。episode 结束后拒绝新观测、请求、释放和动作，允许旧请求完成/丢弃的诊断事件。每个声明 episode 都有唯一 terminal，空或未完成 trace 不作为完整 artifact 通过。

当前一个 run 只包含串行 episodes，同一个 clock domain 的时间跨 reset 连续。多机器人/并行 episodes 暂用独立 run；不能强行把并发数据串入这个版本。扩展之前先明确 agent/session ID、时钟、共享 cache ownership 和跨 run 关联。

多进程生产者先汇集并排序，再生成当前 JSONL。event_seq 是序列化顺序；不同域的 timestamp 没有可直接相减的全局意义。这不是乱序日志收集器。

## 校验没有证明什么

校验不执行 physics、推理、队列、延迟补齐、deadline 或资源 drain；也不核实输入是否真来自硬件、summary 是否由逐次记录正确计算。物理推进、实际 observation age、失败预算执行、数值精度和性能需要对应生产者及行为测试。

当前 schema 是最小时间/关联契约。checkpoint、实际 backend、量化层 map、硬件状态、完整 profile provenance、视频帧引用等是后续字段任务。新增字段先做兼容设计与样例，不能以无约束字典绕过 schema。

`simulation.md` 的手算案例属于未来 runtime 测试规格；当前 `tests/test_contracts.py` 只证明校验器能接受/拒绝所测试的数据。

## 如何修改契约

一个 PR 同时说明生产者为何需要新字段、消费者如何处理、旧样例是否继续合法，以及需要的新成功/失败案例。所有根 schema 当前固定版本且拒绝未知字段；即便添加可选字段，旧严格 reader 也可能拒绝带新字段的数据，不能仅因字段可选就宣称完全兼容。

改变字段含义、单位、clock/ID 语义或必填条件时更新协议版本并提供显式迁移；禁止在原版本下静默改含义。文档/错误信息修订若不改变数据契约，不需要随意升级 schema_version。待需要多版本时再实现明确的 reader/迁移支持，不预先建立复杂注册中心。

持久化 artifact 使用可定位的 run ID 和相对引用；不提交私有路径、凭据、模型权重或大型原始结果。正式实验结果应另存版本、hash 和可访问的 artifact URI，仓内只保留小规模可公开例子。
