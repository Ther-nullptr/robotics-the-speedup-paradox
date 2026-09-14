# 后端状态、重置与能力声明

本文件是后续 adapter/runtime 的接口约定。当前 schema 仅记录 episode/request 生命周期，尚未实现 RPC reset、cache 管理或多任务调度；新增 wire 字段和事件先走 [契约演进](artifacts.md)。

## 会话与 episode 的边界

连接/session 是传输生命周期，episode 是实验生命周期；同一连接可以承载多个 episode。模型状态不能只在断连时清理。reset 操作应显式标识要结束的 episode、下一 episode 及随机 seed，并定义成功确认的时机。

第一版可采用同步 reset：阻止新请求、逻辑结束旧 episode、drain 在途计算、清理 episode 状态后返回完成；新 episode 的请求等 reset 完成再发送。后续允许异步 reset 时，返回完成句柄/ack，旧 epoch 的回调仍只能记诊断，不能入新队列。无法可靠取消 GPU 工作时不得提前复用其 buffer。

## 状态分区

| 状态 | 通常的有效范围 | 失效条件必须由实现声明 |
| --- | --- | --- |
| 不变权重与 packed weights | backend/model 实例 | checkpoint、量化格式、LoRA/模型权重变更 |
| workspace 与 graph 实例 | device/stream/shape/layout | shape、输入地址、graph 约束或并发 ownership 改变 |
| prefix/text/visual KV | 明确的任务或 episode | prompt、输入、模型版本、mask/position、任务切换及 reset |
| action/temporal state | episode | 已执行 offset、时间窗口、episode reset |
| 请求输入输出 | request 至消费者用完 | completion、队列消费、drop/cancel 后可靠 drain |

表中的“通常”不是可直接实现的缓存策略。每个 adapter 给出实际 cache key、共享 scope、写权限与 invalidation 条件。读取窗口缩小、存储容量缩小和减少写入是三项独立变化；它们分别影响算量、内存和状态一致性。

跨任务共享必须证明 checkpoint/adapter/precision、输入前缀、position/mask 等键满足等价条件。多个请求共享只读 tensor 时保留引用至所有消费者完成；共享可写 workspace 则需隔离或串行。先在一个模型中验证，不能把 π0.5 的缓存布局强加给 Cosmos/LingBot。

## 后端能力与加载模式

backend 明确支持的模型变体、shape/dtype/layout/device、并发方式、graph/cache 能力及 fallback。未支持的请求要显式失败或记录选定的 reference fallback，不能无声切换精度。

模型加载区分真实 checkpoint 与 synthetic/random 初始化。找不到指定 checkpoint 应报错；只有调用方显式选择 synthetic 模式才能随机初始化，而且产物必须继续带 synthetic 标记。运行 manifest 至少计划记录 checkpoint/adapter 与预处理摘要、normalization/action 约定、实际 backend；这些 provenance 字段尚需加入 schema，不能把当前最小 schema 当完整模型复现实验格式。

## 后续验收

一个连接连续运行两个 episode 与重新建立两个独立 reference 实例的结果应在预定条件下相符；reset 后旧请求晚完成不能污染新状态；修改 prompt、mask、shape 或 LoRA 后按声明失效；同样输入的测试需固定噪声而非依赖偶然重复。

性能计时明确是否包含 normalization、tokenization、vision、prefix、action suffix、排队和 transport。跨帧完成的文本或辅助任务按完整 request 记录 start/end，未完成请求也保留状态；不能只统计已完成任务或用每帧均值冒充请求延迟。

首版每个 run 仍是串行 episodes；多任务共享 cache 与多 agent 并发是后续有独立需求和任务卡的扩展，不因本协议提及它们而宣称已实现。
