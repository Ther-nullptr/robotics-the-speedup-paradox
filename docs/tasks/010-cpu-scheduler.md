# 010：用 CPU 时间线实现调度协议

- Owner / reviewer：认领时填写。
- 状态：ready；不需要 GPU、模型或外部模拟器。
- 建议文件：`src/robotics_bench/protocols/`、`examples/scheduling/`、`tests/test_scheduling.py`。

先实现 fake clock、scripted policy 和最小 simulator；依次提交同步、延迟注入、异步队列三个 PR。只实现每个 policy 一个在途请求和有界队列，避免一开始引入分布式服务。输出遵循当前 trace 契约。

验收取自 [仿真协议](../protocols/simulation.md)：2×(80+2×50)=360 ms；80 ms 就绪/120 ms 接管对应 age=120、wait=0；180 ms 就绪对应 wait=60；terminal 后不执行旧请求；profile 重放不重复计 host 时间。为未对齐 control dt 与队列欠载提供边界案例。

每次 PR 先用手写期望时间线验证，再实现行为；不要用被测调度器自己算出的答案作为期望值。记录 input buffer 的持有和 drain，不只在 Future 状态上模拟释放。

不在范围内：真实模型、GPU kernel、完整论文曲线、网络服务。首个版本以确定性正确性为完成标准。
