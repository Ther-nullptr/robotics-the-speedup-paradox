# 040：动态仿真 adapter 与延迟源

- Owner / reviewer：认领时填写。
- 状态：依赖 010；先完成确定性 CPU 验收。
- 建议文件：`src/robotics_bench/simulators/`、`src/robotics_bench/experiments/`、对应测试。

先实现匀速目标等可手算动力学和 scripted policy，验证推理等待中世界继续变化。每个环境明确 physics dt、control dt、等待时机器人动作、terminal/预算时钟和 render 副作用；不能靠 sleep 冻住世界。

验收：1 m/s 的目标在 100 ms 模拟延迟中移动 0.1 m；virtual profile 不双算 host 耗时；target-total 超期记录 overshoot；零延迟消融独立命名；terminal 后请求只 drain 不执行。profile 保存原始样本与 scope，而非只有一个设备标签。

契约通过后逐个接 Kinetix、DOM 或选定真实环境，不先扩大到全部任务。原论文离散 delay bin 如需保留，以 legacy 协议单独提交，不改变新协议来凑旧曲线。
