# 050：完成一个可归因的端侧优化案例

- Owner / reviewer：认领时填写。
- 状态：等待至少一个真实 adapter 与目标设备 baseline；不阻塞其他协议工作。
- 范围：一条已 profile 的热点路径、相关数值测试与案例记录。

使用 [端侧优化 skill](../../.agents/skills/edge-inference-optimization/SKILL.md) 的思路：先固定设备、输入、baseline、计时 scope 和质量门槛，再从 pack/concat、shared quant、norm/activation/epilogue fusion、workspace/cache 等候选中选择一个有证据的改动。

验收：实际 backend 与 fallback 可见；算子误差和 action 误差在预定门槛内；完整 policy 延迟有同环境前后样本；任务级验证状态如实记录。若 microbenchmark 变快而 policy 不变快，只接受其作为局部实验结果，不宣称部署加速。

案例记录进入 `docs/optimization-cases/`，只把可复用的已验证经验补进 skill。GEMM 占比提高本身不是完成标准。
