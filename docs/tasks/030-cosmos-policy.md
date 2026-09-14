# 030：Cosmos-Policy 静态实验 adapter

- Owner / reviewer：认领时填写。
- 状态：接口梳理可开始；真实对齐需确认原代码 commit、权重及环境。
- 建议文件：`src/robotics_bench/policies/cosmos_policy/`、对应 configs 与测试。

首个 PR 只固定 observation、图像/状态预处理、action horizon/执行长度、DiT 调用、动作解码、reset 的边界。固定输入及噪声后将 adapter 与上游 reference 动作比较；记录来源与许可，不复制整个训练项目。

第二个独立 PR 才导出量化站点清单，区分模块路径、W/A/累加 dtype、scale/pack/backend/fallback。后续分别实现量化 GEMM 与外围融合。实际操作点来自代码映射，不能仅用“W4 t3”作为完整配置。

验收：reference action 对齐到事前约定容差；adapter 不调用 env.step/sleep 注入延迟；trace 有输入来源；CPU 契约检查能运行。完整 RoboCasa rollout 和论文数字不作为这个基础 PR 的前置。
