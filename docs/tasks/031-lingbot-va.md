# 031：LingBot-VA 静态实验与 cache 契约

- Owner / reviewer：认领时填写。
- 状态：接口梳理可开始；真实对齐需确认原代码 commit、权重及环境。
- 建议文件：`src/robotics_bench/policies/lingbot_va/`、对应 configs 与测试。

先识别 visual/action KV 的写入、可读窗口、物理存储、reset 和输出动作生命周期。读取裁剪与存储裁剪分开命名，避免把少读 token 等同于少占内存。生成 steps、历史 chunk 数与执行动作数分开配置。

验收：固定输入/噪声的 reference action 对齐；跨 episode reset 不残留旧状态；窗口配置改变实际读取路径；cache 写入/分配和 attention 时间独立记录。首个 PR 可以只固定接口和边界测试；不要同时重写 cache layout 和调度协议。

完整 RoboTwin 静态 rollout、性能与 quality sweep 后续提交，协议优先不要求先复现原图。
