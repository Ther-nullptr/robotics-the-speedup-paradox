# 020：离线 trace 状态回放

- Owner / reviewer：认领时填写。
- 状态：ready；可直接使用合成样例。
- 建议文件：`src/robotics_bench/visualization/`、`examples/visualization/`、相关测试。

先阅读 [可视化协议](../protocols/visualization.md)，再制作一个本地离线 viewer，读取 manifest 与 JSONL，不要求 Web 服务或模型运行。显示 observation、inference、release、action、terminal 五类事件；动作可追溯至输入观测。环境视频尚不存在时显示明确的“无帧”状态，不用假截图冒充实际仿真。

验收：选中动作能显示正确关联 ID、同域 age；跨域时间明确不可直接比较；错误 trace 给出诊断；非法/未知事件不会静默画成正常动作。示例始终标注 synthetic。可将关键状态映射写成纯函数测试，布局用实际截图检查。

后续 PR 才接动态环境帧和实时订阅。实时显示不能反向控制 env.step；慢消费者只影响显示，不拖慢控制循环。
