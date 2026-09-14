# 初始协议建设交接

- 任务：建立协作、模型/仿真接口和可执行数据契约；论文复现不作为前置。
- 分支：`main`；基线为本仓库初始协议提交，可通过 `git log --oneline` 定位。
- 状态：本地协议基座完成，待后续贡献任务；远端 CI/保护/发布尚未执行。

## 已确认事实

当前包含协作文件、8 张后续任务卡、JSON schema、合成 trace、校验 CLI、50 个 CPU 测试和项目内优化 skill。没有推理框架、量化 kernel、模拟器修改或实际 viewer。

使用独立 Python 3.11 环境执行以下命令通过：

```bash
python -m ruff check tools tests
python -m ruff format --check tools tests
python tools/validate_contracts.py --examples
python -m pytest -q
```

结果分别为检查通过、2 个文件无需格式调整、`VALID synthetic-run: 8 events`、50 tests passed。将本仓库内容复制到独立目录后，从外部 cwd 调用样例 CLI 同样通过；公共文档不依赖父目录。

## 当前限制

一个 run 仅接受串行 episodes；同域 timestamp 跨 reset 连续；当前不是乱序事件收集器或多 agent scheduler。校验器不验证物理推进、延迟计算、队列容量、资源 drain 或 metrics 是否等于真实结果。

`simulation.md`、`backend-state.md` 和 `visualization.md` 的行为验收是后续任务规格，不能引用为已测试实现。模型、设备、完整 provenance 和媒体字段仍需按任务扩展。

## 接下来先做什么

1. 从 README 运行合成样例，点读一条 action 的 observation→request→chunk 来源。
2. 认领 [010 CPU 调度](../tasks/010-cpu-scheduler.md) 或 [020 Trace viewer](../tasks/020-trace-viewer.md)，按小 PR 推进。
3. 维护者确定共同 owner、许可证与远端启用时间；模型负责人另行提供接入来源、权重/预处理和设备设置，不阻塞上述 CPU 工作。

没有运行中的模型/GPU 作业。未运行远端 CI，也未创建 GitHub Issue/PR。
