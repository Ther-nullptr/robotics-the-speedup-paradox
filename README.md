# Robotics: The Speedup Paradox

面向模型推理、机器人仿真与任务级评测的协作基础设施。**当前版本先建立开发协议和可执行的数据契约**，没有模型推理、量化 kernel 或模拟器实现；论文复现不作为参与开发的前置条件。

计划覆盖 Cosmos-Policy / LingBot-VA 的静态实验、π0.5 的静态实验，以及动态任务仿真。各方向共享计时、实验记录和可视化事件协议；具体模型和设备实现按任务卡逐步接入。

## 从这里开始

需要 Python 3.11 或 3.12。以下命令从本仓库根目录运行，不需要 GPU、模型权重或模拟器：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/validate_contracts.py --examples
python -m pytest -q
```

如果使用 `uv`，也可以用它创建独立 Python 3.11 环境：

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

先阅读 [贡献协议](CONTRIBUTING.md)，再选择一张 [任务卡](docs/tasks/README.md)。使用 AI 时，把任务卡和 [AGENTS.md](AGENTS.md) 一起提供；验收通过后提交 PR，未完成的工作留下 [交接记录](docs/handoffs/README.md)。

## 已提供的基础

| 入口 | 内容 |
| --- | --- |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 分工、分支、测试、审阅与完成标准 |
| [.github](.github) | Issue/PR 模板、CODEOWNERS、CPU CI 定义 |
| [架构边界](docs/architecture.md) | 公共核心、模型 adapter、模拟器、协议和可视化的职责 |
| [仿真协议](docs/protocols/simulation.md) | 两种时钟、同步/异步、注入延迟、观测年龄与失败预算 |
| [可视化协议](docs/protocols/visualization.md) | 阶段面板、时间轴来源、帧映射与非阻塞展示 |
| [数据契约](docs/protocols/artifacts.md) | manifest / trace schema、验证范围和扩展规则 |
| [后端状态协议](docs/protocols/backend-state.md) | reset/ack、cache scope、加载模式和 capability 边界 |
| [模型接入协议](docs/protocols/model-adapters.md) | 三类静态模型与动态仿真的输入输出和责任划分 |
| [Schema](schemas) / [合成样例](examples/contracts) | 可供未来实现共同遵循的机器可读格式 |
| [端侧优化 skill](.agents/skills/edge-inference-optimization/SKILL.md) | profile、量化/融合、正确性和任务指标的迭代方法 |

`tools/validate_contracts.py` 检查数据及部分因果关系，不模拟执行，不证明某个模型更快或某种协议满足实时性。样例数据均为 synthetic，不能用作论文实验结果。

## 当前状态与后续工作

协作文件与 schema 校验可用于本地开发。GitHub CI、必需 review、分支保护需要在仓库远端配置并实际运行；本地文件不代表这些远端规则已生效。

后续先完成 CPU 确定性调度和 trace 可视化，再按 [任务卡](docs/tasks/README.md) 并行接入模型及模拟器。实现用 `planned / implemented-unverified / verified` 标明状态；只有带设备、配置、测试和 artifact 的结果才能标为 verified。

本仓库是独立发布边界。通用模块未来供其他项目通过版本依赖复用；本仓库安装、测试和公共文档不得依赖父目录或未发布代码。

背景论文：[The Speedup Paradox: Rethinking Inference Speed-Quality Trade-off in Embodied Tasks](https://arxiv.org/abs/2606.28529)。本版本提供工程协议，不宣称复现了论文结果。

代码发布许可证尚待维护者确定；当前未附加默认授权。外部实现的引入方式见 [第三方来源记录](THIRD_PARTY_NOTICES.md)。
