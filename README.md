# Robotics: The Speedup Paradox

面向模型推理与机器人实验的基础工具。目前已提供 CPU 数据契约校验、baseline 加速比计算和轨迹分析；模型推理后端与模拟器接入仍在规划中。

实验按静态和动态任务分别组织。每个case绑定模型、权重、任务、环境及协议，只在已验证的范围内选择量化或调度方案。

| 实验入口 | 当前规划 |
| --- | --- |
| [静态任务](benchmarks/static/README.md) | π0.5＋LIBERO；Cosmos-Policy＋对应LIBERO/RoboCasa任务；LingBot-VA的任务组合待明确 |
| [动态任务](benchmarks/dynamic/README.md) | DynamicVLA＋DOM；[Kinetix](benchmarks/dynamic/kinetix/README.md)保留原生JAX策略、环境与rollout |

两个目录已提供case范围、文件归属和接入验收说明，运行代码尚未实现。两条路径共享事件/指标约定和分析工具，按能力复用执行与backend组件；各case维护自己的控制协议和baseline，任务类型与同步/异步调度分别声明。其他后端按具体任务需求接入，具体方向见 [工作路线](docs/tasks/README.md)。

## 快速开始

Python 3.11 或 3.12，从本仓库根目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/validate_contracts.py --examples
python tools/compare_speedups.py --input examples/speedups/synthetic.json --markdown
python tools/embodied/trajectory_metrics.py --input tools/embodied/examples/smooth.csv
python -m pytest -q
```

也可以用 `uv venv --python 3.11 .venv` 创建环境，再运行 `uv pip install -r requirements-dev.txt`。上述工具不加载模型或使用GPU；示例是synthetic数据。绘图额外安装 `tools/embodied/requirements-plot.txt`，详见 [轨迹工具说明](tools/embodied/README.md)。

## 工具与规范

| 入口 | 内容 |
| --- | --- |
| [贡献协议](CONTRIBUTING.md) / [AGENTS.md](AGENTS.md) | Issue、PR、验收与AI协作 |
| [架构](docs/architecture.md) / [图稿源码](docs/diagrams/README.md) | 模块边界及可重建的架构图 |
| [数据契约](docs/protocols/artifacts.md) / [schemas](schemas/README.md) | manifest/trace格式及校验范围 |
| [组合实验](docs/protocols/composable-experiments.md) / [加速比口径](docs/protocols/speedup-metrics.md) | 同模型任务下的方案比较与计算 |
| [轨迹小工具](tools/embodied/README.md) | 路径、速度、加速度、jerk和绘图 |
| [仿真协议](docs/protocols/simulation.md) / [多后端](docs/protocols/simulator-backends.md) / [依赖修改](docs/protocols/simulator-dependencies.md) | 时钟、环境推进、版本与fork接入 |
| [模型接口](docs/protocols/model-adapters.md) / [后端状态](docs/protocols/backend-state.md) / [可视化](docs/protocols/visualization.md) | adapter、cache/reset、事件与帧映射 |
| [edge-model-lightweight 外部 skill](https://github.com/Ther-nullptr/edge-model-lightweight-skill) | 后续搭建与优化的主要方法参考：完整计时、profile、收益估算和质量验证；按访问权限独立安装 |
| [端侧优化 skill](.agents/skills/edge-inference-optimization/SKILL.md) | 按实测热点优化并核对质量 |

## 版本管理边界

Git保留代码、测试、依赖、schema、小型合成样例、工具说明、稳定协议和图稿源码。研究综述、阶段任务卡、个人交接记录、生成图表/HTML、运行日志、性能分析报告和临时备份留在本地；具体忽略规则见 [.gitignore](.gitignore)。共享任务状态与交接摘要记录在Issue/PR，不要求为每次工作新增Markdown文件。

从干净克隆可运行工具并阅读必要说明。生成的SVG/HTML/PNG按需重建，跟踪文档不依赖这些本地产物。根工作区的研究资料不属于本仓库。

CI定义和协作模板已提供；远端CI、分支保护和review权限尚待维护者启用。模型/模拟器功能未实现前，不将协议校验或合成结果称为性能或闭环实验验证。

背景论文：[The Speedup Paradox](https://arxiv.org/abs/2606.28529)。自有代码许可证尚待维护者确定，第三方引入按 [来源记录](THIRD_PARTY_NOTICES.md) 处理。
