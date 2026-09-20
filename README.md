# Robotics: The Speedup Paradox

面向模型推理与机器人实验的基础工具。目前提供CPU数据契约校验、baseline加速比计算、轨迹分析，以及 [π0.5＋LIBERO](benchmarks/static/pi05_libero/README.md) 和 [Cosmos-Policy＋LIBERO](benchmarks/static/cosmos_libero/README.md) 运行入口。两者使用调用者提供的兼容外部源码与checkpoint，支持同步基线及论文静态抽象 `paper_async`。Cosmos 通过独立 engine、simulator 和单环境 runner 接入，当前支持原精度，已完成固定输入动作对齐及同步、论文异步各一个GPU闭环smoke。

实验按静态和动态任务分别组织。每个case绑定模型、权重、任务、环境及协议，只在已验证的范围内选择量化或调度方案。

静态异步按 The Speedup Paradox 的历史观测实验与周期模型定义：`paper_async` 通过历史快照实现观测陈旧，并用声明的推理/动作时间估计重叠收益。真实推理与控制并发可作为独立扩展，不是该模式的验收前提。论文周期、宿主运行时间及仿真时间分别记录，见 [仿真协议](docs/protocols/simulation.md)。

| 实验入口 | 当前规划 |
| --- | --- |
| [静态任务](benchmarks/static/README.md) | π0.5＋LIBERO；Cosmos-Policy＋LIBERO已有独立入口；Cosmos/RoboCasa与LingBot-VA仍待接入 |
| [动态任务](benchmarks/dynamic/README.md) | DynamicVLA＋DOM；[Kinetix](benchmarks/dynamic/kinetix/README.md)保留原生JAX策略、环境与rollout |

两个目录分别组织case。π0.5＋LIBERO已有显式路径配置、CPU预检、权重加载审计与外部评测桥接；Cosmos＋LIBERO新增本仓库控制生命周期、原生模型/环境适配和逐请求历史观测记录，保留基线 `1a4983e` 已有的π0.5功能。两条路径共享事件/指标约定和分析工具，各case维护自己的控制协议、baseline和验证范围。其他后端按具体任务需求接入，具体方向见 [工作路线](docs/tasks/README.md)。

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

π0.5＋LIBERO实验可直接使用命令行：先按 [case说明](benchmarks/static/pi05_libero/README.md) 配好本机路径，再运行 `bash benchmarks/static/pi05_libero/run.sh --schedule paper_async --overlap-actions 2 --quant none --gpu 3 --output-dir runs/static/pi05_libero/trial-001`。`--quant w8a8-single-layer` 自动选择单层量化预设；常用设置无需手写JSON，manifest与结果文件由入口生成。

Cosmos＋LIBERO先按 [case说明](benchmarks/static/cosmos_libero/README.md) 配好外部资源和 `ROBOTICS_COSMOS_PYTHON`，再用 `bash benchmarks/static/cosmos_libero/run.sh --task-ids 0 --episodes 1 --quant none --output-dir runs/static/cosmos_libero/trial-001 --dry-run` 做CPU预检。实际运行去掉 `--dry-run` 并显式传 `--gpu`；日志、episode/请求记录和汇总由入口自动保存。现有环境复用不等于完整新机安装，checkpoint和运行依赖不包含在上述CPU开发依赖中。

## 工具与规范

缺少模型和模拟器资产时，使用 `python tools/prepare_resources.py --case cosmos_libero --dry-run` 查看资源计划，或选择 `pi05_libero`。按 [资源准备说明](tools/RESOURCE_PREPARATION.md) 安装下载依赖后去掉 `--dry-run`：默认模型/可选数据集缓存为 `~/.cache/robotics/hub/`，LIBERO资产为 `~/.cache/libero/assets/`；生成的 env 显式加载后可用于现有启动项。训练轨迹使用 `--with-dataset` 单独启用。

| 入口 | 内容 |
| --- | --- |
| [贡献协议](CONTRIBUTING.md) / [AGENTS.md](AGENTS.md) | Issue、PR、验收与AI协作 |
| [架构](docs/architecture.md) / [图稿源码](docs/diagrams/README.md) | 模块边界及可重建的架构图 |
| [数据契约](docs/protocols/artifacts.md) / [schemas](schemas/README.md) | manifest/trace格式及校验范围 |
| [组合实验](docs/protocols/composable-experiments.md) / [加速比口径](docs/protocols/speedup-metrics.md) | 同模型任务下的方案比较与计算 |
| [实验步数汇总](tools/summarize_experiment.py) / [统计口径](docs/protocols/speedup-metrics.md#8-控制步数统计与实验汇总) | 从episode记录汇总全体/成功控制步数及失败预算惩罚；多个运行分别输出 |
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
