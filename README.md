# Robotics: The Speedup Paradox

[English](README.en.md) · [环境配置](docs/environment_setup.md) · [代码架构](docs/architecture.md) · [完整静态实验](docs/full-static-evaluation.md) · [贡献与 PR](CONTRIBUTING.md)

面向具身模型推理与闭环实验的基础设施：固定模型、任务和场景，比较同步执行、论文静态异步及后续轻量化方案，并统一记录成功率、控制步数和实验配置。

**当前为开发预览。** 已有四个可运行静态 case和一个KINETIX动态case；主要模型执行代码和可选算子在本仓维护，加载框架和部分模拟器仍依赖兼容外部源码；KINETIX模型与环境源码已仓内维护。CPU 分析工具可独立使用。项目自身代码许可证尚待确定，第三方来源见 [许可与发布状态](THIRD_PARTY_NOTICES.md)。

## 功能与验证范围

| 功能 / case | 当前能力 | 已有验证 |
| --- | --- | --- |
| [π0.5＋LIBERO](benchmarks/static/pi05_libero/README.md) | 外部 LeRobot/VLASH evaluator 桥接；原精度 sync / `paper_async` | `libero_object` 各500回合：同步494/500，n′=2异步464/500 |
| [Cosmos＋LIBERO](benchmarks/static/cosmos_libero/README.md) | 独立 engine、原生模拟器、单环境 runner；H=16 | 原生动作对齐；同一 object 任务同步137步、异步154步成功 |
| [Cosmos＋RoboCasa](benchmarks/static/cosmos_robocasa/README.md) | 三相机输入、H=32、场景初始化核对；单环境运行 | `TurnOffMicrowave` 固定场景：同步277步、异步292步成功 |
| [LingBot＋RoboTwin](benchmarks/static/lingbot_robotwin/README.md) | 独立模型worker、原生双臂环境；sync / `paper_async`整体延迟KV/VAE观测历史 | adjust_bottle相同初态单回合：同步115条、n′=2异步120条控制指令成功 |
| [KINETIX动态任务](benchmarks/dynamic/kinetix/README.md) | 仓内flow模型、环境与Jax2D；flow步数×延迟矩阵、原生物理网格 | 两关原生端点逐值对齐；8回合GPU接入验证，7成功1失败 |
| [推理优化](benchmarks/inference/README.md) | 本仓模型热路径；独立BF16融合、CUDA Graph、INT4/INT8开关；逐轮图表 | Ada算子测试、固定输入动作校验与单回合闭环；量化全量质量待验证 |
| [资源准备](tools/RESOURCE_PREPARATION.md) | 下载或复用 checkpoint、tokenizer、统计量、T5；训练数据按需下载 | 两类模型本地资源复用、真实小文件下载、启动预检 |
| [实验统计](tools/summarize_experiment.py) | 成功率、失败按预算惩罚的总体步数、仅成功步数；按任务汇总 | CLI 与实验结束时自动调用 |
| [轨迹分析](tools/embodied/README.md) | 轨迹绘制、速度、加速度、jerk | CPU 数值工具与合成样例 |
| [契约与加速比](docs/protocols/speedup-metrics.md) | manifest/trace 校验、固定 baseline 的加速比计算 | CPU 契约和显式输入计算 |

Cosmos 数字是单回合闭环结果，不能代替全任务集成功率。表中π0.5全量结果来自原始基线，不能转用于新优化配置；任务、初态和采样配置见对应case说明。量化按显式模块范围启用，尚无通过全量质量验收的量化预设。完整policy调用时延比与论文控制周期加速比分别报告。

LingBot优先适配RoboTwin静态任务；KINETIX已接入仓内JAX模型/环境与native-blend动态协议，DynamicVLA＋DOM仍在规划中。静态与动态任务分别组织，模型和模拟器按已验证的 case 绑定；接口形状兼容不代表任意组合可用。

## 快速开始：CPU 工具

推荐 Python 3.11 或 3.12；以下步骤不下载模型、不需要 GPU：

```bash
git clone https://github.com/Ther-nullptr/robotics-the-speedup-paradox.git
cd robotics-the-speedup-paradox
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

python tools/validate_contracts.py --examples
python tools/compare_speedups.py --input examples/speedups/synthetic.json --markdown
python tools/embodied/trajectory_metrics.py --input tools/embodied/examples/smooth.csv
```

示例为合成数据。绘图额外安装 `tools/embodied/requirements-plot.txt`。本仓可选安装为Python包 `python -m pip install -e .`；源码安装本身不会安装模型或模拟器运行栈。

## 准备模型并启动实验

先按 [环境指南](docs/environment_setup.md) 建立对应 case 的独立 GPU 环境，并准备兼容外部源码。下载工具运行在 CPU 环境即可：

```bash
python -m pip install -r tools/requirements-download.txt
python tools/prepare_resources.py --case cosmos_robocasa --dry-run
```

去掉 `--dry-run` 执行下载；可通过 `--source`、`--python` 和 case 专用路径参数生成启动配置，或使用case的 `paths.env.example` 绑定已有资源。默认位置如下：

| 内容 | 位置 |
| --- | --- |
| 模型和可选训练数据 | `~/.cache/robotics/hub/`，可用 `--root` 或 `ROBOTICS_RESOURCE_ROOT` 修改 |
| 生成的路径配置 | `~/.cache/robotics/env/<case>.env` |
| LIBERO资产 | `~/.cache/libero/assets/` |
| RoboCasa厨房资产 | 对应fork的 `robocasa/models/assets/`，单独准备 |
| 实验输出 | 调用者显式指定的 `--output-dir` |

以准备好资源的 RoboCasa 为例：

```bash
source ~/.cache/robotics/env/cosmos_robocasa.env
bash benchmarks/static/cosmos_robocasa/run.sh \
  --task TurnOffMicrowave --episodes 1 \
  --schedule paper_async --overlap-actions 2 \
  --output-dir runs/static/cosmos_robocasa/demo-001 --dry-run
```

预检通过后，去掉 `--dry-run` 并添加 `--gpu 3`，其中GPU编号按机器选择；添加 `--record-video` 录制完整动作。每次真实实验必须选择新的输出目录。实验入口使用离线模式，缺少资源时明确报错。

完整命令：[π0.5＋LIBERO](benchmarks/static/pi05_libero/README.md) · [Cosmos＋LIBERO](benchmarks/static/cosmos_libero/README.md) · [Cosmos＋RoboCasa](benchmarks/static/cosmos_robocasa/README.md)

## 实验协议与输出

`paper_async` 使用控制步 `t−n′` 的历史观测：图像和 proprio 来自同一快照，历史不足使用当前观测。这是论文静态异步抽象；当前 runner 串行推进环境。推理时间、仿真时间和宿主墙钟分别记录，完整口径见 [仿真协议](docs/protocols/simulation.md)。

每次运行保存配置与来源、checkpoint加载审计、逐episode记录和汇总；Cosmos入口还保存逐请求历史偏移及自动 `run.log`。π0.5控制台日志按需重定向或用 `tee` 保存。视频默认关闭，开启后含初始帧、全部执行动作及终止帧；RoboCasa保存三视角横排视频。

实验结束自动输出成功率与两种步数统计：全体episode中失败按声明预算计入，以及仅成功episode的平均步数。可以独立重新统计已有实验：

```bash
python tools/summarize_experiment.py \
  --input runs/static/cosmos_robocasa/demo-001 --per-task
```

RoboCasa的 `--reference-run` 在首次推理前核对baseline初始化；不一致时运行失败，不计为策略失败。详情见 [case说明](benchmarks/static/cosmos_robocasa/README.md)。

## 代码结构

```text
benchmarks/static/             四个静态case的CLI、配置与生命周期
benchmarks/dynamic/            KINETIX原生动态case；其他后端按独立任务扩展
benchmarks/inference/          固定输入计时、数值比较与逐轮可视化
src/robotics_bench/engines/     Cosmos模型加载、预处理与动作块推理
src/robotics_bench/models/      本仓维护的PI0.5和Cosmos执行代码
src/robotics_bench/optimizations/ 独立开关、模型适配与精度选择
src/robotics_kernels/           BF16融合、INT算子与Blackwell FP源码
src/robotics_bench/simulators/  LIBERO和RoboCasa原生环境适配
src/robotics_bench/protocols/   静态动作执行与历史观测选择
tools/                        资源准备、契约校验、统计和轨迹分析
schemas/                      通用数据契约；case运行格式另行声明
tests/                        CPU行为、数据契约与边界检查
docs/                         环境、架构及稳定协议
.github/                      Issue/PR模板、CPU CI和协作说明
```

[架构指南](docs/architecture.md) 说明实际调用关系、模型与环境边界、episode生命周期及后续扩展位置。

## 参与开发

使用主题分支和 PR；提交标题和 PR 使用中英文说明。PR写清目的、改动范围、实际验证、影响与回退，小改动可以各用一句话。默认通过 PR 的 merge commit 进入 `main`，由维护者完成审核和合并。详见 [CONTRIBUTING](CONTRIBUTING.md) 和 [AI协作约定](AGENTS.md)。

Git只保留源码、必要配置、测试、稳定文档及小型合成样例。模型、数据、视频、日志、临时笔记和生成图表留在本地，参见 [.gitignore](.gitignore)。GPU实验不进入公共CPU CI；远端分支保护的实际启用状态须单独核对。

## 参考与许可

方法背景：[The Speedup Paradox](https://arxiv.org/abs/2606.28529)。模型与模拟器的出处和独立条款见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。自有代码许可证尚未选定；当前未声明项目已完成正式开源发布。
## 算子优化入口

[推理优化实验](benchmarks/inference/README.md)提供本仓模型执行代码、独立融合开关、
BF16/INT4/INT8 对照、固定输入动作校验及逐轮 profile-visualizer 图表。
[算子说明](src/robotics_kernels/README.md)记录格式、构建与验证范围。FP4/FP8 为独立
保留的 Blackwell 源码，目标硬件验证另行进行。所有运行结果仍保存在被忽略的 `runs/`。

Cosmos 本轮固定 INT8/INT4、渐进覆盖与共享 VAE 优化已完成实现和性能验证。
[Cosmos 量化指南](docs/cosmos-quantization.md)汇总可运行配方、LIBERO/RoboCasa
复用边界、已有测量和回退方式；量化任务全集的质量仍待验证，所有开关默认关闭。
