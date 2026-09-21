# AI 协作约定

本仓库提供协作协议、数据schema、CPU契约/分析工具，以及基于外部兼容源码的π0.5＋LIBERO和Cosmos＋LIBERO入口。Cosmos引擎、原生模拟器适配与单环境runner位于 `src/robotics_bench/`。静态异步采用论文抽象 `paper_async`；已迁入PI0.5的PaliGemma/Gemma/SigLIP执行代码、Cosmos的policy/sampler/DiT热路径，以及独立BF16融合和INT4/INT8算子；Blackwell FP4/FP8源码保留但尚待目标硬件验证。通用框架、Cosmos VAE/attention依赖和模拟器仍为外部依赖。不要把模板、schema校验、synthetic轨迹或单次smoke说成完整性能/任务集结果。

Cosmos＋RoboCasa已提供单环境入口，当前验证限于TurnOffMicrowave固定场景的sync/paper_async GPU smoke。RoboCasa采用三相机、H=32、独立robosuite1.5.1环境，reset后才读取本回合指令；不要复用LIBERO的H=16或预先固定语言。`--reference-run` 校验相同场景初始化，训练数据与厨房资产分别准备。

`tools/prepare_resources.py` 提供独立的可选联网下载/本地复用入口，默认模型和数据集缓存位于 `~/.cache/robotics/hub`，LIBERO资产沿用 `~/.cache/libero/assets`。下载不进入公共CPU CI，运行入口继续离线；资源准备成功不等于新机运行环境已经安装。

## 开始工作

本仓执行 [贡献与PR制度](CONTRIBUTING.md)：代码、配置和文档都在主题分支开发，通过PR进入main，不直接提交/推送main。依赖未合并功能时明确声明前置分支/PR，并使用对应base；不将已有功能混入文档PR。提交与PR标题采用 `type: 中文摘要 / English summary`，PR正文按模板写双语目的、范围、验证、影响与回退，小改动保持简短。维护者审核后默认以merge commit合并。用户明确授权合并某个PR时，在规则满足后继续执行，不重复询问同一授权；普通开发不自动合并。

1. 阅读 `README.md`、`CONTRIBUTING.md`、当前 Issue/PR 的任务范围及相关 `docs/protocols/` 文档。
2. 确认 `git status --short`、分支和基线 commit，保留其他人的修改。任务范围以 Issue/PR 和用户最新指令为准。
3. 先跑相关最小例子，再改变一个可观察行为。缺少设备、资产或已知实验参数时明确记录，不自动编造。
4. 确认任务归属为 `static/<case>`、`dynamic/<case>` 或 `shared`；case入口在 `benchmarks/static/`、`benchmarks/dynamic/`。模型、权重、任务与环境按case绑定；新增adapter不意味着支持其他组合。静态/动态控制实现分别演进，共享底层工具。

## 轻量化方法参考

后续模型轻量化与推理基础设施工作优先参考外部skill `$edge-model-lightweight`，来源为 [edge-model-lightweight-skill](https://github.com/Ther-nullptr/edge-model-lightweight-skill)。按任务阅读其完整计时、profile、收益估算、算子/精度与流式质量指南；仓内 [edge-inference-optimization](.agents/skills/edge-inference-optimization/SKILL.md) 补充机器人case的具体约束。外部skill按访问权限独立安装，不是运行本仓CPU工具或CI的依赖。

- 保留静态/动态case绑定、Kinetix原生JAX路径和各自控制协议；方法参考不要求统一模型输入、动作空间或执行器。
- 原始实现、优化全精度和轻量化实现可分别比较；量化专项增量收益与主表相对固定case baseline的总收益分别命名，遵循 [加速比协议](docs/protocols/speedup-metrics.md)。
- skill内的估算、示例和外部实测摘要只作方法参考，不作为本仓性能证据；实际结果仍需本case、目标设备及声明时间域的验证。实验过程记录继续按本仓版本管理边界留在本地或Issue/PR。

## 修改边界

- 一个任务一个 owner；只改约定文件。公共接口、时间语义或数值门槛变化须说明影响，并更新对应文档和案例。
- 默认CPU检查和公共文档链接须在干净克隆中自包含。模型试跑可通过显式参数使用case声明的外部源码、checkpoint和资产；不得硬编码、自动扫描或发布父目录私有资料、凭据、本机路径和外部软链接。
- 未来通用核心不依赖机器人专用 adapter；Torch、CUDA 和模拟器不进入当前 CPU 契约工具的默认依赖。
- 区分 wall time、sim time、device duration、result release、boundary wait 和 observation age。
- 静态 `paper_async` 按论文历史观测 `t−n′` 与周期模型验收，`history_observation` 是其实现方式；真实并发是可选扩展。`paper_async` 仅为独立case manifest标签，不改变通用v1 schema中 `async` 的既有语义。state/warmup规则由case显式声明；论文模型的 `Tact` 不修改physics dt，缺少 `Tinf` 时不生成周期/加速比，host `eval_s` 不充当论文周期。
- 保留真实 backend、dtype 和 fallback 证据。fake quant、GPU enqueue 返回和单个 GEMM 数字不代表真实端到端加速。
- 不在公共PR CI中调用实验室GPU runner或下载模型权重。模型/GPU试跑使用独立case环境并记录来源、设备和验证范围；CPU检查不依赖这些外部资源。
- 使用已有风格，避免无关重构和全仓格式化。新增测试针对行为和失败边界；文字修订无需新测试。

## 验证与交接

按改动选择验证。纯文档/模板变更核对链接、命令、图稿和 `git diff --check`，不新增测试或运行GPU。Python行为或数据契约变更先做相关检查，提交前执行CPU检查：

```bash
python -m ruff check tools tests benchmarks src
python -m ruff format --check tools tests benchmarks src
python tools/validate_contracts.py --examples
python -m pytest -q
python -m compileall -q tools benchmarks src
```

开发依赖安装命令见 `CONTRIBUTING.md`。`compileall` 是语法检查，不是 formatter。报告实际运行的命令、结果及未验证条件；schema 校验通过不能证明实时性、数值精度或任务成功率。

完成或暂停时检查 diff，将需要共享的事实、证据和后续动作写入 Issue/PR 的交接摘要。个人过程记录可放 `.local/`、`docs/handoffs/` 或工作区研究目录，这些不进入 Git；不要求每次对话新增文档。代码、稳定协议、工具说明和可复现小样例随版本管理，研究综述、阶段记录和生成图表留本地。已提交文档不得链接被忽略的本地文件。合并和发布由维护者负责。

## 算子优化与可视化

推理优化入口见 `benchmarks/inference/README.md`。每项优化保留独立开关；默认不开启。每轮以实际无优化BF16锚点直接计算完整policy时延比；低精度另外对照共享优化相同的BF16。先验证实际加载的源码、算子命中及动作差异，再报告性能；单个输入的逐值一致不等于任务集成功率已验证。

每轮通过外部 `profile-visualizer` skill生成历史图和可审阅记录，包含变慢/数值失败的候选。skill只渲染实测数据，不启动GPU、不估造缺失比较。实验ledger、trace、输入和生成图留在 `runs/`；稳定代码、来源记录和命令说明进入Git。

INT实现参照本仓独立FP副本的loader/Linear/pack/prepare/forward_packed分层；不导入父目录VLM代码。格式、scale、zero-point、pack版本和实际硬件后端显式区分，不能把INT4称为FP4。

## LingBot＋RoboTwin

LingBot优先适配RoboTwin，入口为 `benchmarks/static/lingbot_robotwin/`。模型和仿真使用独立本地环境；支持sync和paper_async；禁止把WebSocket async命名当成论文抽象。paper_async的n′范围0..16，每个名义关键帧k使用max(0,k−n′)的完整快照，负索引重复初始帧；原始预测动作缓存条件保持原位。不得只延迟推理请求而让KV/VAE读取较新的观测。首次执行16条指令，后续完整块32条，每4条采样实际观测更新KV/VAE；终止/预算截断不更新未执行动作。`primitive_steps` 是RoboTwin接受的take_action指令数，不是250Hz物理tick。此入口默认复用显式提供的本地checkpoint、模拟器资产和环境；执行时不自动下载，输出写入调用者目录，不修改共享资源。
