# AI 协作约定

本仓库当前提供协作协议、数据 schema、合成样例与 CPU 契约校验。没有模型推理、量化 kernel 或模拟器实现；不要把模板、schema 校验或 synthetic 轨迹说成性能与闭环实验结果。

## 开始工作

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
- 公共文件和检查只能依赖本仓库。不得读取、引用或发布父目录的私有项目、评审资料、凭据、机器绝对路径或外部软链接。
- 未来通用核心不依赖机器人专用 adapter；Torch、CUDA 和模拟器不进入当前 CPU 契约工具的默认依赖。
- 区分 wall time、sim time、device duration、result release、boundary wait 和 observation age。
- 保留真实 backend、dtype 和 fallback 证据。fake quant、GPU enqueue 返回和单个 GEMM 数字不代表真实端到端加速。
- 不在公共 PR CI 中调用实验室 GPU runner 或下载模型权重。本轮开发无需 GPU。
- 使用已有风格，避免无关重构和全仓格式化。新增测试针对行为和失败边界；文字修订无需新测试。

## 验证与交接

从仓库根目录运行与变更有关的命令，提交前执行 CPU 检查：

```bash
python -m ruff check tools tests
python -m ruff format --check tools tests
python tools/validate_contracts.py --examples
python -m pytest -q
python -m compileall -q tools
```

开发依赖安装命令见 `CONTRIBUTING.md`。`compileall` 是语法检查，不是 formatter。报告实际运行的命令、结果及未验证条件；schema 校验通过不能证明实时性、数值精度或任务成功率。

完成或暂停时检查 diff，将需要共享的事实、证据和后续动作写入 Issue/PR 的交接摘要。个人过程记录可放 `.local/`、`docs/handoffs/` 或工作区研究目录，这些不进入 Git；不要求每次对话新增文档。代码、稳定协议、工具说明和可复现小样例随版本管理，研究综述、阶段记录和生成图表留本地。已提交文档不得链接被忽略的本地文件。合并和发布由维护者负责。
