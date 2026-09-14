# AI 协作约定

本仓库当前提供协作协议、数据 schema、合成样例与 CPU 契约校验。没有模型推理、量化 kernel 或模拟器实现；不要把模板、schema 校验或 synthetic 轨迹说成性能与闭环实验结果。

## 开始工作

1. 阅读 `README.md`、`CONTRIBUTING.md`、当前任务卡及相关 `docs/protocols/` 文档。
2. 确认 `git status --short`、分支和基线 commit，保留其他人的修改。任务范围以任务卡和用户最新指令为准。
3. 先跑相关最小例子，再改变一个可观察行为。缺少设备、资产或已知实验参数时明确记录，不自动编造。

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

完成或暂停时检查 diff，更新 `docs/handoffs/<task-id>.md` 的事实、证据与后续动作。保留必要错误和尝试结论，不提交整段 AI 聊天记录。合并和发布由维护者负责。
