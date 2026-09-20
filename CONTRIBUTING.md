# 贡献与接力开发

从 [README](README.md) 配置环境，使用 [工作路线](docs/tasks/README.md) 选择方向。任务实验从 [静态入口](benchmarks/static/README.md) 或 [动态入口](benchmarks/dynamic/README.md) 选择case；底层工具任务标记 `shared`。当前工具可在CPU运行，参与开发不要求先复现论文。

## Issue、PR与交接

一个任务有一位owner和一位reviewer。**Issue记录范围、依赖、验收和状态；PR记录实现、验证与兼容变化。** 可复制 [任务模板](docs/templates/task.md) 填写Issue，不要求额外提交一份阶段任务卡。

Issue/PR注明 `static/<case>`、`dynamic/<case>` 或 `shared`。前两类分别维护任务配置、控制协议和baseline；shared任务列出受影响的case。不为新模型自动生成所有模拟器组合，也不为两条路径复制通用工具。

开始工作时确认分支、基线commit和已有修改；使用短分支，例如 `feat/issue-12-clock-case`。多人使用独立checkout/worktree。公共接口的变化先合并规范，再并行修改调用方。

把AGENTS、相关协议和Issue范围提供给AI。提交者检查diff，能解释关键行为及验证范围；AI不能代替作者或reviewer承担确认责任。一次PR解决一个可观察问题，避免混入无关格式化与重构。

完成或换人时，用 [交接模板](docs/templates/handoff.md) 在Issue/PR中留下事实、最后成功命令、当前问题及下一步。未完成也能交接；个人草稿、会话记录可放 `.local/` 或已有的本地笔记目录，不需要进入Git。

## 验证与合并

代码变更运行相应检查：

```bash
python -m ruff check tools tests benchmarks src
python -m ruff format --check tools tests benchmarks src
python tools/validate_contracts.py --examples
python -m pytest -q
python -m compileall -q tools benchmarks src
```

绘图测试需要可选Matplotlib依赖，未安装时跳过这部分，数值核心测试仍运行。纯文档修改核对链接与 `git diff --check`；不为拼写修改新增测试或运行GPU实验。图册脚本修改还需实际重建并检查输出。

使用 [PR模板](.github/PULL_REQUEST_TEMPLATE.md) 提交可审阅结果。行为变化提供输入/时间线、命令、环境、实际输出、相关失败边界和未验证条件。性能变更同时给正确性与完整policy延迟；闭环变更另报SR、时间域和失败预算。无实测的模型/设备保持未验证状态。

reviewer复现关键案例并核对接口语义；维护者在至少一位非作者审阅和相关检查通过后合并。接口或实验口径变化同步更新稳定协议，普通文档PR不等待GPU。

## 哪些文件进入Git

| 保留版本化 | 留在本地 |
| --- | --- |
| 代码、测试、依赖、schema、小型可公开样例 | 权重、数据集、运行日志、临时结果 |
| README、AGENTS、协作模板、工具使用与指标定义 | 会话记录、已完成handoff、阶段任务卡 |
| 稳定架构/接口协议、Mermaid源码、生成脚本与配置 | 研究综述、可重建SVG/HTML/PNG与生成清单 |

忽略规则按具体目录/产物设置，不使用全局 `*.md` 排除必要说明。`.gitignore` 不会自动移除已跟踪文件；停止跟踪时保留本地内容。公共README和稳定规范只链接仓内已跟踪文件或公开来源。

默认安装与CPU检查不得依赖父目录私有资料。可选模型试跑须列明外部源码、checkpoint和资产，由调用者显式提供路径；公开代码不预设某台机器的目录布局。引入外部源码前登记来源及条款，见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。公开PR的CI使用托管临时CPU runner；实验室GPU验证通过单独隔离流程处理。

## 维护者配置远端

首次CI成功后，可启用main必需PR、非作者批准、`CPU contracts`状态检查、解决review讨论和禁止force push。本地配置存在不代表远端规则已生效。

[CODEOWNERS](.github/CODEOWNERS) 当前使用 `@Ther-nullptr`；启用强制owner review前加入真实有写权限的共同维护者，避免唯一owner自己的PR无人可批准。远端设置和发布由维护者操作。
