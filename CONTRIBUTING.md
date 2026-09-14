# 贡献与接力开发

当前先搭好开发协议和数据契约。参与贡献不要求先复现论文，也不需要 GPU、模型权重或模拟器。模型推理与仿真运行时尚待实现，当前测试检查 schema 与部分事件因果关系。

## 第一次开始

使用 Python 3.11 或 3.12，在本仓库根目录创建环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m ruff check tools tests
python -m ruff format --check tools tests
python tools/validate_contracts.py --examples
python -m pytest -q
python -m compileall -q tools
```

如果使用 `uv`，也可以用它创建独立 Python 3.11 环境：

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

样例检查和测试应以退出码 0 完成；以实际输出为准，不把测试数量写成固定承诺。`compileall` 仅检查 Python 语法；Ruff 检查和格式检查由 CPU CI 强制执行。工具报错时保留最小命令及错误；先解决环境或已有失败，再解释本次改动。

从 [任务入口](docs/tasks/README.md) 选择一项工作。先找到一条合成事件及其关联 observation/request，再复现任务卡中的边界案例。不要用合成数据生成论文结果。

## 任务如何认领

一个任务有一位 owner 和一位 reviewer。GitHub Issue 管理认领、讨论和状态；`docs/tasks/<id>-<topic>.md` 是范围与验收的唯一完整记录。新任务使用 [任务卡模板](docs/templates/task.md)，已有任务在 Issue 中引用路径，不复制整份内容。

任务卡必须写清楚可观察问题、允许修改的文件、输入输出契约、依赖、验收与所需环境。没有设备的同学可完成契约、fixture 和接口工作；未实测的模型/设备不得标为 verified。范围扩大时更新任务卡，较大的独立工作拆成后续任务。

首次贡献可选 CPU 事件案例、校验器错误信息或 trace 显示。公共时间语义、数值门槛和并发资源规则由维护者或相应模块负责人专项审阅。

## 分支与 AI 使用

从当前 `main` 创建短分支，例如 `feat/010-clock-case` 或 `fix/001-error-message`。同一任务使用独立 checkout/worktree，避免多人在同一个目录切分支。依赖接口先合并，再并行开发调用方；共享分支不随意重写历史。

把 [AGENTS.md](AGENTS.md)、任务卡和相关协议交给 AI。提交者必须检查 diff，能解释关键输入输出、时间线与验证范围。AI 可以实现和测试，不能代替提交者确认结果或代替 reviewer 批准。不要把数千行生成结果、格式化和功能修改混成一个 PR。

## PR 的完成标准

使用 [PR 模板](.github/PULL_REQUEST_TEMPLATE.md)，提交 Draft PR 即可发起早期审阅。普通文字修订写清改动即可；会改变行为、公共 API、数值、性能或实验结论的 PR 要包含：

- 一个输入或时间线，说明修改前后的可观察差别。
- baseline/PR commit、精确命令、环境和实际结果；相关原始 artifact 使用可访问链接或仓内路径。
- 对应的失败边界或回归案例，以及未验证条件。
- 契约版本和兼容性变化；需要迁移时说明处理办法。
- 已更新的 [交接记录](docs/handoffs/README.md)。

性能改动日后需同时提供正确性与完整 policy 延迟；闭环改动另报 trials、SR、任务时钟和失败预算。当前只有 CPU 契约检查，不宣称上述能力已完成。实验路径可标为 implemented-unverified，只有实际证据支持时升级为 verified。

提交前执行上面的 CPU 检查，并查看 `git diff --check` 与 `git diff`。新增行为使用针对该行为的测试；拼写修改不要求新增测试或 GPU 实验。

reviewer 复现相关最小案例并核对语义；作者回应未解决意见。维护者确认范围、检查结果和至少一位非作者的审阅后合并，通常采用 squash merge。重要接口或时间协议变更需要明确指定懂该语义的人审阅；普通文档 PR 不等待 GPU。

## 交接与暂停

使用 [handoff 模板](docs/templates/handoff.md)，写入 `docs/handoffs/<task-id>.md`。保留当前分支/commit、已确认事实、最后成功命令、当前失败、运行作业及下一位的 1–3 个动作。不要求为了交接先完成任务，也不要把聊天记录当作工程状态。

## 公共边界

本仓库的安装、测试、模板与公共文档只使用仓内文件和公开依赖。不要提交私有项目源码、内部评审资料、密钥、未公开数据、权重或机器绝对路径。可共享的错误日志应移除这些内容。外部代码先登记来源与对应许可，见 [第三方来源记录](THIRD_PARTY_NOTICES.md)。

CPU CI 只在 GitHub 托管的临时 runner 执行，不使用实验室 GPU、私有挂载或仓库写权限。日后测试公开 fork 的 GPU 代码，应由维护者选定 commit 后进入隔离环境，不能自动在持有私有资料的常驻 runner 执行。[GitHub Actions 安全说明](https://docs.github.com/en/actions/reference/security/secure-use)

## 维护者启用远端规则

本地已提供 CI 和模板，尚未在远端运行或验证。建议在首次正常 CI 运行后启用 `main` 的必需 PR、至少一位非作者批准、必需状态检查 `CPU contracts`、解决审阅讨论、变更后重新审阅和禁止 force push。

[CODEOWNERS](.github/CODEOWNERS) 使用已知维护者 `@Ther-nullptr`。生效前需确认该账号对仓库有写权限，再在分支规则中启用 code-owner review；文件本身不会强制批准。启用强制 code-owner review 前，需先在 CODEOWNERS 增加至少一位真实有写权限的共同维护者；仅有当前一位 owner 时，他自己的 PR 无法由自己批准。维护者自己提交 PR 时仍需要另一位有资格的非作者 reviewer。模块 owner 待成员和权限落实后增加，不使用虚构账号。[GitHub CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)

本轮没有修改远端权限、保护规则或创建远端 Issue/PR。
