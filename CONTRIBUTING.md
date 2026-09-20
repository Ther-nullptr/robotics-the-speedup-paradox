# 贡献指南 / Contributing

本仓采用主题分支 → 提交 → PR → 审核 → merge commit 的协作流程，沿用VLM项目的中英文提交与PR说明方式。任何进入 `main` 的代码、配置和文档变更都通过PR；维护者负责最终合并。

Use topic branches and pull requests for code, configuration and documentation changes. Commit titles and PR descriptions use Chinese and English. Maintainers review and merge through PR merge commits.

## 1. 明确范围 / Define the change

一个PR解决一个可独立审阅的问题，注明 `static/<case>`、`dynamic/<case>` 或 `shared`。小改动可以直接在PR中说明；较大功能、跨仓依赖或多人协作使用 [任务表单](.github/ISSUE_TEMPLATE/task.yml) 记录范围和验收，未确定的owner/reviewer写“待认领”。

Keep each PR focused on one reviewable problem and name its case or shared scope. Small changes can explain scope directly in the PR; larger work can use an Issue. Do not commit a separate planning document for every task.

## 2. 分支与提交 / Branches and commits

从最新 `origin/main` 创建主题分支，名称使用 `feat/`、`fix/`、`docs/`、`refactor/`、`perf/` 或 `chore/` 加简短主题。保留工作区已有改动，不重写他人的提交。多个独立任务使用独立分支或worktree。

```bash
git fetch origin
git switch -c docs/environment-guide origin/main
# Edit only the files in scope.
git diff --check
git add -- README.md docs/environment_setup.md
git commit -m "docs: 完善环境说明 / clarify environment setup"
git push -u origin HEAD
```

提交标题使用 `type: 中文摘要 / English summary`，可加scope，例如 `fix(robocasa): 修复初态读取 / fix initial-state loading`。一个提交应完成一个逻辑变化；不要求逐行或每次对话都创建提交。不批量改写历史提交以套用新格式。

Start from current `origin/main` unless a dependency branch is explicitly required. Use focused commits with bilingual titles. Review the diff before staging; list the intended files rather than staging unrelated work.

**有依赖的PR：** 如果本次改动依赖尚未合并的功能，声明前置分支/PR，并以它为当前PR的base。前置PR合并后，再把base调整到 `main` 并复核diff。不要把已有整套实验实现藏进一个标题为“文档更新”的PR。引入此制度前的本地功能提交可作为明确的前置PR提交审查，不需要重写历史。

For stacked work, explicitly name the prerequisite branch/PR and use it as the base. After the prerequisite merges, retarget to `main` and review the resulting diff. Existing local feature commits remain separate from a documentation-only review.

## 3. 创建PR / Open a pull request

使用 [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md)，四部分均提供中英文：

| 部分 / Section | 要说明的内容 / Content |
| --- | --- |
| 目的 / Purpose | 具体问题与改动后的行为 / Problem and resulting behavior |
| 范围 / Changes | 关键文件、职责、case及依赖 / Key files, responsibilities, case and dependencies |
| 验证 / Validation | 实际命令、结果、未验证范围 / Commands, observed results and unverified scope |
| 影响与回退 / Impact and rollback | 兼容性、数值或协议影响，以及回退方式 / Compatibility, numerical or protocol impact and rollback |

小改动每节每种语言一句话即可。PR标题采用与提交相同的中英文格式。尚未完成验收时创建Draft；准备好后提交review。避免把会话记录、模型输出或大型报告粘进正文，只保留审阅所需的事实和证据位置。

Use the four bilingual sections and a bilingual PR title. Keep small changes concise. Open a draft while work is incomplete; report checks that were not run explicitly.

可以使用GitHub网页，或在已登录的GitHub CLI中执行：

```bash
mkdir -p .local
cp -n .github/PULL_REQUEST_TEMPLATE.md .local/pr-body.md
# Fill in the actual change and validation results.
gh pr create --draft --base main \
  --title "docs: 完善环境说明 / clarify environment setup" \
  --body-file .local/pr-body.md
```

CI和后续审核针对PR中的实际commit；推送修复后更新验证说明。服务器认证不可用时，保留已推送分支、准确的base和PR正文，并明确说明PR尚未创建，不能将本地提交称为已提交PR。

## 4. 与改动相称的验证 / Proportionate validation

| 改动 / Change | 验证 / Validation |
| --- | --- |
| 文档、模板、图稿 | 本地链接、命令参数、Mermaid解析/渲染及 `git diff --check`；不启动GPU / Check links, commands, diagrams and whitespace |
| Python行为或数据契约 | 相关行为和失败边界检查，再执行下方CPU检查 / Focused behavior checks and the CPU suite |
| 模型/控制协议 | 相关CPU检查与实际GPU闭环；记录初态、预算、成功和异常 / CPU checks plus scoped real execution |
| 数值/性能优化 | 匹配的baseline、输入、精度和设备；正确性及完整入口测量 / Matched baseline, correctness and complete-entry measurements |

不为文字修改增加测试，不为低影响改动堆积实现镜像测试。已有检查通过后，仅在出现新改动、失败或明确风险时扩大验证。合成数据、单算子结果和单回合smoke各自标明范围。

For behavior changes, run from the repository root:

```bash
python -m ruff check tools tests benchmarks src
python -m ruff format --check tools tests benchmarks src
python tools/validate_contracts.py --examples
python -m pytest -q
python -m compileall -q tools benchmarks src
```

开发环境见 [环境指南](docs/environment_setup.md)。公共CI使用托管CPU runner，不下载模型，不接入实验室GPU。视频或性能采集带来的开销必须与测量口径一起说明。

## 5. 审核与合并 / Review and merge

reviewer核对目的、diff、兼容性和实际证据。本仓保留非作者审阅要求；审核角色须真实存在，不用AI自评代替他人批准。维护者在相关检查通过、讨论解决后，通过PR的 **Create a merge commit** 合并，保留提交与PR的关联；有理由采用其他合并策略时在PR中说明。

A reviewer other than the author checks scope, behavior, compatibility and evidence. Maintainers merge after relevant checks and discussions are resolved. The default is **Create a merge commit**. Explain any different merge strategy in the PR.

不直接提交或推送到 `main`，不通过本地merge后直接push绕开PR，也不自动启用auto-merge。用户已经明确授权某个PR合并时，代理可在规则满足后执行，不重复请求同一授权；普通开发任务不代表已授权合并。

本地规则与模板不能代替GitHub服务端限制。保护 `main`、要求 `CPU contracts` 状态及审查、禁止force push等设置由维护者在服务端核对，见 [.github说明](.github/README.md)。不能声称未检查的远端规则已启用。

## 6. 文件与依赖边界 / Tracked files and dependencies

Git保留代码、必要输入配置、测试、稳定说明和小型合成样例。checkpoint、数据集、下载缓存、视频、日志、profile输出、个人笔记和临时PR正文留在被忽略的本地目录。公开文档只能引用仓内已跟踪文件或可访问来源，不能依赖某人的家目录。

Track code, required inputs, tests, stable documentation and small synthetic examples. Keep models, datasets, generated outputs, scratch notes and draft PR text local. Do not commit credentials or machine-specific runtime paths.

需要修改模拟器内部代码时，在独立fork提交最小改动，记录其来源和许可；主仓PR固定可获取的版本并说明跨仓依赖。参见 [模拟器依赖协议](docs/protocols/simulator-dependencies.md) 和 [第三方来源](THIRD_PARTY_NOTICES.md)。本项目自有许可证仍待维护者决定，不把第三方许可证自动应用到整个仓库。
