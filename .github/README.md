# GitHub 协作配置

这些文件供独立公开仓库使用：

- `ISSUE_TEMPLATE/task.yml`：小任务与认领表单；完整任务卡保存在 `docs/tasks/`。
- `PULL_REQUEST_TEMPLATE.md`：行为变化、实际证据、兼容和交接。
- `CODEOWNERS`：初始 owner 为 `@Ther-nullptr`，生效前确认账号有写权限。
- `workflows/cpu.yml`：稳定 job 名称 `CPU contracts`，Python 3.11，GitHub 托管 CPU runner。

CI 触发事件为 `pull_request` 和 `main` 的 `push`，token 仅有 `contents: read`，checkout 不持久化凭据。当前没有 self-hosted/GPU job，没有模型下载，也没有远端写操作。

Actions 在 2026-09-14 依据官方仓库文档与 `git ls-remote` 的 v7 tag 核实，工作流固定完整提交：

| Action | 固定提交 |
| --- | --- |
| [actions/checkout](https://github.com/actions/checkout) | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| [actions/setup-python](https://github.com/actions/setup-python) | `5fda3b95a4ea91299a34e894583c3862153e4b97` |

更新 action 时重新核对官方提交及 runner 兼容性，保持完整 SHA 固定。[GitHub 安全建议](https://docs.github.com/en/actions/reference/security/secure-use)

本地定义存在不代表远端工作流已运行、Issue form 已展示或分支规则已生效。首次接入后由维护者检查运行结果，再启用必需状态与 code-owner review；设置步骤见 [CONTRIBUTING.md](../CONTRIBUTING.md)。本轮没有执行这些远端操作。
