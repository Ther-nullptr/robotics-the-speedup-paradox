# 模拟器源码修改、版本与协作

本协议适用于 LIBERO、RoboCasa、RoboTwin、Kinetix、DOM 等第三方仿真环境。KINETIX/Jax2D的运行源码已按MIT许可迁入仓内私有包，来源见对应PROVENANCE；其余模拟器沿用显式外部源码，下面的fork/submodule目录仍是接入约定，不表示已创建远端fork。各后端的引擎、环境隔离与执行方式见 [多模拟器后端协议](simulator-backends.md)。

建议先使用公开 API、adapter 或受支持的扩展；确实需要持续修改且许可允许分发的源码时，维护独立 fork，主仓以 submodule 固定提交。SDK、引擎与大型资产按其支持的方式独立安装。调度、实验和展示仍在主仓维护。

## 先判断改哪一层

| 改动 | 默认位置 | 何时需要改第三方源码 |
| --- | --- | --- |
| 同步/异步推理、请求触发、动作队列、延迟采样 | `src/robotics_bench/protocols/` | 现有 API 无法在等待中正确推进世界时，增加最小环境钩子 |
| 输入输出封装、状态快照、单位、trace | `src/robotics_bench/simulators/libero/` 等 adapter | 缺少必要状态/事件接口时，在对应依赖中增加接口 |
| 控制频率、相机、horizon 等已有参数 | 主仓 simulator config，通过 API 传入 | 上游没有所需配置能力时 |
| 新场景、对象、任务定义或物体运动 | 主仓任务/资产扩展目录，使用上游注册机制 | 注册机制不能表达时才修改 simulator；标明新任务版本 |
| physics 子步、controller 更新、观测刷新、底层终止逻辑 | 真正拥有这段实现的依赖 fork | 可能属于 LIBERO、robosuite 或更低层，不能预设都在 LIBERO |
| 可视化、录像、时间轴、结果聚合 | 主仓 viewer/trace/experiments | 通常只需公开采帧或状态接口，不把渲染面板写进 simulator |

LIBERO 的 wrapper 传入 control_freq，并将 step 委托内部环境；其 BDDL 环境调用父类 step 后又处理任务成功状态。控制/物理子步的部分实现位于 robosuite。直接绕过 environment 调 `sim.step()`，可能遗漏 controller、观测刷新、reward 或 terminal 更新，须逐项核对。[LIBERO wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py#L27-L88)、[BDDL step](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/bddl_base_domain.py#L800-L819)、[robosuite step](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/environments/base.py#L171-L398)

例如“推理期间目标继续运动”：runner 决定等待多久及期间采用什么控制；adapter 将持续时间映射到 simulator 支持的步进接口；模拟器负责动力学积分、观测和终止一致性。只有现有接口不能做到时，才在 fork 中新增清晰的步进能力。模型名称、GPU profiler 和 latency profile 不进入该底层接口。

## 引入方式

| 方式 | 适用情况 | 维护代价 |
| --- | --- | --- |
| 原版固定版本 + adapter/扩展 | 公开 API 已足够 | 首选；不承担下游源码维护 |
| **独立 fork + 固定 SHA submodule** | 持续修改内部接口、多人协作、计划向上游贡献 | 两仓关联 PR；由维护者负责版本联动 |
| 固定 upstream + 补丁目录 | 少量、稳定且容易重放的修改 | 检查应用顺序与基础 SHA；补丁频繁冲突时转 fork |
| 直接 vendor/subtree | 明确要求单仓离线分发，且团队愿意维护来源同步 | 主仓有大量第三方 diff；需记录上游基线和更新过程 |

fork 与主仓保留独立历史，方便核查差异和回收上游修复。submodule 在主仓保存指定提交，普通 update 按这个提交检出；环境安装/实验运行不使用 `--remote` 自动追踪最新分支。[Git submodule 文档](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

不要同时手改 submodule 再叠加另一套未记录的补丁。也不要靠安装脚本 sed 修改 site-packages 或全局 monkey patch 承载正式实验语义，否则无法可靠定位真正运行的源码。

## 建议目录与版本来源

```text
robotics/
├── third_party/
│   ├── libero/                       # 需要源码修改时加入的 submodule
│   └── simulators.lock.json           # 上游基线、依赖及环境元数据
├── src/robotics_bench/
│   ├── simulators/libero/             # 我们维护的 adapter/扩展
│   └── protocols/                     # 推理与动作调度
├── configs/simulators/                # 原版/扩展协议分别命名
├── tests/integration/simulators/      # 接口、步进、reset、终止验收
└── docs/third_party/libero.md          # 改动目录、原因、上游/下游 PR
```

实际 fork 提交以主仓 gitlink 为准；lock 文件记录 upstream URL/base SHA、robosuite/MuJoCo 版本、环境约束和资产 hash，避免独立维护两个可能冲突的 fork SHA。实验运行时导出实际 gitlink/工作树 dirty 状态、import 来源及依赖版本到 artifact；正式发布不依赖未提交的本地修改。

当前 v1 manifest 尚未包含完整 simulator provenance。添加上述运行元数据时先走 [契约演进](artifacts.md)，不能将未定义字段直接塞进严格 schema。

开源 `.gitmodules` 使用所有读者可访问的公开 HTTPS 地址，开发者可在本机另设推送凭据。普通 CPU 契约任务仍不需要初始化模拟器；只有 simulator 任务运行相应依赖初始化/安装。大型数据和权重不作为 submodule 内容引入。

## 一次模拟器修改如何提交

1. 在主项目任务卡说明触发场景、为什么公开 API 不够、拟改层及验收；关联模拟器 fork 的任务/PR。
2. 在 fork 建 feature 分支，提交最小环境修改和对应测试。主仓可同时准备 Draft PR，提前 review adapter 与配置变更。
3. 模拟器 PR 在团队 fork 审查并合并后，确定新的可获取 SHA；主仓 PR 再更新 gitlink、adapter、环境锁定信息和集成测试。
4. 主仓 reviewer 必须查看模拟器 diff 或关联 PR，不能只批准一个 opaque SHA。两边验收通过后合并主仓。
5. 通用改进可另向原上游提 PR；主仓无需等待上游接受，但应保留后续去掉下游修改的路径。

主仓引用的提交必须在公开 fork 可获取且有持久分支/tag 保留，不能是学生机器上未推送的 commit，或仅存在于可能被清理的临时 PR ref。学生修改 submodule 前应创建分支，避免在 detached HEAD 上留下不可追踪提交。

CODEOWNERS 分两处配置：主仓审查 adapter 与 gitlink，fork 审查 simulator 源码。主仓的 owner 规则不自动覆盖 fork 内部文件。初期由一位维护者负责模拟器集成，学生按模块贡献，不要求每位新成员管理跨仓发布。

## 验收与实验标识

为修改前的受支持路径保留兼容测试：固定 seed/state/action 序列，比较 step 数、观测、终止及数值容差。新 hook 默认关闭时应保持原行为；新的动态协议单独测试时间推进、控制保持、末次动作/终止及 reset。

如果只是观测钩子或可选参数扩展，验证其关闭时不改变基线。若改变对象运动、成功判据、控制频率或物理参数，应给新 task/protocol ID，例如 `libero_latency_v1`，同时记录原任务来源；不能把修改后的结果无说明地标为官方 LIBERO 分数。

模拟器测试需要对应环境，可以不依赖大模型；必要的渲染/物理依赖单列。普通 CPU 契约 CI 继续保持轻量；修改 gitlink 或 simulator adapter 时，相关集成证据由指定 job/设备补齐。没有环境证据时标未验证，不扩大支持矩阵。

## 发布时一起交付什么

一次发布固定主仓 commit、模拟器 fork commit、上游基线、依赖/资产版本和测试配置。提供能够取回 submodule 的克隆/安装说明；不要假设主仓源码 ZIP 含有 submodule 源文件。如需离线包，另构建完整源码包并检查所有来源与许可。

保留实际引入文件的版权、许可证和修改记录；数据/资产另核对各自条款。发布说明提供模拟器改动摘要与关联 PR，使用户能选择原版基线或扩展协议。相关来源统一登记 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
