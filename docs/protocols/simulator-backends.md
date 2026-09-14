# 多模拟器后端与运行环境

范围包含 LIBERO、RoboCasa、RoboTwin、Kinetix，以及 DynamicVLA 的 Dynamic Object Manipulation（DOM）benchmark。当前均为接入规划，没有安装或运行模拟器。它们跨越不同物理引擎和执行模型，共享实验与数据契约，各自保留运行时实现。

## 后端矩阵

| 后端 | 上游层次与修改归属 | 需要固定和验收的差异 |
| --- | --- | --- |
| LIBERO | task/wrapper 在 LIBERO，控制及物理步进部分位于 robosuite/MuJoCo | init state、控制器、成功/超时、camera/control/physics dt |
| RoboCasa | 厨房任务/场景/对象在 RoboCasa；控制与底层积分在 robosuite/MuJoCo | 原版或365版本、场景/资产、horizon、controller 和依赖组合 |
| RoboTwin | 采用选定上游版本的任务与环境 adapter | 场景/机器人资产、动作/坐标约定、reset 与该版本实际依赖；不因接口相似假定与上述栈通用 |
| Kinetix | 任务/观测/奖励在 Kinetix，物理引擎为 Jax2D，运行于 JAX | state/PRNG/static params、frame_skip、JIT、batch shape、auto-reset、渲染与device同步 |
| DOM | DynamicVLA 仓库中的仿真配置/评估代码，基于 Isaac Lab/Isaac Sim | USD场景/对象、physics/control/camera节拍、动作接收与保持、并行env索引及episode边界 |

RoboCasa 当前官方项目区分原版和365，版本更新还可能改变任务预算；不能只记录一个 `robocasa` 名称就视为相同 benchmark。[官方版本说明](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/README.md#L18-L48)、[依赖定义](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/setup.py#L17-L44)

Kinetix 的公开接口显式传入随机 key、state 和参数，step 含 JIT/多物理帧推进及 reset 逻辑。一个 control step 不等于一个 physics step，也不等于一次独立 host/GPU 同步。[Kinetix step](https://github.com/FLAIROx/Kinetix/blob/67f89f45e32cfd7899bc63e252a6b044eacafb83/kinetix/environment/env.py#L65-L163)、[官方使用示例](https://github.com/FLAIROx/Kinetix#-basic-usage)

DOM 官方推荐分别准备模型和 Isaac Lab 仿真环境，评估入口在 `simulations/evaluate.py`；其场景/对象资产单独下载。因此应将 DOM benchmark adapter、Isaac 运行环境和模型服务分别管理。[DynamicVLA 安装与评估](https://github.com/hzxie/DynamicVLA/blob/bb702465fe3976ac853bc1fbed600668045909f2/README.md)

## 公共接口统一语义，后端声明能力

每个 adapter 对外说明 reset、动作应用、观测采样、推进时间、terminal 和帧引用的含义，输出相同版本的事件。能力声明至少回答：

- 谁拥有步进时钟：runner 手动推进、后端自身循环，还是编译后的 rollout？
- 推理等待期间世界如何演化，机器人使用什么控制？
- 是否支持批量环境、状态保存/恢复、独立采帧和外部事件注入？
- 动作/观测是否在设备上异步就绪，谁负责同步与输入输出生命周期？
- 是否自动 reset，怎样保留 terminal observation 并产生新的 episode ID？

能力声明必须对应实现和验收，不支持的功能显式拒绝。状态快照由后端拥有，不要求将 MuJoCo、JAX PyTree 和 Isaac 场景状态转换成同一种对象；公共层只保存版本化引用及恢复条件。只保存 RGB 图像不足以支持状态恢复。

接口语义可以通过本地调用或进程间消息实现，无需现在建立庞大的统一继承树。批量/编译接口可一次推进多个受控步骤，但要记录实际步数、终止和采样时刻；不能牺牲时间可解释性。

## 三类执行方式

**LIBERO/RoboCasa 等环境：** 优先用 adapter 与受支持参数控制步进。修改 task/scene 与修改 physics/control 分开提交；同 seed/action 的兼容检查限定在明确的软件/资产版本内，不承诺跨引擎逐位一致。

**Kinetix：** 在 JAX 路径保留函数式 state、PRNG 和兼容 shape，延迟可以表达成 tick 或编译循环中的状态。不要求每个 physics step 回 Python 发 JSON/等待 RPC，否则会改变原本批量执行的性能。trace 可先写设备数组再批量导出，记录原始事件发生时间；compile/warmup、设备执行、同步、像素观测和展示分别测量。评估时关闭 auto-reset 或捕获 reset 前 terminal state，不能把下一 episode 的观测归给上一条动作。

**DOM：** 原评估循环会非阻塞读取最新动作，无新动作时继续用上次动作或初始姿态推进环境。接入前保留并声明这一调度语义，随后通过独立协议再改变它；不能在外面叠一层未说明的等待/丢动作策略。接收“最新动作”也不能替代检查其来源观测和episode。运行慢于控制周期时要记录超期与实际推进，不把限频 sleep 当作物理时间追赶证明。[DOM 评估循环](https://github.com/hzxie/DynamicVLA/blob/bb702465fe3976ac853bc1fbed600668045909f2/simulations/evaluate.py#L270-L371)

## 独立环境与目录

以下目录按首个实际接入 PR 创建，不预先拉取所有仓库或资产：

```text
src/robotics_bench/simulators/
    libero/  robocasa/  robotwin/  kinetix/  dom/
configs/simulators/               # 每个后端的版本化配置
environments/
    libero/  robocasa/  kinetix/  dom/   # 各自依赖锁定/启动配方
third_party/
    libero/  robocasa/  kinetix/  dynamicvla/  # 仅必要的源码依赖
tests/integration/simulators/     # 按后端分开的验收
```

共享 robosuite/MuJoCo 的名称不代表版本兼容；LIBERO 和 RoboCasa 先分别锁定，验证后才能共用环境。Kinetix 保留自己的 JAX 栈。DOM 按其 Isaac SDK 要求启动，模型服务可在另一个 Python 环境；当前协议工具的 Python 3.11/3.12 不强加给仿真 SDK。

源码可直接修改并按其许可分发时，再采用 [fork/submodule 协议](simulator-dependencies.md)。引擎安装、插件和资产使用上游支持的分发方式，不要求所有 SDK 都 fork 或放进 Git。DynamicVLA 当前使用 S-Lab License 1.0，包含非商业使用条件，不能按其他 MIT 依赖一并处理；对应源码和资产分别记录来源条款。[上游许可证](https://github.com/hzxie/DynamicVLA/blob/bb702465fe3976ac853bc1fbed600668045909f2/LICENSE)

大模型推理与仿真/渲染可能竞争同一设备。分开记录 policy-only profile、simulator-only profile 和共部署结果；这些是不同测量范围。隔离进程不会消除 GPU 资源竞争，必须在结果中保留实际共部署条件。

## 统一验收与后端专项验收

公共要求是明确单位/时间域、来源ID、延迟期间动力学、终止与重置、失败预算以及媒体时间。后端专项测试分别覆盖：RoboCasa 资产/任务与 controller；Kinetix frame_skip/JIT/PRNG/auto-reset；DOM 动作保持、通信、camera更新和独立运行进程。普通 CPU 协议 CI 不安装它们。

每个后端/协议组合单独标 planned、implemented-unverified、verified。支持某个后端的静态场景，不代表其全部动态协议或渲染模式都已验证。

当前 trace v1 只支持单 run 中串行 episodes。Kinetix/DOM 批量环境先按环境导出独立 run；真正需要联合时间线时，再扩展 env/agent ID 与时钟协议。不得仅增加多个 episode ID 就宣称已有并行语义。后端 capability、环境锁定和媒体扩展同样应先走 [契约演进](artifacts.md)。
