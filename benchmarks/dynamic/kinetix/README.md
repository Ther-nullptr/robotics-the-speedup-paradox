# Kinetix 动态任务接入

Kinetix作为独立的原生JAX实验单元接入，绑定自己的policy、checkpoint、关卡、观测/动作空间和rollout。状态为planned：本仓尚未加载策略、运行JAX环境或实现时延注入。本入口明确接入边界，不提供占位运行命令。

源码审查固定为 [FLAIROx/Kinetix `67f89f4`](https://github.com/FLAIROx/Kinetix/tree/67f89f45e32cfd7899bc63e252a6b044eacafb83)。上游是JAX二维物理任务系统；模型和环境均不采用其他VLA case的默认接口。

## 原生边界

| 部分 | 保留的原生设计 | 本仓接入责任 |
| --- | --- | --- |
| Policy | 按配置构建Flax网络；参数PyTree、可选RNN carry、动作分布和value | 固定模型配置/checkpoint，声明采样或确定性选择，管理carry与PRNG；无需文本输入、去噪或action chunk |
| Observation / action | 像素、symbolic或entity观测；continuous、discrete或multi-discrete动作 | 固定观测/动作类型、实体容量及mask、motor/thruster绑定；不转换成机械臂7维动作 |
| Environment | 显式 `EnvState`、`EnvParams`、`StaticEnvParams`，Jax2D物理步进 | 保留PyTree和函数式状态，记录实际dt、frame_skip、关卡及reset规则 |
| Rollout | 原生 `jax.vmap` / `jax.lax.scan`、设备数组 | 在JAX状态中处理延迟和控制规则；按批次导出事件，不要求逐步host RPC或PyTorch进程worker |

源码依据：[模型构建](https://github.com/FLAIROx/Kinetix/blob/67f89f45e32cfd7899bc63e252a6b044eacafb83/kinetix/models/utils.py)、[actor-critic与carry](https://github.com/FLAIROx/Kinetix/blob/67f89f45e32cfd7899bc63e252a6b044eacafb83/kinetix/models/actor_critic.py)、[观测/动作空间](https://github.com/FLAIROx/Kinetix/blob/67f89f45e32cfd7899bc63e252a6b044eacafb83/kinetix/environment/spaces.py)、[环境步进](https://github.com/FLAIROx/Kinetix/blob/67f89f45e32cfd7899bc63e252a6b044eacafb83/kinetix/environment/env.py)、[评估rollout](https://github.com/FLAIROx/Kinetix/blob/67f89f45e32cfd7899bc63e252a6b044eacafb83/kinetix/util/learning.py)。

## 代码与配置归属

首次实现优先在本目录维护 `policy.py`、`environment.py`、`rollout.py`、`export.py`、`run.py` 与 `case.yaml`，一起形成可运行case。这些文件目前不存在；先保持Kinetix策略与环境的原生组合，出现实际复用需求后再抽取公共模块。训练算法和数据生成不属于首次接入。

配置必须绑定network结构、checkpoint/参数hash、关卡列表/hash、观测与动作类型、实体容量、recurrent设置、采样规则、JAX/Jax2D及设备版本。PRNG拆分为环境、policy采样和延迟随机流；batch大小与编译shape显式记录。具体checkpoint和关卡在首次运行前选定，不通过下载整个训练数据集建立入口。

公共层统一case/baseline标识、单位、时间域、事件关联和指标定义；Kinetix可按能力复用backend工具，但不强制复用Torch量化kernel、Future执行器、VLA processor或DOM时钟驱动。批量环境按 [现有trace限制](../../../docs/protocols/simulator-backends.md) 分环境导出run；当前schema尚不支持一个run中的并行episodes，不直接往v1塞入新字段。

## 时间与动作协议

首个延迟实验优先采用虚拟时间/profile重放：在rollout状态中显式携带采样tick、结果释放tick、待用动作、当前控制和episode边界。等待期间仍推进物理状态；需要亚控制步粒度时增加经过验证的子步协议，不把延迟粗略丢弃。控制步长由实际physics dt与frame_skip确定，渲染FPS不参与推导。

等待时重复旧控制与使用 `action_type.noop_action()` 是不同策略，按case声明。尤其离散空间的noop不保证为0；电机/推进器的持续施力也不等于DOM保持绝对末端目标。carry在实际policy请求时更新，等待或跳过推理的tick不得偷偷用新观测更新carry；reset按每个环境分别清理carry和未释放动作。

`vmap/scan`仅表示原生批量/编译执行，不证明模型与环境真实并发。profile重放结果与真实设备实测分开。设备计时以明确完成同步为边界（例如 `jax.block_until_ready`），分别报告编译、warmup、policy计算、完整rollout与渲染；不为每个模拟tick虚构独立墙钟计时。

## 首次验收

1. 固定关卡与checkpoint，核对动作分布/确定性输出、动作映射与carry reset。原生 [example_inference.py](https://github.com/FLAIROx/Kinetix/blob/67f89f45e32cfd7899bc63e252a6b044eacafb83/examples/example_inference.py) 会把采样动作覆盖为noop，不能直接用该示例宣称已完成策略闭环；须检查实际送入环境的动作。
2. 验证零延迟与注入延迟的状态时间线、等待控制、首次terminal和reset前状态。处理frame_skip内部终止以及auto-reset；短批量rollout也须区分每个环境的episode。
3. 对比同case的适用优化，记录return/score、按任务定义的成功率、任务时间、动作/观测年龄与失败。离散动作编号的差分不作为机器人jerk；若分析物体轨迹，先明确实体、坐标和物理时间。

Kinetix的baseline独立建立。真实低精度、异步或其他优化仅在原生backend已实现并验证后开放；不为填满2×2矩阵自动宣称支持。复杂rollout使用实际事件计算任务时间，固定chunk公式仅在假设成立时使用。返回 [动态任务入口](../README.md)。
