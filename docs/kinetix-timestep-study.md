# KINETIX物理时间片细分研究

结论：**减小物理步长可以是合理的数值积分研究，但不能直接作为“只提高延迟粒度、任务保持不变”的评估机制。** 本次同动作对照中，细分网格改变了关节/接触任务轨迹和一个终止时刻；只缩放碰撞Baumgarte参数没有恢复原生轨迹。另一方面，隔离自由落体的解析误差随步长缩小而下降，因此不能把所有轨迹差异都称为仿真错误或undefined behavior。

本研究在 `feat/kinetix-timestep-study` 分支进行，基于KINETIX接入提交 `b79c92f`。生产case的原生dt/frame_skip及native-blend默认规则没有修改。

## 方法

使用仓内KINETIX/Jax2D源码，固定三关 `mjc_walker`、`mjc_half_cheetah`、`mjc_swimmer`。这些是KINETIX中的任务名称，实际求解器是Jax2D，并非MuJoCo。

每关使用两种确定性开环输入：全零动作；四个motor通道依次为 `[0.5,-0.5,0.5,-0.5]`、两个thruster通道为0。没有policy、注入延迟或动作噪声，因此不存在量化误差、flow质量、推理时长或噪声频率的混杂。每组从同一关卡初态开始，观察前32条控制或首次原生终止。

| 细分倍数r | 物理dt/ms | frame_skip | 控制周期/ms | 每模拟秒的solver迭代数 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.6667 | 2 | 33.3333 | 600 |
| 2 | 8.3333 | 4 | 33.3333 | 1200 |
| 4 | 4.1667 | 8 | 33.3333 | 2400 |
| 40 | 0.416667 | 80 | 33.3333 | 24000 |

每物理步保持10次solver迭代，episode预算仍为256条控制，warm-start、joint/motor/friction参数和控制边界终止判据保持原值。r>1同时比较碰撞β固定0.2和β/r两种设置；后者保持β/dt=12。共 `3任务×2输入×(1+3×2)=42` 条轨迹。

这里的“控制边界终止”只固定返回done的时刻。旧矩阵仍在每个物理子步检测reward，再聚合到控制边界，因此细分同时增加了任务接触检测点。它不能单独隔离终止检测频率的影响。下面的真实细分视频试验进一步固定了原生reward采样时刻。

比较仅使用双方都已观察到的相同控制时刻，位置/速度/旋转指标只覆盖初态中活动且可移动的刚体。转角差按2π周期处理。相对原生的RMSE衡量benchmark轨迹偏移，不是相对物理真实解的误差。两个试验均在RTX 6000 Ada/JAX float32上执行；未用host耗时作性能结论。

## 相同动作下的任务轨迹

下表列出r=40、β/r设置。单位为原生world units；最后一列对应同一模拟时刻的速度RMSE。

| 任务 | 输入 | 共同控制前缀 | 位置RMSE | 速度RMSE |
| --- | --- | ---: | ---: | ---: |
| mjc_walker | zero | 32 | 0.0110843 | 0.2519425 |
| mjc_walker | fixed-motor | 32 | 0.0258653 | 0.0042431 |
| mjc_half_cheetah | zero | 32 | 0.0252490 | 0.0367760 |
| mjc_half_cheetah | fixed-motor | 9 | 0.0425961 | 0.9303584 |
| mjc_swimmer | zero | 32 | 0.0055700 | 约1.12e-8 |
| mjc_swimmer | fixed-motor | 32 | 0.0066342 | 0.0003737 |

`mjc_half_cheetah`的fixed-motor输入在原生设置下于第9条控制发生失败终止（0.3秒）；r=40下于第10条终止（约0.3333秒）。r=2/4均在第9条终止，但同一时刻的状态已经不同。这说明细分可能使事件更早或更晚，不能预设它只会降低质量。

上述r=40任务的β固定与β/r设置都没有恢复原生轨迹。某些配置中两种β策略给出相同结果，另一些配置有差异；这不能证明碰撞项永远无关，只表明它不是足以保证全系统等价的单一补偿量。

本次观察到的是固定输入下的轨迹偏移和终止时刻变化。记录的活动刚体运动量保持有限；这不足以推断所有内部量、所有关卡或所有历史异常都已排除。42条固定输入轨迹不是成功率统计，也没有验证完整policy在细分环境中的质量。

## 自由落体：细分本身可以改善积分精度

另建无地面、无墙、无关节、无thruster的单活动圆形刚体场景，显式断言隔离条件。初始y=5、vy=0、g=-9.81，积分至T=8/30秒，解析位置为4.6512。

| r | 位置绝对误差 |
| ---: | ---: |
| 1 | 0.02179975 |
| 2 | 0.01089973 |
| 4 | 0.00545043 |
| 40 | 0.00054425 |

Jax2D先更新速度再更新位置，此无接触条件下对应semi-implicit Euler；恒定重力的解析离散误差大小为 `|g|×T×dt/2`，与实测一致到float32舍入量级。这支持“更小dt可改善简单积分精度”，同时也说明原生轨迹不应被当作物理真值。

首轮研究脚本的自由落体辅助段误用了带默认地面/墙体的 `create_empty_sim`，该部分已明确排除；上表和绘图只使用显式禁用这些对象并断言隔离条件后的独立重跑。任务轨迹段不受这项修正影响。原始数据的排除项保留在 `analysis-exclusions.json`。

## 为什么只保持控制周期还不够

源码显示，重力、motor/thruster冲量及位置积分包含dt；这些是预期的时间积分因素。碰撞速度偏置含β/dt，因此β/r只保持这一个比值。关节位置校正、角度限制及固定关节校正使用不同系数；它们没有同一套全局dt缩放公式。

更小dt还会增加每模拟秒的solver迭代、warm-start与接触几何更新次数。状态裁剪与约束处理也按物理步重复执行。保持 `dt×frame_skip` 不会自动保持这些离散过程。相关实现在 [physics engine](../src/robotics_bench/kinetix/native/jax2d/engine.py)、[collision](../src/robotics_bench/kinetix/native/jax2d/collision.py) 和 [joint](../src/robotics_bench/kinetix/native/jax2d/joint.py)。本次没有逐项消融各机制的贡献，不能把全部偏移归因于某一个系数。

历史strict-substep代码还可能改变噪声采样和终止检查频率。本研究特意固定这两项，已经观察到偏移；若要解释某一份历史结果，还需另外核对其实际协议。单纯 `sleep`、控制步取整、物理细分和native-blend是不同机制，不能只用相同的“delay/ms”标签合并。

## 对当前项目的决定

1. 主实验保留原生物理网格，使用已声明近似边界的native-blend表示延迟；不同flow/量化方案共用同一协议。
2. 细分物理网格作为独立模拟器版本或数值收敛研究。需要同动作轨迹、终止时刻与配对policy质量验证后，才能讨论它是否适合替代原benchmark。
3. 不把细分后的质量变化计作推理优化收益，不把β/r当作“已等价”的充分条件。
4. native-blend也不是精确槽内积分。后续可以增加独立边界敏感性对照，尤其关注快速碰撞任务；当前结果不证明它在所有任务上的连续时间精度。

## 运行与产物

```bash
source .local/kinetix.env
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/research_timestep.py \
  --levels mjc_walker,mjc_half_cheetah,mjc_swimmer \
  --factors 1,2,4,40 --patterns zero,fixed-motor --controls 32 \
  --gpu 2 --no-freefall --output-dir runs/dynamic/kinetix/timestep-study-001

"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/research_timestep.py \
  --only-freefall --levels mjc_walker --factors 1,2,4,40 \
  --gpu 1 --output-dir runs/dynamic/kinetix/freefall-study-001

"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/plot_timestep_study.py \
  --matrix-run runs/dynamic/kinetix/timestep-study-001 \
  --freefall-run runs/dynamic/kinetix/freefall-study-001 \
  --output-dir runs/dynamic/kinetix/timestep-study-figures
```

原始轨迹为NPZ，逐组配置/结果为JSONL，汇总为CSV/JSON；绘图脚本生成PNG/SVG/PDF，并拒绝未验证隔离条件的自由落体数据。产物留在忽略目录。源码与本说明在研究分支管理，接入PR的默认环境没有改动。

## 真实物理细分与MP4对照

本轮实际把physics dt从1/60秒缩到1/2400秒，每条控制执行80次完整物理更新；重力、碰撞检测、约束求解和积分都在小时间片上执行。**完整KINETIX任务的一致性尚未通过**。没有使用原版状态投影、未来状态或复制动画帧来制造一致性。

以下已报告的视频结果使用30 Hz动作更新：`research_refinement.py`先从原生环境的本地RTC policy记录动作带（N=5，每4条控制重新预测），随后各变体使用同一动作带。seed=0、零动作噪声、无注入延迟，控制周期保持1/30秒。动作带在原版首次终止或观察上限处结束；细分版若尚未终止，明确记为`censored=true`和`action_tape_exhausted`，不计为256步超时。这是固定输入的模拟器研究，不是闭环policy成功率评估。

新的reward采样固定为每个原生1/60秒窗口的起点：r=40时取每条控制的第0、40个微步生成的接触manifold，仍沿用原版max-reward、any-terminal、最后一个被选中采样点的GoalR规则。所有物理接触仍在微步上更新，只固定任务判据的观察时刻。

| 变体 | 物理步长 | 电机更新 | 碰撞β |
| --- | --- | --- | --- |
| 原版 | H=1/60秒 | 原版离散冲量，60 Hz | 0.2 |
| direct | h=H/40 | 每h重新计算反馈，2400 Hz | 0.2 |
| collision | h | 每h重新计算反馈，2400 Hz | 0.2/40 |
| held_motor_collision | h | 每H计算冲量，均摊到40个微步 | 0.2/40 |
| impulse_motor_collision | h | 每H起点施加完整电机冲量，其余微步为0 | 0.2/40 |

电机冲量始终由该变体自己的当前状态计算。两种保留60 Hz电机更新的变体仍执行80次物理求解；它们是对原生离散驱动的兼容性研究，不是连续力矩积分精度的证明。研究分支给`PhysicsEngine.step`增加了可选`motor_impulses`参数，普通入口不传该参数，保持默认motor计算路径。

源码中的局部反馈解释了一个偏差来源：在孤立、无限位、小误差条件下，电机相对角速度误差的单步系数近似为`1 - 900 * motor_power * H * 0.1`。Walker的四个活动电机`motor_power=3`，该系数为−3.5；直接细分40次后约为`(1-4.5/40)^40=0.00844762`。因此，加密物理更新也改变了原版的离散速度伺服行为。

隔离电机GPU检查中，`impulse`在一个H窗口末的角速度和转角都与原版逐值相同；`held`角速度最大误差约4.66e−9，但转角误差约1.10e−4。这个局部验证没有恢复整个关节/接触系统的等价性：

| 任务 | 原版结果 | direct r=40 | impulse_motor_collision r=40 |
| --- | --- | --- | --- |
| car_launch | 第40步成功，1.3333秒 | 第35步失败，1.1667秒 | 至第40步未终止，右删失 |
| mjc_walker | 第111步失败，3.7秒 | 至第111步未终止，右删失 | 第48步失败，1.6秒 |

并排视频在各自共同观察的同一时刻比较位置RMSE：car_launch第35步，direct为1.533889、impulse为0.186729；Walker第48步，direct为0.621941、impulse为0.593984。它们都没有达到原生任务一致。不能拿各变体不同终止时刻的RMSE直接排序；每条原始记录的`position_rmse_at_common_end`仅对应它与原版这一对轨迹的共同末时刻。

早期额外候选还尝试了`1-(1-beta)^(1/r)`的约束松弛重标定，以及孤立重力Euler位移补偿；它们没有恢复完整任务一致。重力位移补偿在受支撑的接触物体上仍会增加向下位移，不能当作通用修复，默认候选列表不包含它。早期试验使用所有细分接触采样，原始产物保留对应源码快照，和新采样协议分开解释。

### 运行

先按[入口说明](../benchmarks/dynamic/kinetix/README.md)配置独立Python环境和本地checkpoint目录。研究工具仅访问显式提供的本地资源。

研究入口现在默认`--control-hz 10`：每个外层动作保持100 ms（3个原生tick），新policy每4次动作更新推理一次，名义频率2.5 Hz。原生和impulse变体的电机反馈仍为60 Hz；r=40的物理求解仍为2400 Hz。`--control-hz 15`对应每动作保持2个tick，`--control-hz 30`恢复下面已有视频的设置。仅接受与原生tick网格整齐对齐的频率，不做隐式取整。

终止检查、录像和预算仍使用30 Hz原生tick，所以一个动作保持期间也能终止；256 tick仍是8.5333秒。为兼容既有产物，`control_dt_seconds`、`observed_controls`和`terminal_control`继续表示原生tick及其计数，`step_unit=native_control_tick`明确其单位；新增`command_hz`、`command_updates`、`policy_inference_calls`及`command_update_native_ticks`区分命令更新和模型调用。研究入口的10 Hz默认值不修改正式`run.py`的原生协议。

新policy在10 Hz下会延长每个预测动作的执行时间，这是新的部署条件，不能用它直接替换上述30 Hz结果。若使用已有30 Hz动作带，则按原时间轴因果采样第0、3、6…项并保持到下次更新，输入轨迹的总时长不变；保存的有效动作带仍逐原生tick记录，供全部物理变体配对回放。细分变体本身不再次运行policy。

以下命令显式使用30 Hz以复查上表；后续10 Hz实验改为`--control-hz 10`或省略该参数，并使用新的输出目录。

```bash
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/research_refinement.py \
  --levels car_launch,mjc_walker --factors 1,40 \
  --variants direct,impulse_motor_collision --controls 256 --control-hz 30 \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" --record-video \
  --gpu 0 --output-dir runs/dynamic/kinetix/physical-refinement-001

"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/compare_refinement_videos.py \
  --run-dir runs/dynamic/kinetix/physical-refinement-001 \
  --variants direct,impulse_motor_collision

"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/validate_refinement.py \
  --gpu 0 --output-dir runs/dynamic/kinetix/refinement-validation-001
```

已有动作带可用`--action-tape-dir <previous-run>`代替`--policy-dir`，无需再次加载模型。原始MP4为500×500、30 FPS，每条控制一帧并包含初态；这是从实际模拟器状态采样，不是对轨迹插值。并排MP4默认1/3倍速，结束的面板保留终止帧并明确标记`[held]`；右删失显示`TAPE END: no terminal observed`。这类视频不展示全部2400 Hz微步，但底层确实执行了这些微步。

`validate_refinement.py`检查隔离电机、两关各4条控制的原版执行与提交`9add270`的冻结step逐值一致，并保存一条控制内的微步位置、关节位置、contacts和rewards。此前r=40验证记录了原版每条控制2个物理步、细分80个，记录路径与正常路径末状态逐值一致。当前检查采用下面的温和r=2候选，验证2/4个实际物理步以及相应求解迭代数。此检查不是完整轨迹接触一致性证明。研究源码和必要说明随分支管理，MP4、NPZ和运行记录均留在忽略的`runs/`目录。

微步trace中的`polygon_position`和`circle_position`是该微步结束状态，`contacts`/`rewards`来自微步起点生成的manifold；`joint_position`是solver缓存，不能当作微步结束状态重新计算的关节锚点。分析事件与轨迹时需保留这一区别。

## 温和细分：默认2倍，保留10 Hz动作更新

研究入口现在默认`--factors 1,2 --variants impulse_motor_collision`，即原生16.667 ms与细分8.333 ms对照；每条原生tick从2个物理步变为4个。外层命令仍为10 Hz、电机反馈60 Hz，终止检查和256 tick预算保持原时钟。4倍、40倍细分以及其他候选均需显式选择。这个默认值减少了细分强度，**不是原生等价性认证**。

本轮复用前述10 Hz录像的同一动作带和初态，没有重新采样policy。比较r=2/4/40时，统一裁到各关共同观察的前缀，再计算整段采样轨迹的位置RMSE：

| 任务 | 共同原生tick数 | r=2 | r=4 | r=40 |
| --- | ---: | ---: | ---: | ---: |
| car_launch | 58 | 0.250164 | 0.414750 | 0.499854 |
| mjc_walker | 256 | 0.035469 | 0.048821 | 0.055723 |

位置指标下降不代表所有指标改善。Car Launch的旋转RMSE在r=2时为1.575770 rad，r=40时为0.827596 rad；原版第71 tick失败，r=2第58 tick失败，r=40第63 tick失败，r=4至动作带末尾第71 tick未终止。Walker三种设置均到256 tick预算，这不证明自然任务终止时刻相同；r=2的最大单刚体位置偏移仍为0.294727 world units。完整任务一致性尚未达成。

另外做了两项独立消融，均未设为默认：

- `impulse_motor_collision_budget`：r=2每微步只做5轮约束求解，每个原生H窗口仍合计10轮，避免把约束迭代数翻倍。r必须整除原生迭代数10，否则拒绝，不做隐式取整。相同迭代数不代表相同计算量或动力学，因为碰撞检测、warm-start及积分仍更频繁。
- `impulse_motor_collision_joint_clock`：保持每微步10轮速度、角度/限位和接触求解，仅让关节直接位置校正保持60 Hz。r=2时每条原生tick的系数为`[0.7,0,0.7,0]`，每个H窗口重置；校正依据候选自身的约束误差，没有读取原版位置。

两项消融都没有稳定降低误差。相同58 tick前缀下，Car Launch的位置RMSE由普通r=2的0.250164变为budget候选的0.317113，虽然旋转RMSE减小；Walker位置RMSE由0.035469变为0.061688。joint-clock候选在Car Launch第55 tick失败，Walker整段位置RMSE为0.038686。它们仅保留作研究开关。

工具现在同时输出共同前缀RMSE、峰值坐标差和最大单刚体欧氏位移差。共同初态不参与这些整段指标，旋转差先按2π周期处理。这些指标来自30 Hz状态采样，不覆盖所有细分步或全部接触事件。视频叠加文字注明模拟时刻、细分倍数、求解迭代数和关节位置校正频率；详细指标保存在`comparison_videos.json`。

```bash
# Default: native physics versus 2x refinement, with 10 Hz commands.
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/research_refinement.py \
  --levels car_launch,mjc_walker --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" \
  --record-video --gpu 0 --output-dir runs/dynamic/kinetix/conservative-001

"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/compare_refinement_videos.py \
  --run-dir runs/dynamic/kinetix/conservative-001

# For a run explicitly containing factors 1,2,4, display all three side by side.
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/compare_refinement_videos.py \
  --run-dir runs/dynamic/kinetix/conservative-matrix-001 \
  --variants impulse_motor_collision --factors 2,4
```
