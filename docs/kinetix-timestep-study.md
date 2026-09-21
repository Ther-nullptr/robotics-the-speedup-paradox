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
