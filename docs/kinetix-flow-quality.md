# KINETIX：迭代步数与任务成功率

本实验固定原生仿真和模型权重，只改变flow采样迭代次数。源码在仓内维护，外部路径仅用于已有checkpoint与Python框架环境。物理时间片研究不属于本实验。

## 实验设置

| 项目 | 固定设置 |
| --- | --- |
| 任务 | 12个仓内KINETIX关卡，每关绑定自己的原RTC BC checkpoint |
| 模型执行 | 仓内JAX/Flax flow policy，FP32，8条预测动作 |
| 迭代次数N | 1、2、3、4、5；只改变flow ODE的迭代次数 |
| 重规划 | 每执行4条控制指令重新推理，无初始prefetch |
| 物理 | dt=1/60秒，frame_skip=2，原生solver与碰撞参数 |
| 控制 | 原生30 Hz，最多256条控制；首次原生终止即结束 |
| 噪声 | 每控制步动作噪声标准差0.1 |
| 时延 | 注入时延为0，host运行耗时不推进仿真 |
| 配对 | 每格相同seed 0..127，共12×5×128=7680回合 |

准确率指首次回合的任务成功率，沿用原生`GoalR`/`returned_episode_solved`口径。环境、动作噪声和policy使用独立的确定性随机流。失败按256控制步预算进入总体步数统计；成功单独统计实际步数。不把回合异常记作策略失败。

## 参考来源与差异

- [VLASH sim/kinetix](https://github.com/mit-han-lab/vlash/tree/fa1e65c32b35109904de66e2404880a8bc1ed52b/benchmarks)固定子模块[vlash-kinetix aab1c8e](https://github.com/Sakits/vlash-kinetix/tree/aab1c8e0f6b3ab5f7498d030dafd99707ace4d7f)。其`EvalConfig`可设置`num_flow_steps`，默认5，默认2048个批量评估样本；只统计每个rollout的首次回合，支持多GPU分片。
- 该子模块主要扫描控制方法、延迟和执行horizon，并介绍了async训练checkpoint。本实验复用已有原RTC BC权重，固定horizon=4和零延迟，专门研究N，不声称复现其async checkpoint结果。
- 本地前期研究的可信原生网格矩阵使用128次评估；后期native-blend质量实验扫描N=1..5、零注入延迟、噪声0.1和horizon=4。本轮沿用这些控制条件，增加到128个配对seed。
- 本地历史结果包含Torch FP16及不同RNG/批量组织。本轮固定仓内JAX FP32，不与历史数值直接合并，也不强制拟合单调递增的质量曲线。

## 运行

按[环境说明](../benchmarks/dynamic/kinetix/README.md)准备本地路径变量。下载不是运行入口的一部分。

```bash
source .local/kinetix.env

# Preflight only; no simulation is started.
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/flow_quality.py \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" --gpus 1,2,3 \
  --output-dir runs/dynamic/kinetix/flow-quality-001 --dry-run

# Use currently idle GPUs. One persistent process evaluates all N/seeds of a task.
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/flow_quality.py \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" --gpus 1,2,3 \
  --output-dir runs/dynamic/kinetix/flow-quality-001
```

`--levels`、`--flow-steps`、`--episodes`、`--start-seed`可显式缩小范围。每次使用新输出目录。调度器只分配独立任务，不改变单回合执行；同一任务的全部N和seed由同一个常驻进程运行，以复用模型及JIT编译。主任务不录视频，减少无关开销。

## 产物

- `sweep-manifest.json`：预先声明的任务矩阵、完整命令、GPU分配、完成状态及总wall time。
- `logs/<task>.log`：每关运行进度与异常信息。
- `<task>/case-manifest.json`：代码、checkpoint、环境参数和运行依赖身份。
- `<task>/episodes.jsonl`：逐回合成功、实际控制步数、预算、seed与推理次数。
- `<task>/requests.jsonl`：动作块及控制事件记录。
- `<task>/summary.{csv,json,md}`：每个N的成功率和失败预算惩罚步数。

任务进程的非零退出或未完成覆盖使整个sweep标为失败；部分结果保留，不能写成完整实验。生成数据、日志、图和checkpoint不进入Git；运行工具、必要测试和本文档进入Git。

## 评估加速的边界

加速评估单独测量冷启动、编译/预热和稳态rollout时间。不能把模型调用时间或kernel时间当作完整评估耗时。物理dt、frame_skip、模型/噪声RNG顺序、动作顺序、预算与逐控制步终止/非有限状态检查均需保持。

当前优先评估任务级并行、编译复用及外层调度/数据传输。任何改变执行组织的候选，必须先与串行参考逐回合比较初态、动作、完整仿真状态、奖励、终止时刻和结果；没有通过等价验证的候选不进入主成功率矩阵。GPU实验使用独立运行环境，不加入公共CPU CI。
