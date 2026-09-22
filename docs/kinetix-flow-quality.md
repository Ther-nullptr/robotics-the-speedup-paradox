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
- 该子模块主要扫描控制方法、延迟和执行horizon，并介绍了async训练checkpoint。其`step`参数选择checkpoint，`num_flow_steps`才是采样迭代次数。本实验复用已有原RTC BC权重，固定horizon=4和零延迟，专门研究N，不声称复现其async checkpoint结果。
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

## 分析与绘图

```bash
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/analyze_flow_quality.py \
  --input runs/dynamic/kinetix/flow-quality-001
```

输出默认在实验目录的`analysis/`中：

| 文件 | 内容 |
| --- | --- |
| `analysis.json`、`report.md` | 完整性、统计设置、运行身份与英文报告 |
| `cell_metrics.csv` | 逐关卡、逐N的成功率、Wilson 95%区间和步数统计 |
| `macro_metrics.csv` | 固定任务集等权平均、区间及相对最高N的配对差异 |
| `paired_vs_reference.csv` | 按关卡比较各N与最高N，包含胜/负配对及检验结果 |
| `task_success_vs_flow_steps.png/.pdf` | 各关卡成功率曲线 |
| `macro_success_vs_flow_steps.png/.pdf` | 等权任务平均成功率曲线 |

分析只依赖NumPy，绘图额外需要Matplotlib；`--no-plots`可关闭绘图。`--allow-partial`仅查看已完成单元，不输出总体或配对结论。脚本拒绝重复/缺失回合、不同初态、非零延迟、改变的已检查物理参数、不完整运行身份以及混用代码/依赖版本。

同一seed在各任务和N之间共享派生随机流，因此bootstrap每次联合抽取整个seed块，保留跨任务和跨N的相关性。默认5000次、固定bootstrap seed；任务本身不重抽样，区间只描述这组固定任务的seed不确定性。每关的配对比较使用精确McNemar检验，对全部任务×非参考N的比较做Holm校正。未检出差异不等于已证明等价，也不假设质量随N单调增加。

## 已完成验证

原RTC BC `24` checkpoint、JAX/Flax FP32、上述固定设置下，完成12关×5个N×128个seed的7680回合。每个N均为1536回合，任务等权与按回合汇总在这次平衡设计中相同。

| N | 成功数 / 回合数 | 平均成功率 | 95% seed块bootstrap区间 | 全体预算惩罚平均步数 | 仅成功平均步数 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 1205 / 1536 | 78.45% | 77.08%–79.75% | 111.816 | 72.211 |
| 2 | 1317 / 1536 | 85.74% | 84.05%–87.37% | 103.357 | 77.974 |
| 3 | 1372 / 1536 | 89.32% | 87.89%–90.69% | 99.953 | 81.300 |
| 4 | 1382 / 1536 | 89.97% | 88.61%–91.28% | 98.029 | 80.425 |
| 5 | 1365 / 1536 | 88.87% | 87.37%–90.43% | 100.838 | 81.400 |

N=3、4相对N=5的配对差值区间分别为−1.50至+2.35、−0.72至+2.86个百分点，均包含0。因此目前只能说3～5次迭代的观测成功率接近，不能断言N=4最好或三者等价。提升主要集中在walker和trampoline；部分简单任务N=1已经接近饱和。48项任务级比较经Holm校正后，仅walker的N=1和trampoline的N=1、2显著低于N=5。

失败预算惩罚均值用成功实际步数与失败256步求和后除以全部回合；仅成功均值只使用成功回合。后者的样本组成随N变化，不能单独作为固定样本的任务速度比较。零注入延迟下的质量结果也不直接给出真实硬件latency–quality最优点。

## 评估加速的边界

加速评估单独测量冷启动、编译/预热和稳态rollout时间。不能把模型调用时间或kernel时间当作完整评估耗时。物理dt、frame_skip、模型/噪声RNG顺序、动作顺序、预算与逐控制步终止/非有限状态检查均需保持。

`flow_quality.py`已按任务并行，并在每关复用模型及不同N的编译结果。本轮三GPU总wall time约99.6分钟；没有配套单GPU全矩阵基线，不能据此声称三倍加速。历史多worker研究曾出现总吞吐提高、单回合却变慢的情况；任务调度吞吐与单回合延迟需分开测量。

独立工具`benchmark_eval.py`可测量原始batch=1回合，或在同一进程交替对照动作预处理JIT。该候选只编译原有motor/thruster绑定、clip和select操作，减少细碎dispatch；仍使用原生物理内核及逐控制步检查。它是benchmark内的显式实验选项，未改变`run.py`或本次主质量矩阵的执行路径。

```bash
# Check that the selected GPU is idle before timing.
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv

"$ROBOTICS_KINETIX_PYTHON" -B -u benchmarks/dynamic/kinetix/benchmark_eval.py \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" --gpu 0 \
  --level car_launch --flow-steps 5 --seeds 0,1,2,3 --repeats 3 \
  --compare-preprocess --output-dir runs/dynamic/kinetix/eval-speed-001
```

`benchmark.json`保存源码/关卡/checkpoint SHA256、Git commit及未提交状态、Python/依赖版本、GPU型号/UUID/驱动、初始化、独立预热、各回合时间分项以及精确轨迹。`reference.pstats`和`reference_profile.txt`是计时结束后的独立cProfile诊断，不混入性能样本。单独运行原始路径可省略`--compare-preprocess`；`--preprocess-jit`只运行候选，单路径数据不能自动产生已验证加速比。

配对比较先按repeat交替顺序计时，再在计时外逐seed记录完整状态hash、模型输出hash和语义事件。只有轨迹逐值一致且所有计时回合也匹配已验证结果，才生成`validated_speedup = reference rollout wall / candidate rollout wall`；否则为null。状态覆盖每个原生控制边界，事件比较只排除host policy耗时。计时范围包含reset、完整回合和每步检查，不含冷启动/JIT预热、文件日志、录像、trace采集和cProfile。

RTX 6000 Ada、JAX 0.4.35/jaxlib 0.4.34上的最终配对测试：

| 指标 | Reference | Action preprocessing JIT |
| --- | ---: | ---: |
| 12次稳态回合总wall time | 29.563 s | 26.424 s |
| 平均每回合 | 2.464 s | 2.202 s |
| 回合/秒 | 0.406 | 0.454 |
| 环境步进累计host span | 28.476 s | 25.295 s |
| 模型调用累计host span | 0.974 s | 1.003 s |

本轮加速比1.119×，三轮分别为1.097×、1.118×、1.141×；耗时减少约10.6%。12对计时结果一致，另外4对完整轨迹中，每种实现的392个控制状态、98次模型输出及全部语义事件一致。前一轮同设置测得1.132×，但缺少完整身份记录；最终数字采用补齐身份后的重新测量。

环境host span约占原始rollout的96%，其中包含GPU等待；cProfile中的`device_get`等待不能直接称为物理kernel计算时间。因此，仅降低模型推理耗时对本机整体评估收益有限。动作预处理JIT的收益来自环境调用路径；当前证据限于Car Launch、N=5、4个seed，不代表12关普遍收益、完整CLI冷启动收益或模型推理加速。

本地旧jaxlib也不接受循环/条件分支的command-buffer扩展选项，相关探测在编译参数阶段失败，没有性能结果；不把现代运行栈的选项直接套入旧环境。进一步研究可考虑保持每步RNG、done/finite检查的设备端回合驱动或跨环境批量化，但必须重新验证逐步状态和首次终止，并单独评估编译时间与显存。目前未启用这些改动，未修改物理时间片。GPU实验不加入公共CPU CI。
