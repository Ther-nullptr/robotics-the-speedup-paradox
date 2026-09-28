# KINETIX：硬件延迟与任务成功率

本工作流在固定原生物理参数下，分别测量零延迟质量、固定采样步数的延迟响应，以及硬件延迟映射后的任务成功率。支持先冻结预测、再验证，输出与 rebuttal Figure 2 对应的四联图。代码在本仓维护，参考项目不参与模型或模拟器执行。

## 两种延迟模式

| CLI 名称 | 含义 | 历史别名 |
| --- | --- | --- |
| `coarse` | 粗粒度：将推理延迟取整到完整控制步 | `legacy-round` |
| `fine`（默认） | 细粒度：在边界物理步内混合新旧执行器指令 | `native-blend` |

两者均使用 `dt=1/60 s`、`frame_skip=2`，即物理步16.667 ms、控制步33.333 ms；预测8条动作、执行4条后刷新，名义周期133.333 ms。终止判定和预算仍遵守原生控制边界。细粒度延迟表示不会缩小物理积分步长。

```text
coarse: effective_delay = round(requested_delay / control_dt) * control_dt
fine:   effective_delay = requested_delay

old_weight[k] = clip(effective_delay / physics_dt - k, 0, 1)
command[k] = old_weight[k] * process(old_action)
           + (1 - old_weight[k]) * process(new_action)
```

`round` 使用 nearest-even。fine 的一个分数边界槽采用指令时间平均近似，不重建槽内部的碰撞或关节状态；零延迟与完整旧动作边界走原生执行路径。新旧候选使用同源的控制步噪声，无初始预取，旧队列从零动作开始。

旧别名继续可用；显式使用旧别名时保留旧记录标签，兼容已有扫描和分析。新工作流使用 `coarse`／`fine`，不重写历史原始记录。

## 实验范围与任务选择

默认选择既有零延迟扫描中随采样步数增加而总体改善的四个任务，允许高步数区间的小幅波动。选择依据是已有零延迟结果，不以新延迟扫描是否出现内部最优点筛选任务。

| 任务 | 已有 N=1 成功数／512 | 已有 N=5 成功数／512 | 选择依据 |
| --- | ---: | ---: | --- |
| `mjc_walker` | 42 | 379 | 对采样步数敏感；N=3高于N=5，保留非单调形状 |
| `catapult` | 278 | 362 | 明显质量差异，并在较高步数趋于平台 |
| `trampoline` | 80 | 407 | 强质量差异；检验时序敏感任务的可分解近似 |
| `mjc_half_cheetah` | 367 | 448 | 总体上升；N=4到5的轻微回落未显示明确配对差异 |

half-cheetah的N=4相对N=5差异为+1.17个百分点，配对95%区间为[-2.93,+5.27]；walker的N=3相对N=5差异为+3.71个百分点，区间为[-1.56,+8.79]。这些波动不构成严格单调或等价的证明；两关从N=1到5的整体增益仍明显。接近饱和且缺少上升趋势的catcher不属于默认主任务。

这些是本项目JAX FP32零延迟扫描的选题依据，不是本轮硬件映射结果。每个实验单元采用512个配对种子，默认0～511；种子控制策略与动作噪声，关卡初态仍固定，不代表512个独立场景。

| 阶段 | 配置 | 默认规模 |
| --- | --- | ---: |
| 对齐检查 | 四任务、两模式、N=1～5，各复查两个已有seed | 80回合，不计入正式统计 |
| 零延迟质量 G | 四任务、N=1～5、L=0 | 复用10,240回合 |
| 延迟校准 H | 固定N=5；fine为0～66.667 ms、间隔8.333 ms；coarse为0／33.333／66.667 ms | 新增16,384回合 |
| 硬件映射验证 | 四任务、四档硬件、N=1～5、两模式 | 新增47,104回合 |

默认共124个不同的新执行配置、63,488个新回合。逻辑配置之间只有在任务、N与实际float32指令权重完全相同时才复用结果，例如快速GPU在coarse下均映射到零延迟。每个逻辑单元保留来源引用，复用不会增加独立样本数，也不会将已用于校准的点计为留出验证。

硬件数据来自[四档延迟档案](kinetix-hardware-latencies.md)：RTX 6000 Ada、RTX 3090、AGX Orin 15W／30W。它们是历史PyTorch eager FP16、同架构随机权重的完整动作生成计时。注入当前已训练JAX FP32策略时，称为硬件延迟场景重放；当前宿主推理墙钟不额外推进模拟器。

## 环境与输入

模拟依赖见 [KINETIX入口](../benchmarks/dynamic/kinetix/README.md#环境与资源)。需要兼容Python 3.11／JAX运行环境、可信的原始RTC checkpoint和一个完整的匹配零延迟扫描。执行过程中不下载模型或数据。

```bash
export ROBOTICS_KINETIX_PYTHON=/path/to/kinetix-env/bin/python
export ROBOTICS_KINETIX_POLICY_DIR=/path/to/rtc-policies
```

编排器只使用Python标准库。拟合和画图使用独立CPU环境：

```bash
python3 -m venv .local/kinetix-analysis-env
.local/kinetix-analysis-env/bin/python -m pip install \
  -r benchmarks/dynamic/kinetix/requirements-analysis.txt
export KINETIX_ANALYSIS_PYTHON="$PWD/.local/kinetix-analysis-env/bin/python"
```

已有质量结果要求采用 `<quality-root>/<task>/case-manifest.json` 和 `episodes.jsonl` 布局。新环境可以按任务建立这组结果：

```bash
for task in mjc_walker catapult trampoline mjc_half_cheetah; do
  bash benchmarks/dynamic/kinetix/run.sh \
    --levels "$task" --flow-steps 1,2,3,4,5 --latencies-ms 0 \
    --mapping fine --episodes 512 --start-seed 0 --gpu 0 \
    --output-dir "runs/dynamic/kinetix/quality_512/$task"
done
```

按本机空闲设备调整GPU编号。控制器仅使用显式传入的卡；不会扩展到未指定的GPU。配对复用前会核对完整seed覆盖、源码、checkpoint、原生参数及运行栈，并执行零延迟回放检查。允许的源码迁移仅限已审查的模式名称变更。

## 运行工作流

以下命令从仓库根目录执行。输出目录必须是新的；恢复已有任务使用 `run`，不重新执行 `plan`。

```bash
export KINETIX_STUDY=runs/dynamic/kinetix/latency_quality_512_001

python benchmarks/dynamic/kinetix/latency_study.py plan \
  --quality-run runs/dynamic/kinetix/quality_512 \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" \
  --episodes 512 --start-seed 0 --output-dir "$KINETIX_STUDY"

python benchmarks/dynamic/kinetix/latency_study.py run \
  --study-dir "$KINETIX_STUDY" --phase verification \
  --python "$ROBOTICS_KINETIX_PYTHON" --gpus 0,1,2

python benchmarks/dynamic/kinetix/latency_study.py run \
  --study-dir "$KINETIX_STUDY" --phase calibration \
  --python "$ROBOTICS_KINETIX_PYTHON" --gpus 0,1,2

"$KINETIX_ANALYSIS_PYTHON" benchmarks/dynamic/kinetix/analyze_latency_study.py freeze \
  --study-dir "$KINETIX_STUDY"

python benchmarks/dynamic/kinetix/latency_study.py run \
  --study-dir "$KINETIX_STUDY" --phase validation \
  --python "$ROBOTICS_KINETIX_PYTHON" --gpus 0,1,2

"$KINETIX_ANALYSIS_PYTHON" benchmarks/dynamic/kinetix/analyze_latency_study.py report \
  --study-dir "$KINETIX_STUDY" --plots
```

也可用一个命令顺序完成检查、校准、冻结、验证和报告；任一阶段失败会停止后续阶段：

```bash
python benchmarks/dynamic/kinetix/latency_study.py finish \
  --study-dir "$KINETIX_STUDY" \
  --python "$ROBOTICS_KINETIX_PYTHON" \
  --analysis-python "$KINETIX_ANALYSIS_PYTHON" --gpus 0,1,2
```

默认每张卡运行一个独立进程，单个环境的步进逻辑不变。每个worker默认绑定16个可用CPU核，可用 `--cpus-per-worker` 调整。控制器等待指定GPU空闲后启动；长任务可在tmux或后台进程中运行。

`--workers-per-gpu`可显式提高每卡进程上限，例如在有96个可用逻辑CPU的机器上，用三张卡运行最多六个独立作业：

```bash
python benchmarks/dynamic/kinetix/latency_study.py finish \
  --study-dir "$KINETIX_STUDY" \
  --python "$ROBOTICS_KINETIX_PYTHON" \
  --analysis-python "$KINETIX_ANALYSIS_PYTHON" \
  --gpus 1,2,3 --workers-per-gpu 2 --cpus-per-worker 16
```

已有且身份匹配的worker计入上限，包括尚未初始化CUDA的进程；恢复时继续接管它们。仅与当前study的进程共享GPU，发现其他计算进程时等待。CPU集合在本study的运行worker之间互不重叠；总预留超过控制器CPU亲和范围时，启动前报错。模型及模拟器各加载一份，显存随进程数累加。上限与CPU分配进入状态记录，原有数据复用和coarse→fine完成边界保持不变。

这是评估吞吐开关。各进程仍使用相同模型、物理参数、seed与外部注入延迟，宿主墙钟不会推进模拟器。并发时的`host_policy_call_seconds`包含资源争用，不能作为独占硬件推理时延或回写延迟profile。吞吐收益须按具体任务测量；同时核对逐回合成功、终止原因和控制步数。原生GPU浮点运算也可能在串行重复运行中产生细微差异，因此保留原生源码不等于保证完整轨迹逐位一致。

验证阶段按 **全部 coarse → 全部 fine** 顺序执行。只有所有任务的coarse作业完成并通过结果校验后，才会派发fine作业；断点恢复时也会等待仍在运行的coarse worker。coarse失败会停止后续派发，已有完整结果继续保留。`status.json`及`status`命令中的`validation_order`和`validation_modes`记录顺序与各模式进度。

新计划将跨模式共享的验证配置归入先执行的coarse作业。如果旧自定义硬件计划中的coarse条件依赖fine作业，验证入口会要求新建计划，不修改原有`study.json`。四档默认硬件的既有计划可直接恢复。

每种模式都覆盖四个任务的四档硬件×N=1～5，即80个逻辑配置，每配置512个配对seed。coarse复用零延迟和校准重叠点后，需要新增12个不同配置、6,144回合；fine新增80个配置、40,960回合。最终表格仍按每个任务、硬件与N展开，保留预测最佳N、实测最佳整数集合、置信区间及复用来源，用来比较两种映射下最优点的位置。

查询进度无需加载模型环境：

```bash
python benchmarks/dynamic/kinetix/latency_study.py status --study-dir "$KINETIX_STUDY"
```

恢复时重新执行对应的 `run --phase ...` 命令。完整且通过原有哈希证据复核的job直接复用；控制器退出后仍存活的worker按PID和完整输出路径接管。已停止但未完成的job保留旧attempt，在新attempt中重新运行整个job。运行异常不作为策略失败加入统计。锁文件防止同一study同时启动两个控制器。冻结模型不得在验证开始后覆盖；关卡文件与原生参数在每个阶段重新核对固定基线。

调整任务集合时，建立新的study，并通过 `run` 或 `finish` 的 `--reuse-study /path/to/previous-study` 显式引用匹配的已有job。工具核对源码、checkpoint、物理参数、运行栈与配置；已完成job复核覆盖和哈希，仍在运行的job等待完整结束后才纳入。来源目录须保留，部分回合不会冒充完整配置。

## 输出与版本管理

| 路径 | 内容 |
| --- | --- |
| `study.json` | 固定任务、配置、种子、硬件档案、执行来源和源码指纹 |
| `profiles/*.csv` | 各硬件的完整精度延迟输入 |
| `status.json` | 阶段、job、GPU、PID、输出位置及完成验证 |
| `executions/<task>/<job>/attempt-*/` | case manifest、逐回合记录、事件、日志和统计 |
| `calibration/frozen-model.json` | 验证前固定的G/H模型与所有映射条件的预测 |
| `calibration/cells.csv`、`report.md` | 固定N延迟扫描和拟合诊断 |
| `analysis/cells.csv` | 各逻辑单元成功率、区间、步数和复用来源 |
| `analysis/validation.csv` | 预测与实测最佳整数N、留出RMSE及选点损失 |
| `analysis/*-four-panel.png/.pdf` | 每任务、每模式的四联图 |
| `analysis/report.md`、`analysis.json` | 完整结果与可审计来源 |

所有运行产物留在被忽略的 `runs/` 下。Git保留代码、历史小型输入档案、测试和稳定说明；checkpoint、事件日志、工作区配置、实验图片及中间输出不提交。

## 拟合、验证与统计口径

四联图对应：硬件延迟、零延迟质量、N=5延迟响应、硬件映射预测与实测。组合关系为：

```text
predicted_success(N, hardware, mode)
    = G(N) * H_mode(effective_latency_mode(L_hardware(N)))
```

G保留零延迟实测概率，绘图时采用PCHIP形状保持插值，不强制单调。fine下的H使用二项似然比较Hill曲线与无延迟效应模型；coarse下按实际有效延迟网格查表。粗粒度预测遵循取整台阶，不将同一延迟档位误画成连续的动力学变化。

所有模型只读取零延迟质量与固定N的校准数据。整数预测和图中曲线采样点一并冻结，重绘直接读取固定数值；之后才进入硬件映射验证。校准与验证使用相同配对seed，属于实验条件留出，不能称为新seed或新场景泛化验证。拟合不佳或出现明显向上反转时保留诊断，不能为获得某种曲线而改物理参数、删除任务或挑seed。

主要决策直接枚举整数N=1～5。实测最佳整数集合保留所有并列点，不代表统计显著优于其他点；预测并列时选择较小N。成功率提供95% Wilson区间；选点损失使用配对seed bootstrap，条件为已经冻结的预测，不包含拟合与硬件profile的不确定性。

步数仅提供全体失败按预算惩罚的口径，以及仅成功回合口径。失败回合统一按256控制步计入总体；仅成功均值以成功回合数为分母。所有模拟回合使用完整终止／预算语义。

## English summary

The `coarse` and `fine` modes share the original physics grid. This workflow verifies a complete zero-delay baseline, calibrates delay response, freezes predictions, and then evaluates hardware-mapped conditions with 512 paired seeds per cell. Equivalent command-weight configurations have explicit shared provenance and never increase independent sample counts. Historical FP16 timings define replay scenarios for the trained JAX FP32 policy. All selected tasks and fit limitations remain visible in the generated reports; outputs stay outside Git.

Hardware validation completes and verifies all coarse jobs before dispatching any fine jobs, including when resuming detached workers. Each mode covers four tasks, four hardware profiles and N=1..5. The default study needs 6,144 new coarse episodes and 40,960 new fine episodes after explicit reuse; per-hardware result tables retain all logical combinations.

Use `--workers-per-gpu 2` to allow two independent processes per selected GPU; the default remains one. Only matching workers owned by the current study may share a GPU. Adopted and pre-CUDA workers count toward the limit, with separate CPU reservations. This changes evaluation throughput, not simulated latency. Contended host timings are not dedicated-device latency measurements. Validate task outcomes and control counts; unchanged native GPU code does not imply bitwise-identical floating-point trajectories.
