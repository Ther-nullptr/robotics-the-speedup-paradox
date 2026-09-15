# 相对统一 baseline 的加速比

本协议参考 [The Speedup Paradox，arXiv:2606.28529v2](https://arxiv.org/pdf/2606.28529v2) 的 Eq.1–5 与成功条件统计 Eq.31。统一方向是 **baseline 时间 / 当前配置时间**：大于1表示更快，小于1表示变慢，baseline自身为1。

量化、减少生成步数、cache优化和异步都可以按同一baseline比较，但推理、动作块周期和整项任务的加速比分别输出。推荐跨方法主表保留 `speedup_chunk_estimated` 与 `speedup_task_success`，同时报告SR；模型内部收益另看 `speedup_inference`。

## 1. baseline 按实验案例固定

一个比较案例使用同一基础模型、任务/初态集合、硬件、环境版本、world progression、控制周期、执行长度、计时范围和统计规则。baseline通常选择未采用待测轻量化方案的全精度同步配置。不同硬件各自建立baseline，不能用某张GPU的FP16时间归一化所有设备。

四组 F-S、Q-S、F-A、Q-A 均引用同一个baseline ID；不让“量化＋异步”私下改为相对于“量化＋同步”报告。后者可以另列为增量效果，但不使用主表同名字段。矩阵定义见 [组合实验协议](composable-experiments.md)。

## 2. 论文的周期时间与统一写法

记：

- `I0`：baseline的一次完整policy推理时间，即论文的 `Tinf`。
- `Ic`：配置c对应的推理时间；量化配置可写作 `α I0`。
- `n`：每个完整动作块实际执行的原子动作数；预测horizon另记。
- `a`：一个原子动作的执行时长，即论文的 `Tact`。
- `m`：可以与下一次推理重叠的动作数，即论文的 `n′`，范围0到n。

baseline动作块周期为：

$$
C_0=I_0+na.
$$

论文量化等模型内优化对应 `Cc=α I0+na`；执行感知优化对应 `Cc=I0+na−min(I0,ma)`。[论文 pp.3–4，Eq.1–3](https://arxiv.org/pdf/2606.28529v2#page=3)

为了同时表示量化、异步和组合，采用以下统一写法：

$$
\boxed{C_c=I_c+na-\min(I_c,ma)=na+\max(0,I_c-ma)}.
$$

这是把论文两类周期模型合并的工程推导，**不是论文直接给出的组合公式**。它假设规则重叠、固定动作长度和代表性推理成本；额外的排队、通信或资源竞争需通过对应范围的测量或实际trace另行反映。

静态 `paper_async` 采用论文附录B.1的 `t−n′` 历史观测实验，并将上述周期单列为 `paper_model`；真实host并发不是这个抽象实验的验收条件。π0.5 case默认n为5、`Tact=1000/30 ms`，后者只用于解析模型，不修改仿真步长。没有提供对应 `Tinf` 时，该case不生成周期/加速比，也不以0或宿主 `eval_s` 代填；可以先完成历史观测下的质量实验。[论文§3.1与B.1](https://arxiv.org/html/2606.28529v2)

| 配置 | `Ic` | `m` |
| --- | --- | --- |
| baseline | `I0` | 0 |
| 量化＋同步 | 量化变体的推理时间 | 0 |
| 原精度＋异步 | 理想情况下为`I0`；实际以测量为准 | 指定重叠深度 |
| 量化＋异步 | 组合条件下的推理时间 | 指定重叠深度 |

更复杂调度无法满足这个模型时，仍可用同范围的实测周期作baseline/variant比值，但必须另记为 measured，不能将其当作这个解析模型的无条件预测。

## 3. 三层加速比分别计算

### 单次推理

$$
S_{\mathrm{inf},c}=\frac{I_0}{I_c},\qquad
\alpha_c=\frac{I_c}{I_0}=\frac1{S_{\mathrm{inf},c}}.
$$

`α`是时延比例，不是加速倍数。纯异步在理想情况下没有减少模型计算时间，所以推理加速比为1，收益体现在周期与任务层。实际异步资源竞争可能改变Ic，应报告真实输入值，不强制写1。

### 每个动作块周期

$$
\boxed{S_{\mathrm{chunk},c}=\frac{C_0}{C_c}}
=\frac{I_0+na}{I_c+na-\min(I_c,ma)}.
$$

这对应论文的 `ηchunk`，是量化与异步可以直接使用的共同速度坐标。不要把 residual wait `max(0,Ic−ma)` 当作整个推理时间；完全隐藏等待也不会让推理加速比变成无穷大。

### 整项任务

论文用动作块数量与周期分解任务时间：

$$
\widehat T_{\mathrm{task},c}=N_c C_c,
\qquad
\boxed{\widehat S_{\mathrm{task},c}
=\frac{N_0}{N_c}S_{\mathrm{chunk},c}}.
$$

帽号表示本工具按论文稳态模型计算的估计。Nc必须来自对应配置的任务统计，不能从位宽或overlap推断。减少周期成本可能被更多动作块抵消，这正是论文指出的speedup paradox。[论文 p.4，Eq.4–5](https://arxiv.org/pdf/2606.28529v2#page=4)

## 4. 成功条件均值与直接任务计时

为对应论文附录D.1，任务时间与N的默认统计范围是成功试验：[论文 p.23，Eq.31](https://arxiv.org/pdf/2606.28529v2#page=23)。直接任务加速比定义为：

$$
S_{\mathrm{task,succ},c}
=\frac{\operatorname{mean}_{e\in\mathcal S_0} T_{0,e}}
       {\operatorname{mean}_{e\in\mathcal S_c} T_{c,e}}.
$$

先按每个配置的成功样本计算均值，再求比值。它不同于“逐试验加速比的平均”：例如baseline耗时 `[1,9]`，候选 `[1,3]`，均值之比为2.5，而逐试验比值平均为2。

两组成功集合可能不同，必须同时报告 trials、successes、SR及SR差值（百分点）。任意一侧没有成功样本时，成功条件任务加速比为null，并给出原因；不能填0或无穷大。共同成功seed上的配对结果可作为补充，不能代替完整样本SR。

实际任务时间应按声明的起点到terminal记录，包含首轮、末轮和提前终止。即使理想条件下完整执行N个等长chunk，首轮没有旧动作可重叠时，也可能出现：

$$
T_{\mathrm{finite},c}=I_c+na+(N-1)C_c
=NC_c+\min(I_c,ma).
$$

这是额外的有条件边界推导，不是当前calculator默认输出。实际时延抖动、变长chunk或复杂队列下，直接使用任务trace/统计；不要为了符合NC公式丢掉这些时间。

## 5. 四组手算例子

设baseline推理100 ms，量化后50 ms；每块执行10个动作，每动作20 ms；异步重叠3个动作，即60 ms。以下均为合成演示数据。

| 配置 | 推理/ms | 周期估计/ms | 推理加速比 | 周期加速比 |
| --- | ---: | ---: | ---: | ---: |
| FP16＋同步 | 100 | 300 | 1.00× | 1.00× |
| 量化＋同步 | 50 | 250 | 2.00× | 1.20× |
| FP16＋异步 | 100 | 240 | 1.00× | 1.25× |
| 量化＋异步 | 50 | 200 | 2.00× | 1.50× |

若baseline平均需要10个chunk、组合方案平均需要16个chunk，那么任务时间估计为3000与3200 ms，组合任务加速比仅为 `3000/3200=0.9375×`。周期变快与整项任务变快因此要分开报告。

本例的两个单项周期加速比乘积恰好等于组合值，不能据此一般化。若量化后改为60 ms，组合周期仍为200 ms，但单项乘积为 `(300/260)×(300/240)≈1.4423`，并不等于组合的1.5。

## 6. 计算工具

在仓库根目录：

```bash
python tools/compare_speedups.py --input examples/speedups/synthetic.json
python tools/compare_speedups.py --input examples/speedups/synthetic.json --markdown
```

输入格式与示例见 [examples/speedups](../../examples/speedups)。这是一份独立的比较摘要格式，不是既有run-manifest v1；工具不读取模型、不运行仿真，也不从输入时间反推出成功率或N。

共同输入声明case、时间范围、时钟域、n和动作时长；每行提供推理时间、overlap、试验数/成功数，可选提供成功条件N和任务时间均值。baseline由ID显式指定，必须是同步配置。

| 输出字段 | 口径 |
| --- | --- |
| `speedup_inference` | baseline推理时延 / 当前推理时延 |
| `speedup_chunk_estimated` | 解析周期C0/Cc |
| `speedup_task_estimated` | 成功条件N与解析周期形成的比值 |
| `speedup_task_success` | 输入提供的成功条件任务时间均值之比 |
| `latency_ratio_alpha` | 当前推理时间 / baseline推理时间 |

缺少N或任务时间均值时，相应任务指标为null并标原因；已有推理/周期比值仍可输出。`paper_steady_state`保存估计，`provided_summary`保存调用方提供的任务均值，避免两者互相冒充。

`case_id`和计时元数据是调用方对可比性的声明，计算工具不能核实这些时间是否来自同硬件、同初态、同真实实验。合成样例始终标 `synthetic=true`；工具没有测得这些数值。

## 7. 汇总与可视化规则

标量周期模型适合固定/代表性时延。若profile含jitter，先逐样本计算周期再按声明规则汇总，或直接用trace。例如推理20/100 ms各占一半、overlap60 ms时，平均残余等待20 ms；将平均推理60 ms代入却得到0。**当前工具只计算输入标量对应的估计**，不声称得到真实抖动下的平均周期。

同样，周期可变时一般有 `mean(N*C) != mean(N)*mean(C)`。直接成功任务时间均值优先于用两个均值凑乘积。推理时延均值/分位数比较要范围一致，不把baseline均值与候选p95相除。

静态任务图可使用周期加速比作横轴、成功条件任务加速比或任务时间作纵轴，并同时展示SR。动态任务重点报告SR/score、环境速度与观测年龄，速度是共同参考坐标；SR比值不命名为“加速比”。打分任务的score也不能当作二项成功率。

固定预算失败惩罚指标使用独立字段，不能混入成功条件任务时间。`compare_speedups.py`处理调用方提供的时间摘要，不聚合原始episode或计算置信区间；下面的控制步数工具独立处理episode统计，不生成时间加速比。

## 8. 控制步数统计与实验汇总

[summarize_experiment.py](../../tools/summarize_experiment.py) 从 `episodes.jsonl` 汇总实际控制步数。每条记录包含 `task`、`init_state_id`、布尔 `success` 和非负整数 `primitive_steps`，可选 `env_seed`；同一运行内重复的任务、初态与seed组合会被拒绝。这些步数不是任务耗时、模型调用数或动作块数量，不能直接作为前面公式中的N。

以下是本工具的工程统计定义。设本次运行有N个episode、S个成功episode，成功集合为 $\mathcal S$，失败集合为 $\mathcal F$；$b_i$ 为实际控制步数，$B_i$ 为该episode声明的最大步数预算。

| 统计范围 | 均值 | 含义 |
| --- | --- | --- |
| 全体实际（JSON复核项） | $\sum_i b_i/N$ | 成功和失败均使用实际步数，不在终端、Markdown或CSV中单列 |
| 仅成功 | $\sum_{i\in\mathcal S} b_i/S$ | 仅在成功样本内归一化；S=0时为null |
| 仅失败实际（JSON复核项） | $\sum_{i\in\mathcal F} b_i/(N-S)$ | 失败样本的实际步数；无失败时为null，不单列到终端、Markdown或CSV展示 |
| 失败按预算惩罚 | $(\sum_{i\in\mathcal S}b_i+\sum_{i\in\mathcal F}B_i)/N$ | 失败实际步数被对应预算替换 |
| 成功加权分量（JSON复核项） | $\sum_{i\in\mathcal S} b_i/N$ | 等于SR乘成功条件均值；不在终端、Markdown或CSV中单列，不能标成“仅成功均值” |

报告提供样本数、成功数、SR，以及总体预算惩罚与仅成功两种口径的总步数、均值。当前π0.5＋LIBERO的未成功episode运行至预算上限，因此总体统一使用预算惩罚口径，不再并列展示实际全体步数。没有成功样本时，成功总步数为0，成功均值仍为null。JSON中的实际步数及加权分量用于复核，不作为额外展示指标；对于允许提前失败的其他环境，总体也按声明的预算计入失败代价。

跨任务合并时，任务t有 $n_t$ 个episode、$s_t$ 个成功episode：总体预算均值按 $n_t/N$ 加权，成功条件均值按 $s_t/S$ 加权。任务样本数或成功数不等时，不直接平均各任务的均值。无成功任务对成功总和贡献为0，但其成功均值不填0。多个运行输入分别汇总，不混合成一个样本池。

预算可以统一提供，也可以按任务覆盖；不同任务的失败样本分别使用自己的预算。新运行的 `coverage.json` 记录 `max_primitive_steps`，工具可自动读取。旧结果没有该字段时，需要调用者明确提供预算；不从最大观测步数猜测。若有失败样本缺少预算，惩罚总数和均值为null，并列出缺少预算的任务；实际统计照常输出。全成功样本不需要失败预算。提供的预算不得小于该任务任何已记录episode的实际步数。

从仓库根目录运行：

```bash
# 新运行：读取目录中的ledger与完成/预算元数据，展开任务明细
python tools/summarize_experiment.py --input runs/static/pi05_libero/sync-001 --per-task

# 旧LIBERO object结果：显式声明280步预算；两次运行分别汇总
python tools/summarize_experiment.py \
  --input runs/static/pi05_libero/sync-001 runs/static/pi05_libero/paper-async-001 \
  --max-steps 280 --per-task --output results/episode-comparison.csv

# 独立ledger：任务名须与记录一致；可重复传入任务预算
python tools/summarize_experiment.py --input PATH/TO/episodes.jsonl \
  --max-steps 280 --task-max-steps TASK_NAME=320 --format json
```

`--input` 接受一个或多个运行目录或JSONL文件。默认向stdout输出Markdown；`--format markdown|json|csv` 显式选择格式，未指定时可由 `--output` 的扩展名推断。输出文件必须是新文件。JSON始终包含各任务统计；Markdown和CSV使用 `--per-task` 展开任务行。缺失值在JSON中为null，Markdown显示N/A。

工具依据相邻的 `case-manifest.json`、`coverage.json` 核对完整结果的状态、数量和覆盖。未完成或失败的运行默认拒绝，显式添加 `--allow-partial` 才汇总已记录样本，并标为 `partial`。裸ledger或缺少充分完成元数据的输入标为 `unverified`；统计可计算不代表运行已完整。该工具只使用CPU，不加载模型或启动仿真。
