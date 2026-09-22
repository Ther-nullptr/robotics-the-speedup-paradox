# KINETIX：原生时间下的 latency–quality 实验

本case在仓内维护 RTC flow policy、KINETIX 环境、Jax2D物理核心、12个关卡定义和所需渲染素材。运行不导入参考仓库源码，不需要外部Kinetix/Jax2D安装包；外部资源仅包括兼容的Python计算依赖和显式提供的可信checkpoint。

当前入口使用JAX/Flax、symbolic观测、6维motor/thruster动作、8步预测块，默认每4条控制指令重新推理。质量轴为flow采样步数，时间轴为显式注入的虚拟延迟；它是动态任务的动作可用性重放，不使用静态case的 `paper_async`。

## 环境与资源

独立运行环境要求Python 3.11。本机验证的主要组合为JAX 0.4.35、jaxlib 0.4.34、Flax 0.10.2、NumPy 1.26.4和CUDA 12 JAX插件。依赖列表见 [requirements-runtime.txt](requirements-runtime.txt)。CPU契约/统计测试不安装这套运行栈。

新环境安装方式如下；这是按已验证版本整理的安装配方，尚未在空白机器完成独立GPU安装验收：

```bash
python3.11 -m venv .local/envs/kinetix
.local/envs/kinetix/bin/python -m pip install -r benchmarks/dynamic/kinetix/requirements-runtime.txt
```

模型参数采用原始RTC checkpoint。目录中应有 `worlds_l_car_launch.pkl`、`worlds_l_mjc_walker.pkl` 等按关卡命名的文件。原始资源来源为 [RTC项目](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) 的 `gs://rtc-assets/bc/`；训练轨迹不是评估前提。入口不下载资源。pickle仅用于调用者明确指定且可信的本地模型文件。

复制 [路径模板](paths.env.example)，填写checkpoint目录与运行Python即可；没有 `--source` 参数：

```bash
mkdir -p .local
cp -n benchmarks/dynamic/kinetix/paths.env.example .local/kinetix.env
# Fill in the two existing local paths, then source the file.
source .local/kinetix.env
```

## 启动与矩阵评估

先预检，预检只读取仓内代码/关卡和checkpoint文件身份，不导入JAX、不创建实验目录：

```bash
bash benchmarks/dynamic/kinetix/run.sh \
  --levels car_launch,mjc_walker --flow-steps 1,3,5 \
  --latencies-ms 0,7.465,21.938 --episodes 2 --start-seed 0 \
  --output-dir runs/dynamic/kinetix/matrix-001 --dry-run
```

真实运行添加空闲GPU编号；首次迁移或修改物理代码时添加 `--validate-native`：

```bash
bash benchmarks/dynamic/kinetix/run.sh \
  --levels car_launch,mjc_walker --flow-steps 1,5 \
  --latencies-ms 0,21.938 --episodes 1 --start-seed 0 \
  --gpu 3 --validate-native --record-video \
  --output-dir runs/dynamic/kinetix/matrix-001
```

`--episodes` 是每个“关卡×flow步数×延迟”单元的回合数，上例共8回合。所有单元使用相同的seed列表；环境、动作噪声和policy使用独立随机流。`--levels all` 覆盖12个内置关卡。每个关卡绑定自己的checkpoint，不把模型与环境任意组合。

`--max-steps` 可以设置更小的控制步预算，但不能超过原生任务预算。CPU运行可以用 `--cpu` 代替 `--gpu`。每次真实运行使用新输出目录。当前是单环境逐回合执行；JAX编译policy与物理内核，不宣称已实现跨环境并行或真实后台推理。

完整回合覆盖定义为：每个指定seed从本关卡初态开始，遇到原生成功/失败接触或预算耗尽时停止，不自动reset后继续计入同一回合。运行异常、非有限状态和非法模型输出使实验失败，不计为策略失败回合。

## native-blend 时间协议

原生关卡保持 `dt=1/60 s`、`frame_skip=2`，即每条控制指令33.333 ms；默认预算256条。每4条指令刷新一次policy。flow ODE中的积分步长 `1/N` 与这些物理时间量无关。

设注入延迟为L，物理槽k的旧指令占比为：

```text
old_weight[k] = clip(L / physics_dt - k, 0, 1)
command[k] = old_weight[k] * process(old_action)
           + (1 - old_weight[k]) * process(new_action)
```

L和physics_dt使用相同单位。两组动作先经过原生motor/thruster绑定、裁剪及自动电机处理，再在执行器指令域混合；同一个控制噪声样本同时作用于新旧候选，噪声默认标准差0.1，每条控制指令采样一次。

例如21.938 ms延迟对应1.31628个物理槽：第一槽使用旧指令，第二槽使用约31.6%旧指令和68.4%新指令，后续槽使用新指令。每条控制仍调用原生物理求解器两次，碰撞/关节参数、求解器迭代数和原生控制边界终止规则保持不变。

旧动作取前一预测块尚未执行的尾部；初始旧队列为原生动作空间的零向量，没有初始预取。零向量仍经过原生执行器转换，不假定它能冻结世界。延迟不得超过当前执行窗口，也不得超过可用旧动作尾部；默认H=8、执行4条，因此最多允许133.333 ms。

零延迟、完整旧指令及恒定指令直接使用仓内原生步进函数，保留原有编译边界；只有控制内部需要切换的槽走混合路径。`--validate-native` 将这些端点和恒定指令与原生状态、观测、奖励及终止结果逐值比较。

**分数槽混合是指令时间平均近似。** 它保留原生物理离散化，但不是在槽内部重新执行碰撞检测，也不声称得到精确中途切换的状态轨迹。宿主推理墙钟不会再推进模拟器，避免把注入延迟和本机计算时间重复相加。

## 旧映射与硬件profile

`--mapping legacy-round` 只把延迟按原生控制周期进行nearest-even取整，保持相同的零初始队列和物理参数；它用于映射敏感性对照，不是原上游RTC初始预取、RTC guidance或BID的完整复现。小于半个控制周期的延迟可能映射为0，requested/effective延迟分别记录。

当前不提供修改dt、frame_skip或碰撞系数的参数。此前参考实验将物理步长细分到约0.417 ms，虽然保留了控制周期，却改变了约束求解次数和部分任务动力学；此类研究应在独立分支开展。

可用CSV为每个flow步数绑定一个延迟场景，代替笛卡尔扫描：

```text
flow_steps,latency_ms
1,7.465
3,21.938
5,36.738
```

以上仅展示文件格式，不能作为本机性能证据。调用时使用 `--flow-steps 1,3,5 --latency-profile profile.csv`，不再传 `--latencies-ms`。manifest保存文件哈希，并标记为外部profile场景；不会认证目标硬件、dtype或后端是否与本次JAX/float32 policy等价。参考仓库中的Torch/FP16、低精度算子profile不得直接解释为当前JAX路径的实测时延。

## 输出与统计

| 文件 | 内容 |
| --- | --- |
| `case-manifest.json` | 仓内源码/关卡/checkpoint身份、物理配置、模型结构、设备及状态 |
| `run.log` | 加载、编译/运行进度和异常 |
| `requests.jsonl` | 每次观测/推理、虚拟release、新旧动作来源、物理槽权重及控制步 |
| `episodes.jsonl` | 每回合成功、实际控制/物理步数、预算、return、模拟时间和推理次数 |
| `summary.md/.json/.csv` | 分关卡、flow步数、延迟单元汇总SR、全体预算惩罚步数和仅成功步数 |
| `videos/` | 可选完整回合录像，初始帧加每条实际控制后的帧 |
| `failure.json` | 基础设施异常；已完成部分保持partial状态 |

原生渲染为125×125，录像仅在边缘补齐到126×126以适配编码器，不改变模拟器或模型观测。播放帧率来自原生控制周期，视频开关默认关闭。

失败按声明预算计入总体步数，成功使用实际步数；仅成功均值以成功回合数为分母。不同实验单元分别汇总，不按成功率挑选最佳horizon或合并不兼容协议。失败可能在预算之前由原生接触判据确定，实际步数仍保留在raw ledger。

可以独立重新汇总：

```bash
python benchmarks/dynamic/kinetix/summarize.py --input runs/dynamic/kinetix/matrix-001
```

中断实验必须显式加 `--allow-partial`。请求中的host时间可能包含首次JIT编译，不能直接当作稳态推理时延；当前不自动输出真实硬件加速比。

## 源码与范围

代码位于 `src/robotics_bench/kinetix/`：`flow_model.py` 是仓内模型执行，`policy.py` 加载参数和管理policy随机流；`native/` 是私有包名隔离的KINETIX/Jax2D运行模块，`environment.py` 实现原生步进/边界混合；`protocol.py`、`runner.py`、`results.py` 分别负责延迟、回合和统计。

移植来源、原始文件hash和变更边界见 [PROVENANCE.json](../../../src/robotics_bench/kinetix/PROVENANCE.json)，许可见 [第三方说明](../../../THIRD_PARTY_NOTICES.md)。编辑器、PPO训练、云端下载和中间实验产物没有迁入。参考代码审查见 [latency–quality整理](../../../docs/kinetix-reference-review.md)。

当前接入flow采样步数、虚拟延迟、原生噪声控制和配对评估。Torch等价policy、低精度内核、结构化student、RTC guidance/BID以及质量曲线拟合尚未接入执行入口。DOM继续独立规划。

## 当前验证

RTX 6000 Ada上完成 `car_launch` 与 `mjc_walker`、N∈{1,5}、L∈{0,21.938 ms}、seed0的8回合接入检查。car_launch四个单元均成功，控制步数分别41、43、44、46；mjc_walker分别为成功167步、失败、成功149步、成功207步。失败按256条预算进入总体统计。每单元仅一个回合，不能据此报告总体质量或最优flow步数。

两关分别完成zero-delay、full-old和constant-command的逐值原生对齐；四个配置的初始观测hash各自一致。八段视频均可解码，为126×126，帧数严格等于实际控制步数加1。wheel中的关卡、纹理、许可证和独立CPU统计已检查。完整12关全量评估、目标硬件profile匹配和真正异步并发尚未验收。

## English summary

This case owns the RTC flow model, Kinetix runtime and Jax2D physics source. Supply only a trusted local checkpoint directory and a Python 3.11 JAX environment. Native-blend replays declared action availability on the unchanged native physics grid; it is a processed-command time-average approximation, not exact substep integration or hardware-in-the-loop execution. Flow-step and latency sweeps use paired seeds and per-cell failure-budget statistics. Optional videos retain every evaluated control frame. See the commands above; `--source` is intentionally absent.
