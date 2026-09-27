# KINETIX：保持原生控制语义的评估加速

本页区分两类工作：定位原生GPU执行的数值重复性，以及减少单环境评估的host调度/同步。模型计算、物理源码与默认`run.py`的执行方式保持原样；policy新增GPU结果接口，原同步NumPy接口保留。新入口都是显式实验工具。环境与checkpoint配置见[KINETIX入口](../benchmarks/dynamic/kinetix/README.md)，质量矩阵和动作预处理JIT基准见[flow步数实验](kinetix-flow-quality.md)。

## Device chunks

`device_chunk_candidate.py`将原本在host循环中逐条调用的4条动作放入同一个JAX扫描。模型仍预测8条动作，每4条重新推理；每条控制仍执行原生`frame_skip=2`物理更新，保持dt=1/60、30Hz控制和原生预算。每条控制之后检查原生终止、成功和非有限状态，结束后剩余槽不再调用物理更新。

扫描返回每个实际控制边界的状态和观测，host按原顺序逐条暴露它们、更新计数和记录事件。首次host步进等待整个执行窗口完成，后续步进读取缓存。这个组织方式适用于host墙钟不推进模拟时间的离线评估；没有实现模型后台推理，也不改变模拟器控制频率。

当前限定**零注入延迟、execute horizon=4、原生控制步预算**。非零延迟会被拒绝；不用于自定义较短预算、其他horizon或实时硬件。候选保留在benchmark中，没有加入默认runner或主质量矩阵。

两个数值边界必须保留：

- `random.normal`结果与噪声缩放分别经过编译屏障，避免扩大JIT范围后重排浮点运算。20个seed/control组合的诊断中，直接融合均与原始噪声不同；保留边界后全部逐位相同。
- 物理`EnvParams`仍作为运行时参数传入编译函数，避免闭包常量专门化改变原始计算。初始候选和仅修复噪声的候选都在第1控制步未通过检查；两项边界修正后才通过下表中的完整轨迹检查。失败候选没有生成性能结论。

## 启动与验收

```bash
source .local/kinetix.env
# Choose a GPU with no compute process and zero utilization.
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv

"$ROBOTICS_KINETIX_PYTHON" -B -u benchmarks/dynamic/kinetix/benchmark_device_chunks.py \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" --gpu 3 \
  --level car_launch --seeds 0,1,2,3 --flow-steps 5 --repeats 3 \
  --output-dir runs/dynamic/kinetix/device-chunks-car-001
```

每个seed先执行两次原版以及一次候选，核对完整状态hash、模型输出hash、语义事件和结果。任意原版重复性或候选逐位检查失败，状态为`equivalence_failed`，不开始计时。通过后，同一进程交替测量原版/候选，并再次确认每个计时回合的结果等于已验证结果。

输出`benchmark.json`以及两个工具源码快照。报告记录代码/checkpoint SHA、Python/依赖、GPU/驱动、XLA flags、实际模型和环境参数、原始trace及计时样本。加速比是匹配回合的原版wall time总和除以候选总和；包含reset与整回合，不含资源加载、首次编译、逐位trace采集和文件输出。不与其他进程的旧动作预处理JIT数字相除计算增量收益。

RTX 6000 Ada、JAX0.4.35/jaxlib0.4.34、原RTC BC24 FP32、N=5、seed0..3、每seed三轮计时，默认XLA执行模式的验证：

| 任务 | 原版合计 | Device chunks合计 | 加速比 | 三轮比值范围 |
| --- | ---: | ---: | ---: | --- |
| Car Launch | 15.984 s | 11.237 s | 1.422× | 1.409–1.431× |
| H17 Unicycle | 21.231 s | 17.496 s | 1.213× | 1.194–1.227× |
| Chain Lander | 7.892 s | 5.542 s | 1.424× | 1.415–1.442× |

共36对计时回合，全部匹配已验证结果；每种实现的参考/候选轨迹覆盖878个控制状态、222次模型调用。Car的seed2执行256步后预算耗尽；Unicycle执行75/72/75/70步，覆盖窗口中途成功；Chain Lander的seed2执行51步后原生失败，覆盖窗口中途失败。三轮范围是实测波动范围，不是置信区间。这些结果只覆盖上述3个任务、N=5和4个seed，不能作为全任务加速或成功率结果。

独立检查噪声舍入边界，无需加载模型：

```bash
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/check_noise_rounding.py \
  --gpu 1 --output-dir runs/dynamic/kinetix/noise-rounding-001
```

## 动作驻留GPU

原来的device-chunk路径仍会把模型输出转成CPU NumPy，再把动作传回GPU。`KinetixFlowPolicy.infer_device()`新增不做显式host回读的JAX数组接口；原`infer()`仍返回同步完成的NumPy数组，两者共享同一个采样器和RNG请求计数。

`resident_action_runner.py`将完整8条预测动作直接交给独立编译的控制窗口。shape/dtype检查只读取metadata；全部动作的有限值检查在GPU进行，检查失败时执行0条控制并保持状态不变，包含本轮不执行的尾部动作。每条实际控制的终止、成功、非有限状态和预算检查保留。模型与物理没有合并成一个JIT；中间状态/观测仍完整返回，没有同时进行状态压缩或多环境批量化。

计时路径只回传窗口metadata，不读取完整动作值；完整动作仅在明确的trace采集阶段回读，用于hash/事件核对。`policy_enqueue_seconds`只表示主机提交跨度，不能当作完成的模型推理时延；`window_call_and_wait_seconds`也会等待尚未完成的模型GPU执行，不能解释为纯环境耗时。比较以完整回合wall time为准。

```bash
"$ROBOTICS_KINETIX_PYTHON" -B -u benchmarks/dynamic/kinetix/benchmark_resident_actions.py \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" --gpu 3 \
  --level car_launch --flow-steps 5 --seeds 0,1,2,3 --repeats 5 \
  --reference-run runs/dynamic/kinetix/device-chunks-car-001/benchmark.json \
  --output-dir runs/dynamic/kinetix/resident-actions-car-001
```

`--reference-run`可选，用于核对既有device-chunk产物中的原生轨迹；关卡、checkpoint、模型/物理配置、N、seed和执行模式须匹配。脚本先检查原生重复性、原生/device-chunk/驻留路径的完整轨迹及驻留路径自身重复性，任一失败就不计时。随后按轮交替测量host往返版device chunks与驻留版，并确认计时结果匹配轨迹、完整动作回读次数为0。

以下增量来自RTX 6000 Ada、原RTC BC24 FP32、N=5、每任务seed0..3、五轮配对计时。每个任务的两种实现使用同一卡和同一进程；物理与控制条件沿用上节。

| 任务 | Host往返版device chunks | 动作驻留版 | 增量加速比 | 五轮比值范围 |
| --- | ---: | ---: | ---: | --- |
| Car Launch | 18.309 s | 17.540 s | 1.044× | 1.036–1.056× |
| H17 Unicycle | 29.334 s | 28.415 s | 1.032× | 1.031–1.034× |
| Chain Lander | 19.190 s | 18.555 s | 1.034× | 1.032–1.038× |

共60对计时回合，逐回合结果一致，驻留计时分支的完整动作回读次数均为0；五轮比值均大于1。各路径的12个seed/任务组合覆盖878个控制状态和222次模型调用，并与此前保存的原生轨迹一致。额外GPU检查确认“首条动作含Inf”和“未执行尾部含NaN”均不会推进物理状态。五轮范围是实测波动，不是置信区间。

这些是相对host往返版device chunks的**增量**，不能乘以上节不同运行的加速比作为原生总加速。新runner也减少了CPU数组复制、旧动作尾部维护和调度检查，因此结果归属于整条驻留动作实现，未单独隔离PCIe传输的贡献。计时覆盖参数检查、协议构造、reset及完整回合；排除首次编译、trace和文件输出。当前仍只验证这3个任务、N=5、4个seed的零延迟/原生预算/execute-four场景，默认`run.py`不自动启用此路径。

输出为`benchmark.json`和工具源码快照，包含各条路径的原始trace、动作有限值检查、配对计时、依赖/代码/设备身份以及可选历史引用的hash。脚本和报告字段使用英文。

## 原生重复性诊断

`diagnose_native_repeatability.py`记录每个原生控制后的完整状态，比较状态叶的数值变化与signed zero，检查轨迹长度和回合结果。发现状态分歧后，保存该步的完整输入、两种输出及可选优化后HLO，再重复执行同一个原生`engine_step`，排除模型、随机数流和动作预处理的影响。这里增加了诊断同步，不用于测量性能。

```bash
"$ROBOTICS_KINETIX_PYTHON" -B -u benchmarks/dynamic/kinetix/diagnose_native_repeatability.py \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" --gpu 0 \
  --level grasp_easy --seed 0 --episode-repeats 10 --frozen-repeats 50 \
  --dump-hlo --output-dir runs/dynamic/kinetix/native-repeatability-001
```

Grasp Easy seed0的实测在第19控制步首次出现数值差异；之前的状态、加噪动作和执行器指令均逐位相同。变化包括polygon速度/角速度、关节冲量及碰撞状态，最大绝对差异约1.43e−6，不是正负零。固定同一组原生物理输入后，50次调用得到两种状态输出，分别出现43/7次；输入hash始终不变。因此差异已定位到原生GPU `engine_step`内部，但尚未锁定具体scatter/kernel。

输出包含`diagnostic.json`、`frozen_input.npz`、分歧状态NPZ、各唯一冻结输出、诊断源码快照，以及可选`native_engine_optimized_hlo.txt`。选择的是采样回合相对第一回合最早出现的状态分歧；长度或结果不同时也单独记录。没有观察到分歧不等于证明任意次数、seed和任务均可重复。

可复用已记录的输入进行编译模式对照：

```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true \
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/diagnose_native_repeatability.py \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" --gpu 0 --level grasp_easy \
  --frozen-input runs/dynamic/kinetix/native-repeatability-001/frozen_input.npz \
  --frozen-repeats 50 --output-dir runs/dynamic/kinetix/native-deterministic-001
```

工具验证关卡、原始代码身份、数组布局和输入hash，不修改原数据。该确定性开关在本机旧运行栈已实际测试；[OpenXLA的参数说明](https://raw.githubusercontent.com/openxla/xla/main/xla/debug_options_flags.cc)描述其运行间确定性目标，其他版本仍需独立验证。

同一冻结输入在该模式下50次输出相同，但输出hash不等于本次默认模式观察到的两种输出；不能把它称为保持默认轨迹的修复。Grasp Easy在该模式下的4seed预处理JIT对照通过检查，参考/候选合计63.958/63.164秒，只有1.013×模式内比值。另两项较长诊断触发300秒截止，没有完整性能结果。该模式因此仅作诊断对照，不进入默认设置，也不与默认模式结果合并。
