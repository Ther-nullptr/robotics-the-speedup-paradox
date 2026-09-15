# π0.5＋LIBERO 静态 case

使用 [run.sh](run.sh) 直接从命令行选择同步、论文异步和单层量化，不需要手写JSON。它通过 [run.py](run.py) 调用用户提供的兼容VLASH sim evaluator，使用原生LeRobot π0.5及同源processors；不使用VLASH微调checkpoint，也不做预测状态替换。

## 配置一次本机路径

从仓库根目录复制 [路径模板](paths.env.example)，填写本机资源位置后显式加载：

```bash
mkdir -p .local
cp -n benchmarks/static/pi05_libero/paths.env.example .local/pi05-libero.env
# 编辑 .local/pi05-libero.env，填写自己的Python和资源路径
source .local/pi05-libero.env
```

模板中的变量是：

| 变量 | 内容 |
| --- | --- |
| `ROBOTICS_PI05_PYTHON` | 已安装模型/仿真依赖的Python解释器 |
| `ROBOTICS_SIM_SOURCE` | 兼容VLASH sim checkout |
| `ROBOTICS_CHECKPOINT` | 任务匹配的原生LeRobot π0.5 LIBERO checkpoint |
| `ROBOTICS_TOKENIZER` | 与checkpoint匹配的tokenizer目录 |
| `ROBOTICS_LIBERO_CONFIG_DIR` | 含 `config.yaml` 的LIBERO配置目录 |
| `ROBOTICS_QUANT_SOURCE` | 量化适配源码根目录，量化实验需要 |
| `ROBOTICS_KERNEL_SOURCE` | 兼容量化kernel源码根目录，量化实验需要 |

本地env文件不进入Git，脚本也不会自动加载它。路径仍可用 `--sim-source`、`--checkpoint`、`--tokenizer`、`--libero-config-dir`、`--quant-source`、`--kernel-source` 覆盖环境变量。也可直接使用 `python benchmarks/static/pi05_libero/run.py`，选择具备依赖的Python即可；`ROBOTICS_PI05_PYTHON`用于shell入口选择解释器。

## 四种运行方式

下面使用GPU 3；按本机情况替换索引或填写完整GPU UUID。第一条带 `--dry-run` 预览，实际执行基线时去掉该选项；其余三条直接运行。每次实际执行都使用新的输出目录。

```bash
# 同步基线：先预览，去掉 --dry-run 即执行
bash benchmarks/static/pi05_libero/run.sh \
  --schedule sync --quant none \
  --gpu 3 --output-dir runs/static/pi05_libero/sync-001 --dry-run

# 论文异步：重叠动作数 n′=2
bash benchmarks/static/pi05_libero/run.sh \
  --schedule paper_async --overlap-actions 2 --quant none \
  --gpu 3 --output-dir runs/static/pi05_libero/paper-async-001

# 同步＋单层W8A8
bash benchmarks/static/pi05_libero/run.sh \
  --schedule sync --quant w8a8-single-layer \
  --gpu 3 --output-dir runs/static/pi05_libero/w8a8-001

# 论文异步＋单层W8A8：组合尚未进行闭环验证
bash benchmarks/static/pi05_libero/run.sh \
  --schedule paper_async --overlap-actions 2 --quant w8a8-single-layer \
  --gpu 3 --output-dir runs/static/pi05_libero/w8a8-paper-async-001
```

`--quant w8a8-single-layer` 自动选择内置单层配置和profile，只替换 `TXT.B00.mlp.down`。不需要另外指定JSON；仍需提供量化源码和kernel路径。该预设使用 `skip_calibration` 静态筛选，不运行数据集探测，checkpoint自带normalization照常加载。量化实验保持 `--no-compile-model`。

`--dry-run` 检查文件并打印解析后的计划，不导入GPU运行栈、不加载模型或启动模拟器，也不生成结果目录。预览成功不等于模型、设备或checkpoint数值正确。入口拒绝已有输出目录，避免混合不同实验结果。

## 录制机器人完整执行视频

在运行命令后添加 `--record-video`，即可录制本次实际评估的所有episode。每条视频从episode初始画面开始，包含成功、失败或超时的终止画面；成功和失败都会保存，使用环境的主相机。

```bash
bash benchmarks/static/pi05_libero/run.sh \
  --schedule paper_async --overlap-actions 2 --quant none \
  --episodes 1 --record-video \
  --gpu 3 --output-dir runs/static/pi05_libero/video-001
```

默认关闭录制，也可显式传 `--no-record-video`。只想选取部分episode时，将 `--record-video` 换成 `--video-episodes-per-task 1`：每个参与评估的任务录前1条完整episode，录满后停止采帧。上述三个选项互斥。可选 `--video-fps 30` 设置播放帧率（默认30），不改变模拟器控制周期或论文计时参数。

视频自动保存到 `<output-dir>/videos/<task>/init_<初态编号>_<success|fail>.mp4`。`eval_results.json` 的 `video_paths`、`video_metadata` 记录路径、任务/初态、帧数及播放帧率；manifest也记录录制设置和成功写出的路径。编码使用环境中的imageio及MP4/FFmpeg后端；写入失败会令运行明确报错。录制和编码会增加评估墙钟耗时。

此前两组500-episode评估没有保存画面，现有结果保持原样，不补录或重跑。当前视频功能已覆盖CPU录制逻辑与入口测试，尚未重新运行模型验证实际闭环录像。

## 常用参数

直接在上述命令后追加参数，例如 `--suite libero_object --episodes 10 --batch-size 1 --seed 42`。默认设置如下：

`--episodes` 是本次评估的总episode预算，由兼容评测器分配到suite中的任务，并非每个任务各跑这么多次。少量episode不保证覆盖整个suite。

| 参数 | 默认值 / 含义 |
| --- | --- |
| `--suite` | `libero_object` |
| `--episodes` / `--batch-size` | `1` / `1` |
| `--seed` | `42` |
| `--n-action-steps` | `5`，每块实际执行动作数 |
| `--num-inference-steps` | `10`，flow推理步数 |
| `--compile-model` / `--no-compile-model` | 默认不编译；量化smoke要求不编译 |
| `--schedule` | `sync` 或 `paper_async` |
| `--overlap-actions` | 默认0，范围0到 `n-action-steps` |
| `--quant` | `none` 或 `w8a8-single-layer` |
| `--record-video` / `--no-record-video` | 录制所有实际评估episode / 关闭，默认关闭 |
| `--video-episodes-per-task` | 可选非负整数，每任务最多录前N条完整episode，0关闭 |
| `--video-fps` | 默认30，仅设置视频播放帧率 |
| `--paper-action-time-ms` | `1000/30`，仅用于论文周期估计 |
| `--paper-inference-time-ms` | 默认不提供，需同配置的推理profile |
| `--paper-inference-time-source` | 提供推理时间时必填，说明其来源与口径 |

常用实验参数的优先级为 **命令行 > `--config` JSON > case默认值**。可用 `bash benchmarks/static/pi05_libero/run.sh --help` 查看完整参数。

切换调度时同时写清 `--schedule` 和 `--overlap-actions`。例如，覆盖一份异步JSON为同步时使用 `--schedule sync --overlap-actions 0`；显式同步与正的重叠步数冲突会被拒绝。

## 论文异步与时间口径

`--overlap-actions` 对应论文的 $n'$，范围为 $0\le n'\le n$；$n$ 是实际执行的 `n_action_steps`，不是模型预测的50步。入口将n′映射到外部evaluator的 `async_delay`：控制步 $t$ 使用 $t-n'$ 的历史观测，策略在需要生成新chunk时接收该输入。这是论文B.1使用的静态异步近似，验收不要求建立并发worker。[论文§3.1与B.1](https://arxiv.org/html/2606.28529v2)

本case固定图像和state使用同一历史快照，历史不足时使用当前观测。这两项是case的工程约定，论文未逐项规定。`action_quant=1` 保证历史步数对应原始控制步；该动作合并参数与模型量化位宽无关。

[paper_async.py](paper_async.py) 与 [加速比工具](../../../tools/compare_speedups.py) 复用同一个周期公式：

$$
C(n')=T_{\mathrm{inf}}+nT_{\mathrm{act}}-\min(T_{\mathrm{inf}},n'T_{\mathrm{act}}).
$$

默认 `Tact=1000/30 ms` 取论文π0.5静态设置的30 Hz作为估算参数，不设置LIBERO物理步长、控制频率或视频帧率。提供 `--paper-inference-time-ms` 时必须同时提供非空 `--paper-inference-time-source`，说明同模型、精度、设备和输入口径的推理profile来源；入口记录声明，不验证来源文件。没有推理时延也可以运行任务，周期及残余等待时间保持 `null`，状态为 `requires_inference_profile`。

周期与残余等待属于 `paper_model` 时间域。外部evaluator的 `eval_s` 是实际串行评估墙钟耗时，不能代替该周期；公式残余等待也不等于观测年龄。相对固定baseline的加速比通过共享工具按 [加速比协议](../../../docs/protocols/speedup-metrics.md) 计算，不从episode步数推造模型调用数或任务加速比。真实并发执行可作为后续独立扩展，不是本次 `paper_async` 的前置条件。

## 已验证范围与外部依赖

此前正式入口完成过 `libero_object` 的同步单episode（157步成功）、同checkpoint单个文本层W8A8的同步单episode（138步成功），以及原精度 `paper_async`、n′=2的单episode（200步成功）。这些运行使用旧reset语义，仅证明加载/替换与闭环连通性，不能作为修正初态协议后的全量基线。加载审计通过，W8A8记录确认安装1个对应包装层；论文异步试跑未提供推理profile，不报告周期或加速比。量化与论文异步组合尚未闭环验证；W4A4仅完成单Linear功能小试。

当前运行记录 `environment_protocol=explicit_initial_state_v1`：[环境适配器](libero_adapter.py) 先重置底层环境，再应用指定初态和settling；只有显式reset推进初态列表，成功后的内部autoreset被抑制，终止环境在下一次显式reset前保持吸收态。[评估审计](evaluation_audit.py) 从环境读取实际任务/初态ID，逐episode记录并检查完整覆盖、重复及汇总一致性。

该协议下已完成 `libero_object` 的全量两组评估：每组10任务×50初态，同checkpoint、seed42、batch10、n=5、10个flow推理步、原精度且不编译。同步成功494/500（98.8%），论文异步n′=2成功464/500（92.8%）；两组实际初态均各出现一次，加载与覆盖审计通过。这是该固定设置下的任务结果，未采集对应推理profile，因此不报告论文加速比。启动上述实验时设置 `--episodes 500 --batch-size 10`，运行产物仍保存在调用者指定的本地目录。

全量验证目前限于上述原精度两组；量化检查仍为加载、替换和连通性smoke，不是全模型量化质量验收。用户须自行提供兼容资源，本仓库不下载或内置：

- VLASH sim checkout须支持 `runtime_stack=lerobot`、历史观测及入口所用参数。已跑通来源为 `d618e497eb6279fc9b95a78e1fa5635a51626a26` 工作树，含未提交改动，不能等同于原始上游 `sim/libero` 分支；每次运行记录实际source身份与dirty状态。
- 使用普通、任务匹配的LeRobot π0.5 LIBERO checkpoint，以及同源processor、normalization和tokenizer。指定路径不代替预处理兼容性验证。
- LIBERO环境、任务资产和配置目录须已就绪；Python环境须具备兼容的LeRobot、PyTorch、CUDA和模拟器依赖。本仓CPU开发依赖不足以运行模型。
- 量化源码与kernel依赖由调用者显式提供，入口不安装它们，也不声称全部kernel已公开可安装。

## 单 Linear 量化检查

[quant_smoke.py](quant_smoke.py) 不加载π0.5或模拟器，使用合成Linear检查一个量化backend。已加载上述env时可运行：

```bash
"$ROBOTICS_PI05_PYTHON" benchmarks/static/pi05_libero/quant_smoke.py \
  --quant-source "$ROBOTICS_QUANT_SOURCE" \
  --kernel-source "$ROBOTICS_KERNEL_SOURCE" \
  --scheme w8a8 --profile \
  --gpu 3 --output-dir runs/static/pi05_libero/linear-w8a8-001
```

检查W4A4时改为 `--scheme w4a4` 并选择新目录；每个scheme在独立进程运行。`--profile`记录各shape的实际CUDA kernel名称，不测时延。输出 `report.json` 中的 `functional_pass` 仅表示该功能检查通过，不表示已建立精度门槛或测得加速。

## 汇总实验控制步数

运行正常结束后自动生成 `episode-summary.md` 和 `episode-summary.json`，汇总全体实际步数、仅成功/仅失败步数、失败按预算惩罚的统计，以及成功部分对全体均值的贡献。汇总同时保留样本数、成功数和SR；`coverage.json` 中的 `max_primitive_steps` 提供本次控制步数预算。

也可以用CPU工具重新查看已有结果；无需启动模型或模拟器。从仓库根目录运行：

```bash
# 新结果自动读取预算；显示各任务明细
python tools/summarize_experiment.py --input runs/static/pi05_libero/sync-001 --per-task

# 旧libero_object结果缺少预算字段时，显式提供280步
python tools/summarize_experiment.py \
  --input runs/static/pi05_libero/sync-001 runs/static/pi05_libero/paper-async-001 \
  --max-steps 280 --per-task --output results/pi05-step-comparison.csv
```

`--input` 也接受 `episodes.jsonl` 文件，多个运行分别汇总。默认输出Markdown，可用 `--format json` 或 `--format csv`，也可由 `--output` 扩展名选择格式；目标文件须不存在。JSON始终包含各任务，Markdown/CSV以 `--per-task` 展开。不同任务预算可重复指定 `--task-max-steps TASK=LIMIT`，覆盖通用预算。未提供失败预算时惩罚项为null，不猜测观测最大值。

全体均值以总episode数为分母，成功均值以成功数为分母；跨任务分别按总样本数和成功数加权。零成功时成功均值为null；“成功步数总和/全体episode数”只是成功贡献。控制步数不等于时间或模型调用数，完整定义见 [统计口径](../../../docs/protocols/speedup-metrics.md#8-控制步数统计与实验汇总)。

未完成或失败的运行需显式加 `--allow-partial`，结果标为 `partial`；裸ledger缺少完成元数据时标为 `unverified`。完整状态还会核对伴随manifest、coverage与ledger的一致性；手动生成汇总不会将不完整运行改成完成。

## JSON各自做什么

普通运行使用上述命令行；JSON用于保留可复用设置和自动记录结果，不要求逐项手填。

| 类别 | 文件 | 用途 |
| --- | --- | --- |
| 输入默认值 / 预设 | [case.json](case.json)、[论文异步预设](paper-async.case.json)、[单层W8A8预设](w8a8-single-layer.case.json) | 保留可复用实验设置；高级用法可传 `--config PATH`，命令行继续覆盖 |
| 高级量化层映射 | [quant-profiles.json](quant-profiles.json) | 选择量化层；单层快捷选项自动使用，换层时才需要自定义 |
| 自动输出 | manifest、加载审计、评测结果等 | 运行后自动生成，无需手工创建或修改 |

自定义量化配置可使用 `--quant-ladder`、`--quant-selected-profile`、`--quant-profile`，并提供量化源码/kernel路径。入口拒绝错误或空的profile字段、重复profile名，记录实际安装的包装层，没有对应包装层时运行失败。包装层标记不替代实际CUDA kernel验证。[load_guard.py](load_guard.py) 按共享Parameter的tied-weight别名审计加载覆盖，不替代processor一致性、动作数值或任务能力验证。

每次实际运行在指定目录自动保存：

| 输出文件 | 内容 |
| --- | --- |
| `case-manifest.json` | 解析后的配置与指纹、外部source身份、checkpoint/processor/tokenizer身份、运行环境 |
| `paper-async.json` | 同步/论文异步约定、n′、历史/state规则、推理时延来源及周期估计；manifest中同时保存 |
| `checkpoint-load.json` | 加载覆盖、tied-weight别名与失败原因 |
| `eval_results.json` | 外部evaluator的评测结果 |
| `episodes.jsonl` | 每个实际任务/初态的成功标记、首次终止步数与环境seed |
| `coverage.json` | 已完成/预期episode数、各任务初态覆盖、控制步数预算 `max_primitive_steps` 及审计状态 |
| `episode-summary.md` / `episode-summary.json` | 正常结束时生成的控制步数统计；区分全体、成功、失败与失败预算惩罚 |
| `videos/<task>/*.mp4` | 启用录制时的完整episode视频；路径与帧信息同时进入评测结果 |
| `failure.json` | 实际执行失败时的错误与traceback |

`schemas/` 下的JSON是格式定义，也无需作为每次运行的输入。此入口的manifest使用独立 `pi05-libero-smoke-v1` 格式，不是通用run-manifest v1，不能直接用其schema校验；目前尚未生产通用逐action trace。加载开始前失败时不一定有 `checkpoint-load.json`，dry-run只打印计划。输出均为本地实验产物，默认不提交Git。

任务范围与共享交接写入Issue/PR；协议见 [组合实验](../../../docs/protocols/composable-experiments.md) 和 [模型接入](../../../docs/protocols/model-adapters.md)。
