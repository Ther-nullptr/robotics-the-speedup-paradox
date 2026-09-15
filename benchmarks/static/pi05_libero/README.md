# π0.5＋LIBERO 静态 case

[run.py](run.py) 是本 case 的正式启动入口，使用 [case.json](case.json) 固定评估设置，调用用户提供的兼容 VLASH sim evaluator，并选择原生 LeRobot π0.5 policy 和同源 processors。支持同步基线和《The Speedup Paradox》的静态 `paper_async` 抽象，可与量化配置组合；不使用 VLASH 微调 checkpoint，也不做预测状态替换。

正式 `run.py` 入口已完成 `libero_object` 的1个同步episode，157个控制步后成功。随后，同checkpoint仅将文本第0层down projection替换为W8A8，1个同步episode在138步后成功，运行记录确认安装了1个W8A8包装层。这些是加载、替换与闭环连通性smoke，不是完整任务集SR、性能对照或全模型W8A8质量验证。W4A4仅完成单Linear功能小试。

`paper_async` 已以原精度、`n′=2` 完成同case的1个episode，200个控制步后成功，加载审计通过；运行记录包含论文异步约定。该次未采集对应推理profile，周期仍为 `null`，不报告加速比。量化与论文异步组合尚未进行闭环验证。

## 运行前准备

用户自行提供以下外部资源，本仓库不下载或内置它们：

- 兼容的 VLASH sim checkout：必须支持 `runtime_stack=lerobot`、历史观测选项及当前入口所用评估参数。已跑通的本地来源为 `d618e497eb6279fc9b95a78e1fa5635a51626a26` 工作树，含未提交改动；不能将其等同于原始上游 `sim/libero` 分支。实际 source 身份与 dirty 状态由每次运行记录。
- 普通、任务匹配的 LeRobot π0.5 LIBERO checkpoint，以及同源 processor、normalization 资产和 tokenizer。tokenizer 参数只指定资源位置，仍须符合该 checkpoint 的预处理约定。
- 配置完成的 LIBERO 环境、资产和配置目录，以及兼容的 Python、LeRobot、PyTorch、CUDA 和模拟器依赖。实际运行使用已经具备这些依赖的 Python 环境；本仓库的 CPU 开发依赖不足以运行模型。
- 一张由调用者明确选择的 GPU。入口不默认使用某台机器的源码路径、模型路径或设备。

## 先预览，再运行

以下命令从仓库根目录执行。先将所有 `PATH/TO/...` 和 `GPU_INDEX_OR_UUID` 替换为自己的配置：

```bash
python benchmarks/static/pi05_libero/run.py \
  --sim-source PATH/TO/compatible-vlash-sim-checkout \
  --checkpoint PATH/TO/native-lerobot-pi05-libero-checkpoint \
  --tokenizer PATH/TO/matching-tokenizer \
  --libero-config-dir PATH/TO/libero-config \
  --output-dir artifacts/pi05-libero-smoke \
  --gpu GPU_INDEX_OR_UUID \
  --dry-run
```

`--dry-run` 预览解析后的配置与启动计划，不导入 GPU 运行栈、不加载模型、不启动模拟器；它不证明模型、设备或 checkpoint 数值正确。实际运行时去掉 `--dry-run`，并使用新的输出目录。入口拒绝已有结果目录，避免混合不同 case 或来源的结果。

`--config PATH` 可指定另一份 case JSON，默认使用本目录的 `case.json`。默认基线为：

| 设置 | 值 |
| --- | --- |
| suite / episodes / batch | `libero_object` / `1` / `1` |
| seed | `42` |
| n_action_steps / num_inference_steps | `5` / `10` |
| runtime_stack / compile_model | `lerobot` / `false` |
| schedule / overlap_actions | `sync` / `0` |
| action_quant | `1` |
| delay_state_with_observation | `true` |
| paper_action_time_ms | `1000 / 30`，仅用于论文周期估算 |
| paper_inference_time_ms / paper_inference_time_source | `null` / `null`，等待同配置的推理profile |
| quant_ladder | `none` |

## 论文异步抽象

在上述命令中改用新输出目录，并添加 `--config benchmarks/static/pi05_libero/paper-async.case.json`，即可启用 [论文异步配置](paper-async.case.json)：

```json
{
  "schedule": "paper_async",
  "overlap_actions": 2
}
```

`overlap_actions` 对应论文的 $n'$，范围为 $0\le n'\le n$；$n$ 是实际执行的 `n_action_steps`（默认5），不是模型预测的50步。入口将它映射到外部 evaluator 的 `async_delay`：控制步 $t$ 使用 $t-n'$ 的历史观测，策略在需要生成新chunk时接收该输入。这是论文B.1使用的静态异步近似，验收不要求建立并发worker。[论文§3.1与B.1](https://arxiv.org/html/2606.28529v2)

本 case 固定 `delay_state_with_observation=true`，图像和 state 使用同一历史快照；历史不足时使用当前观测。这两项是当前case的明确约定，论文未逐项规定。`action_quant=1` 保证历史步数对应原始控制步；这是动作合并参数，与模型量化位宽无关。旧配置 `async_delay` 仍作为 `overlap_actions` 的输入别名；两者冲突时拒绝运行。仅指定正的 `overlap_actions` 或旧别名时自动选择 `paper_async`，显式 `sync` 必须为0。

[paper_async.py](paper_async.py) 与 [加速比工具](../../../tools/compare_speedups.py) 复用同一个周期公式：

$$
C(n')=T_{\mathrm{inf}}+nT_{\mathrm{act}}-\min(T_{\mathrm{inf}},n'T_{\mathrm{act}}).
$$

默认 `paper_action_time_ms=1000/30` 取论文π0.5静态设置的30 Hz作为估算参数；它不设置LIBERO物理步长、控制频率或视频帧率。提供 `paper_inference_time_ms` 时必须同时填写非空 `paper_inference_time_source`，说明同模型、精度、设备及输入口径的推理profile来源；入口记录该声明，不验证来源文件。未提供推理时延时仍可评估任务，但周期及残余等待时间保持 `null`，状态为 `requires_inference_profile`。

周期与残余等待属于 `paper_model` 时间域。外部 evaluator 的 `eval_s` 是实际串行评估墙钟耗时，不能替代该周期；公式残余等待也不等于观测年龄。相对固定baseline的加速比仍通过共享工具按 [加速比协议](../../../docs/protocols/speedup-metrics.md) 计算；本入口不从episode步数推造模型调用数或任务加速比。

量化与 `paper_async` 分别配置。例如，在单层W8A8 case JSON 中加入上述两个字段，再提供相同量化依赖，即可组合运行；组合支持不代表其成功率或加速已验证。

## 单 Linear 量化 smoke

[quant_smoke.py](quant_smoke.py) 使用合成 Linear 输入，每次在独立进程中检查一种量化方案，不加载 π0.5 checkpoint 或模拟器。先预览：

```bash
python benchmarks/static/pi05_libero/quant_smoke.py \
  --quant-source PATH/TO/compatible-quant-source \
  --kernel-source PATH/TO/compatible-kernel-source \
  --scheme w8a8 \
  --output-dir artifacts/pi05-linear-w8a8-smoke \
  --gpu GPU_INDEX_OR_UUID \
  --dry-run
```

实际执行时去掉 `--dry-run`；检查 W4A4 时改为 `--scheme w4a4` 并选择另一新输出目录。可选 `--profile` 采集一次实际 CUDA kernel 名称，不能把它当成时延 benchmark。dry-run 仅用标准库检查路径并输出计划，不导入 GPU 运行栈、不生成产物。

运行后生成 `report.json`。已完成的 W8A8/W4A4 检查均为 synthetic Linear functional smoke；`functional_pass` 表示功能检查通过，不表示已建立精度门槛、测得加速或通过整模型质量验证。量化源码与 kernel 依赖均需用户自行提供，入口不安装它们，也不声称全部 kernel 已公开可安装。

## 整模型扩展与运行产物

整模型量化通过 case JSON 中的量化字段声明。`quant_ladder` 不为 `none` 时，必须提供 `--quant-source PATH`、`--kernel-source PATH`、`--quant-profile PATH`，设置有效的 `quant_selected_profile`，并保持 `compile_model=false`。入口读取已有量化 profile，不自动完成校准；支持这些参数不代表所选组合已经验证。先保留未量化基线，再分别验证加载、实际 kernel、数值与闭环结果。

可使用 [单层W8A8配置](w8a8-single-layer.case.json) 和 [profile](quant-profiles.json)。在前面的运行命令中选择新输出目录，并追加：

```text
--config benchmarks/static/pi05_libero/w8a8-single-layer.case.json
--quant-source PATH/TO/compatible-quant-source
--kernel-source PATH/TO/compatible-kernel-source
--quant-profile benchmarks/static/pi05_libero/quant-profiles.json
```

该例仅替换 `TXT.B00.mlp.down`，使用现有 `skip_calibration` 静态筛选路径，不运行数据集探测；checkpoint自带的normalization仍照常加载。入口拒绝错误/空的profile字段、重复profile名，并记录模型上实际安装的量化包装层；没有对应包装层时运行失败。这些标记不替代实际CUDA kernel验证。

[load_guard.py](load_guard.py) 审计 checkpoint 加载：按共享 Parameter 的 tied-weight 别名核对覆盖，拒绝未完整加载的模型进入评估。此检查不替代 processor 一致性、动作数值或任务能力验证。

正式 `run.py` 在指定输出目录保存：

| 文件 | 内容 |
| --- | --- |
| `case-manifest.json` | case 配置与指纹、外部 source 提交/dirty 身份、checkpoint/processor/tokenizer 文件身份和运行环境元数据 |
| `paper-async.json` | 同步或论文异步约定、$n'$、历史/state规则、推理时延来源及论文周期估计；manifest中同时保存 |
| `checkpoint-load.json` | checkpoint 加载覆盖、tied-weight 别名与失败原因 |
| `eval_results.json` | 外部 evaluator 的本次评估结果 |
| `failure.json` | 实际执行失败时的错误与 traceback |

`case-manifest.json` 使用独立的 `pi05-libero-smoke-v1` 格式，不是 `schemas/run-manifest.schema.json` 的通用 manifest v1，不能直接按该通用 schema 校验。当前入口记录上述来源、模型资源及运行元数据，尚未生产通用的逐 action trace。加载尚未开始便失败时，不一定产生 `checkpoint-load.json`；dry-run 只打印计划。输出均为本地实验产物，不作为默认提交内容。

真实并发执行可作为后续独立扩展，届时再接入请求、执行队列和实测时钟；它不是本次 `paper_async` 的前置条件。任务范围与共享交接写入 Issue/PR；协议见 [组合实验](../../../docs/protocols/composable-experiments.md) 和 [模型接入](../../../docs/protocols/model-adapters.md)。
