# π0.5＋LIBERO 静态 case

[run.py](run.py) 是本 case 的正式启动入口，使用 [case.json](case.json) 固定评估设置，调用用户提供的兼容 VLASH sim evaluator，并选择原生 LeRobot π0.5 policy 和同源 processors。当前先提供同步基线；不使用 VLASH 微调 checkpoint，也不做预测状态替换。

正式 `run.py` 入口已完成 `libero_object` 的1个episode，157个控制步后成功。随后，同checkpoint仅将文本第0层down projection替换为W8A8，1个episode在138步后成功，运行记录确认安装了1个W8A8包装层。这些是加载、替换与闭环连通性smoke，不是完整任务集SR、性能对照或全模型W8A8质量验证。W4A4仅完成单Linear功能小试；真实 `pure_async` 尚未实现。

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
| async_delay / action_quant | `0` / `1` |
| delay_state_with_observation | `true` |
| quant_ladder | `none` |

配置中的 `async_delay` 是历史观测步数，不是后台推理并发或毫秒延迟注入。若将其设为正数，图像和 state 一起使用历史样本；默认0表示不施加历史偏移。本入口固定要求 `delay_state_with_observation=true`、`action_quant=1`，不允许分别替换图像/state或合并多步动作。`action_quant` 是外部评估器的动作合并参数，不是权重量化位宽。

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
| `checkpoint-load.json` | checkpoint 加载覆盖、tied-weight 别名与失败原因 |
| `eval_results.json` | 外部 evaluator 的本次评估结果 |
| `failure.json` | 实际执行失败时的错误与 traceback |

`case-manifest.json` 使用独立的 `pi05-libero-smoke-v1` 格式，不是 `schemas/run-manifest.schema.json` 的通用 manifest v1，不能直接按该通用 schema 校验。当前入口记录上述来源、模型资源及运行元数据，尚未生产通用的逐 action trace。加载尚未开始便失败时，不一定产生 `checkpoint-load.json`；dry-run 只打印计划。输出均为本地实验产物，不作为默认提交内容。

后续 `pure_async` 应在本 case 已验证的 policy/processor 与环境接口上单独接入请求、执行队列和时钟，不从 `async_delay` 名称推断它已经存在。任务范围与共享交接写入 Issue/PR；协议见 [组合实验](../../../docs/protocols/composable-experiments.md) 和 [模型接入](../../../docs/protocols/model-adapters.md)。
