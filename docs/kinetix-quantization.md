# KINETIX量化研发

量化研发在 `feat/kinetix-quantization` 分支进行，基于已合入main的 [latency／flow-step基础设施](kinetix-latency-study.md)。原有512种子延迟实验继续使用其固定源码快照。

## 当前阶段

已实现独立的配方描述与选层规则，可在CPU上检查配置。Torch/native FP16模型、实际W4A16／W4A4 CUDA执行和GPU计时尚待迁入；现有 `run.py` 仍使用JAX FP32。配方的 `execution_available=false` 表示它尚不是可运行的量化模型。

```bash
python benchmarks/dynamic/kinetix/quantization_recipes.py \
  --family ptmix --layers 0,2,4,6,8 --flow-steps 5
```

| 配方 | 选层粒度 | 权重scale | 激活 |
| --- | --- | --- | --- |
| `fp16` | 不量化 | 无 | FP16 |
| `cmix` | 0～8个通道Linear | 每输出通道 | FP16 |
| `ptmix` | 0～8个通道Linear | 每张权重矩阵 | FP16 |
| `a4mix` | 0／2／4／6／8个通道Linear，整块选择 | 每输出通道 | 每块静态A4，需要独立校准 |

顺序为block3→2→1→0，每块先 `channel_mix_in` 后 `channel_mix_out`。CMix／PTMix可以只选择块内一个Linear，A4Mix必须同时选择两个。未选择的通道Linear保持FP16；时间条件、投影等模块不属于当前选层范围。`requested_modules` 是目标覆盖范围，实际内核覆盖必须由后续执行器单独报告。

这里PTMix指“部分W4、其余FP16”的后期定义，不采用历史上“全部W4、改变scale颗粒度”的同名定义。零量化层的配方统一归为FP16基线。`recipe_id` 同时包含flow步数和配方，便于之后为每个硬件绑定完整模型延迟。

## 研发顺序

1. 将等价Torch策略与共享优化的native FP16执行路径迁入本仓，读取现有原始checkpoint，验证权重、观测、动作块和随机数口径。
2. 迁入真实packed W4A16内核与初始化pack、workspace复用、完整调用计时；保留精度、scale、布局版本与fallback证据。
3. 固定N=5，先验证CMix／PTMix覆盖，再在fine协议和选定任务上补齐每单元512个配对种子的量化质量评估。
4. 独立校准A4激活scale后，再接入W4A4。最后研究步数与量化组合，分别记录两种变化。

以同样开启通用优化的FP16作为量化增量基线，测量输入就绪至完整动作块就绪的模型调用。查询旧结果不会自动触发新压测。GPU性能采集使用空闲设备，保留用户指定的空卡。

## 来源与边界

方法参考是历史 `dynamic-experiments` 中的 `pure_tiny_stage_masks.py`、等价 `rtc_torch_policy.py`、W4A16 runtime和后期A4Mix校准实现；本阶段重新实现配方定义，没有导入旧工程。实际源码迁移时再记录对应文件、版本、许可与修改边界。

已有W4A16的部分200种子任务结果采用原生native-blend；后期校准W4A4的32种子比较仍使用1/2400秒物理子步。它们不能合并成当前fine协议的统一结果。新研发保持原生物理步长，不恢复物理细分实验；校准seed与评估seed分开，硬件档位沿用已声明的四档。

## English summary

This development branch currently defines quantization recipes and requested module coverage only. Actual Torch/native FP16 and low-bit GPU backends are not integrated yet. CMix and corrected PTMix select individual channel Linear layers; calibrated A4Mix selects complete blocks. The next stages port the execution backend and validate matched model latency and paired task quality under unchanged fine simulation.
