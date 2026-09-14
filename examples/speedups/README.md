# 统一加速比：CPU 手算工具

从仓库根目录运行，无模型、GPU 或第三方运行依赖：

```bash
python tools/compare_speedups.py --input examples/speedups/synthetic.json
python tools/compare_speedups.py --input examples/speedups/synthetic.json --markdown
```

默认 stdout 为 JSON，`--json` 可显式指定。错误写 stderr，退出码为 2；成功为 0。本文件与样例用于解释计算，所有样例数字均为 synthetic，不代表论文实验结果。工具独立于 manifest/trace v1，不改变现有 schema 或 validator，也不运行模拟器。

## 一份输入代表一个可比较案例

提交一份 case 即表示输入作者声明：所有行采用同场景、同硬件、同控制设置和同统计范围。工具只能验证字段与计算，不能验证这一声明、数据真实性或两行是否使用同一批成功 seeds。`case_id` 应能定位共同实验条件；`timing_scope` 与 `clock_domain` 明确时间范围及时间域。不要把不同设备或不同范围的行混入一个 case。

`inference_time_ms` 是完整待比较推理范围的固定或代表性延迟标量，不是 microkernel speedup。若提供均值且延迟抖动跨越 overlap 阈值，`min(mean(I), L)` 不等于 `mean(min(I, L))`，计算结果仍只是该标量模型估计，不能称为实测平均 cycle。

| 字段 | 约束 |
| --- | --- |
| `format_version` | 字符串 `1.0` |
| `synthetic` | 必需 boolean；合成数据必须为 true |
| `case_id`, `timing_scope`, `clock_domain`, `baseline_id` | 非空字符串 |
| `n_actions` | 每周期执行动作数，正整数；所有行相同 |
| `action_time_ms` | 每个动作的固定执行时间，正有限数 |
| `runs` | 非空列表，每行 ID 唯一；baseline 必须存在且 overlap 为 0 |
| 每行 `id`, `inference_time_ms`, `overlap_actions` | ID 非空；推理时间正有限；overlap 为 0 到 n 的整数 |
| 每行 `trials`, `successes` | trials 正整数，successes 为 0 到 trials 的整数 |
| 每行 `successful_chunk_count_mean` | 可缺省/null；否则为成功样本的正有限 chunk 均值 |
| 每行 `successful_task_time_mean_ms` | 可缺省/null；否则为提供的成功任务时间正有限均值 |

bool 不作为数字接受；NaN/Infinity、零分母、非法 overlap、重复 ID 均拒绝。successes 为 0 时两个成功条件均值只能缺省或 null。计算中超出有限浮点表示范围也报错，不输出 0/Infinity 伪比值。

## 四类加速比分开报告

令 baseline 推理时间为 I0，第 i 行为 Ii，动作数为 n，每动作时间为 A，overlap 动作数为 n′。

```text
C0 = I0 + n*A
Ci = Ii + n*A - min(Ii, n′*A)
speedup_inference       = I0 / Ii
speedup_chunk_estimated = C0 / Ci
task_time_estimated_i  = Ni * Ci
speedup_task_estimated = (N0*C0) / (Ni*Ci)
speedup_task_success   = T0_success / Ti_success
latency_ratio_alpha   = Ii / I0
```

`estimate_model=paper_steady_state` 表示规则、稳态、等长动作块的标量模型。组合周期是对串行/重叠周期表达式的工程推广；本工具不声称它是论文直接给出的组合公式。它没有模拟首尾 pipeline、提前 terminal、有限 episode 或调度 jitter。

输出中的 `paper_steady_state` 保存周期、隐藏/剩余推理时间和 N×C 任务估计；`provided_summary` 保留输入作者提供的成功均值。后两种 task 指标不可混同：一个由 chunk 数和周期估计，另一个是提供的成功条件任务时间均值之比，不是逐 seed 加速比的平均，更不是包含失败的任务效率。

任一侧没有成功样本或缺少所需均值，对应 task ratio 为 null，并在 `missing_reasons` 写明原因。不从缺失任务时间反推数据。SR 和相对 baseline 的百分点变化始终并报；即使两行 SR 相同，成功子集仍可能不同。

## 样例的独立手算结果

| 配置 | I/ms | n′ | 周期估计/ms | 成功 chunk 均值 N | N×C/ms | 任务估计加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 100 | 0 | 300 | 10 | 3000 | 1 |
| quantization | 50 | 0 | 250 | 12 | 3000 | 1 |
| async | 100 | 3 | 240 | 12 | 2880 | 1.041667 |
| combined | 50 | 3 | 200 | 16 | 3200 | 0.9375 |

组合方案的推理快 2 倍、周期估计快 1.5 倍，但 N 增加后任务估计变慢。样例没有提供成功任务时间，因此 `speedup_task_success` 为 null，不能把 N×C 填进这个字段。

也不能一般性地相乘独立优化收益：若把量化推理时间改为 60 ms，独立周期加速比的乘积为 `(300/260)*(300/240)≈1.442308`，组合周期却仍为 200 ms，对应 1.5 倍。

验证命令：`python -m pytest -q tests/test_speedups.py`。测试使用独立手算、输入负例和 CLI smoke，不以该工具验证真实性能。
