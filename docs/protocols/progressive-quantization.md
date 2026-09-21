# 渐进式整数精度协议

本协议参考 [The Speedup Paradox](https://arxiv.org/abs/2606.28529) 附录B.1、Table5的
Cosmos层选择规则。当前是本仓BF16基线上的实现，不复用论文的FP16时延或加速比。
π0.5的固定INT4/INT8使用独立scope，不套用含cross-attention的Cosmos分层规则。
当前量化调优优先Cosmos；π0.5先采用text/LLM scope，暂缓diffusion/action expert。

## 固定的格式与候选集合

候选为28个DiT block中的280个Linear：self-attention和cross-attention的Q/K/V/out，
以及MLP的layer1/layer2。early为B0–B9，mid为B10–B19，late为B20–B27。
块数、模块名或候选初始BF16 dtype不匹配时拒绝执行该协议，不静默缩减覆盖范围。

W8A8和W4A4分别使用signed整数权重与激活、逐输出通道权重scale、逐行动态激活scale、
INT32累加和BF16输出。没有额外smoothing、rotation或微调。FP32敏感模块、VAE和
其他非候选模块保留原始精度。整数源码、打包和固定CUTLASS头文件均在本仓维护。

| 档位 | 本档新增W4A4位置；之前的位置保留 | W4A4 / 候选Linear |
| --- | --- | ---: |
| `w8a8` / tier0 | 无，全部候选使用W8A8 | 0 / 280 |
| `w4-t1` | ca.out，late | 8 / 280 |
| `w4-t2` | mlp.layer1，early | 18 / 280 |
| `w4-t3` | ca.out，early | 28 / 280 |
| `w4-t4` | mlp.layer1，late | 36 / 280 |
| `w4-t5` | mlp.layer1，mid | 46 / 280 |
| `w4-t6` | ca.out，mid | 56 / 280 |
| `w4-t7` | mlp.layer2，all | 84 / 280 |
| `w4-t8` | sa.q/k/out，all | 168 / 280 |
| `w4-t9` | sa.v、ca.q，all | 224 / 280 |
| `w4-t10` | ca.k/v，all | 280 / 280 |

实际运行记录每个模块命中的backend；同一个QKV组出现不同位宽时分别准备输入，
合并投影只作用于格式相同且输入相同的子组。不能把INT8的packed activation传给INT4。

## 完整推理调用对照

从无优化BF16、共享优化相同的BF16开始，再测量化档位。固定checkpoint、输入、
seed、采样步数、设备和计时边界；加载、编译、打包权重、CUDA Graph捕获与warmup单列。
每行使用本轮实测baseline直接相除，不跨轮乘加速比。默认每种配置至少3次warmup，
重复采样取中位数。低精度动作偏差记录为测量结果，不作为停止性能测试的门槛；
非有限动作或运行错误仍按基础设施错误处理。

在已有Cosmos环境和资源变量下运行，`--input`来自本仓观测捕获工具或明确声明的fixture：

```bash
python benchmarks/inference/run.py --case cosmos_libero \
  --input /path/to/observation.npz --task 'the matching task instruction' \
  --gpu 0 --steps 5 --integer-tactic 1 --progressive-sweep \
  --enable modulation --enable gated_residual --enable cuda_graph \
  --output-dir runs/optimization/cosmos/tier-sweep-001 \
  --profile --profile-skill /path/to/profiler-visualizer \
  --render-python /path/to/render-env/bin/python
```

`--progressive-tiers 0 1 3 7 10`可选择子集；单次case启动使用
`--precision int8 --quant-scope dit --quant-tier N`。位宽越低不保证时延单调下降。
Table5采用LIBERO五步推理计时；附录静态任务默认Cosmos为一步。两种设置不能混合
计算加速比。本仓默认沿用已验证的五步设置，需要一步实验时所有对照统一传`--steps 1`。

## 闭环与任务时间

小规模的同步LIBERO对照入口同时支持固定精度和选定档位：

```bash
# 固定BF16 / 优化BF16 / INT8 / INT4
python benchmarks/inference/evaluate_cosmos.py --gpu 0 \
  --suite libero_object --task-id 0 --initial-states 0 1 --steps 5 \
  --output-dir runs/optimization/cosmos/pilot-fixed-001

# 在相同初态上比较渐进档位
python benchmarks/inference/evaluate_cosmos.py --gpu 0 \
  --suite libero_object --task-id 0 --initial-states 0 1 --steps 5 \
  --tiers 0 1 3 7 10 --output-dir runs/optimization/cosmos/pilot-tiers-001
```

这是小样本pilot，不是全量任务集成功率。每档先warmup与计时，再重置相同初态，
统计从策略开始到成功或预算耗尽的控制步数。reset/settling和warmup不计入任务时长。
每个档位的episode ledger独立保存，不能把不同档位的episode混成一个样本池。

输出分别包含：成功率、全体失败按预算计入的步数、仅成功步数、实测policy延迟、
仿真器宿主墙钟任务时间，以及论文模型下的任务时间估计。同步的有限回合估计为：

$$\widehat T_e=\sum_{r\in e} I_r+b_eT_{act}.$$

其中`I_r`是该回合逐请求实测的完整推理时间，`b_e`为实际执行动作数，默认声明
`Tact=50ms`。这个公式处理最后一个未完全执行的动作块；它是工程上的有限回合估计，
不是把仿真宿主运行时间称作物理任务时间，也不修改模拟器physics dt。

成功条件任务加速比按各档成功样本的均值之比计算；任一侧没有成功样本时为null。
总体预算惩罚时间另列：保留已发生的推理和动作，仅对提前终止后缺失的动作/请求按
声明预算与本档代表推理时延补齐；已跑满预算的失败回合不替换其已测耗时。
完整定义见 [统一加速比口径](speedup-metrics.md)。

每轮保留原始采样、按层精度映射、动作对照、profile、绘图与任务统计。可视化由外部
profile-visualizer读取实测ledger生成，产物留在被Git忽略的`runs/`。更大动作误差的
配置仍可继续测量，但不能为零成功配置生成虚假的成功条件任务加速比。
