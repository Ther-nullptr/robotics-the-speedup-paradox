# KINETIX 历史硬件推理延迟

本页固定记录 `dynamic-experiments` 的 **2026-08-11** 批次：同一 RTC flow 网络架构在四个硬件／功耗配置上的完整同步动作生成延迟。后续查询这批历史数据，以本页及[完整精度 CSV](data/kinetix-hardware-latencies-20260811.csv)为准。

按用户约定，本档案仅维护 RTX 6000 Ada、RTX 3090、AGX Orin 15W 和 AGX Orin 30W；AGX Orin 50W 不纳入该档案、后续汇总或默认延迟映射。本次整理未重新运行 GPU 测量。

## 实测数据

单位为 **毫秒／次完整 `policy.action` 调用**。每次生成 8 条动作，N 是 flow 采样迭代次数。表中保留三位小数；CSV 保留原始精度以及三个独立进程的平均值。

| 硬件／功耗档 | Profile ID | N=1 | N=2 | N=3 | N=4 | N=5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 6000 Ada | `local_ada` | 1.373 | 2.605 | 4.041 | 5.127 | 6.898 |
| RTX 3090 | `rtx3090` | 1.853 | 3.538 | 5.232 | 7.074 | 8.806 |
| AGX Orin 15W | `agx15` | 11.349 | 22.744 | 32.682 | 46.159 | 57.634 |
| AGX Orin 30W | `agx30` | 7.465 | 14.343 | 21.938 | 29.532 | 36.738 |

这些值是测量快照中的观测值。旧项目另外拟合过线性时延曲线；本表保留测量点，使用时应明确选择测量点还是拟合曲线。

## 模型与计时口径

| 项目 | 此批次设置 |
| --- | --- |
| 模型 | RTC MLP-Mixer，`teacher-1s` 架构；4 个 Mixer block，channel=256，channel hidden=512，token hidden=64 |
| 输入／输出 | 679 维观测、6 维动作、预测 horizon=8、batch=1 |
| 权重 | **同架构随机初始化权重**，用于架构和运行时性能测量 |
| 后端／精度 | PyTorch eager，FP16，`method=naive`，禁用梯度 |
| 优化开关 | 无量化、无 CUDA Graph、无 conditioning 预计算、无 native-eager runner |
| 每个 N 的重复 | 3 个独立进程；每进程预热 50 次、稳态计时 200 次 |
| 汇总 | 先分别求每进程 200 次调用的均值，再取 3 个均值的中位数 |

原 benchmark 在计时前准备 GPU 观测；每次调用前同步 CUDA，再启动 `perf_counter`，调用完整 `policy.action`，等待末尾 CUDA 同步完成后停止计时。因此数值包含调用区间内的 Python 调度和 GPU 完成等待，并非 kernel 时间之和。模型初始化、首轮启动与预热、仿真步进、录像和结果文件输出不计入表中稳态时间；返回的是 GPU 动作张量，不能与包含输出回传的其他 API 计时直接混用。

统计公式为：

```text
latency_ms = median(process_mean_0_ms, process_mean_1_ms, process_mean_2_ms)
```

它不是把 600 次调用合并后直接求均值，也不是 600 次调用的中位数。CSV 中三个进程均值可用于独立复算。

## 软件栈与来源

| Profile ID | 原记录的 PyTorch 版本 | 原记录的时钟／功耗配置 | 快照文件名 |
| --- | --- | --- | --- |
| `local_ada` | `2.9.1+cu128` | stock dynamic clocks | `local_ada-flow-step-latency-snapshot.json` |
| `rtx3090` | `2.5.0+cu124` | stock dynamic clocks | `rtx3090-flow-step-latency-snapshot.json` |
| `agx15` | `2.3.0` | `MODE_15W`、`jetson_clocks` | `agx15-flow-step-latency-snapshot.json` |
| `agx30` | `2.3.0` | `MODE_30W`、`jetson_clocks` | `agx30-flow-step-latency-snapshot.json` |

来源是旧项目的 `kinetix-native-delay-clock` Git 工作区。每份快照记录 N=1～5 的汇总及对应 `flowN-repeat0/1/2.json` 文件名。本次逐项核对了 **4 份快照、60 份原始计时 JSON**：确认精度、batch、N、预热／测量次数、优化开关和随机初始化说明，并从原始进程均值重新计算中位数。

CSV 的 `snapshot_file` 和 `snapshot_sha256` 保留快照身份；`process_mean_*_ms` 保留复算所需的小型摘要。原始日志、机器路径与 checkpoint 不随文档提交，读取本档案不需要访问旧工作区。

## 使用约定

- 将这批数据标为“历史 PyTorch FP16 架构／运行时 profile”。它不证明任务成功率，也不是本仓当前 JAX FP32、已训练 checkpoint 的模型推理实测。
- 各设备的软件版本不同，跨设备差异反映对应软硬件配置的整体表现，不能全部归因为硬件本身。
- 可以选定一档硬件，将 N 对应的 `latency_ms` 作为明确声明的虚拟延迟场景。完整 CSV 包含多个硬件的重复 N；用于单一 `--latency-profile` 时，应先筛选一个 `profile_id`，只导出 `flow_steps,latency_ms` 两列。映射与原生物理时间的关系见 [KINETIX 入口说明](../benchmarks/dynamic/kinetix/README.md#旧映射与硬件profile)。
- 注入这些延迟不会认证当前模型已经在目标硬件上测量过；真实硬件性能结论需要匹配模型、精度、后端和计时边界的新测量。
- 后续回答历史硬件延迟问题时，先读取本页和 CSV；只查询已有数据时不自动启动新压测。

## English summary

This archive retains four historical profiles from 2026-08-11: RTX 6000 Ada, RTX 3090, AGX Orin 15W and AGX Orin 30W. Measurements use a randomly initialized RTC architecture, PyTorch eager FP16, batch one and complete synchronous action generation. Each reported value is the median of three independent process means, with 50 warmup and 200 timed calls per process. The CSV contains the full-precision values, process means, framework versions and snapshot hashes. These records may define explicit virtual-delay scenarios; they do not measure the current trained JAX policy or isolate hardware-only effects.
