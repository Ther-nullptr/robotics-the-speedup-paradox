# 完整静态实验：手动运行指南

[full_evaluation.sh](../benchmarks/static/full_evaluation.sh) 将现有 case 入口组织成顺序执行的实验矩阵，不要求手写 JSON。一个命令只运行一个 case；模型、模拟器和资源仍使用该 case 的独立环境。脚本不会下载资源、安装依赖或修改模拟器。本次交付仅准备脚本，GPU 实验由使用者手动启动。

## 默认实验矩阵

| Case | 配置 | 每个配置的评估范围 | 总回合数 |
| --- | --- | --- | ---: |
| `pi05_libero` | 原始 BF16 × sync / paper_async | `libero_object`，10 任务共 500 个初态 | 1000 |
| `cosmos_libero` | 原始 BF16 / 优化 BF16 / INT8 / INT4 × sync / paper_async | `libero_object`，10 任务共 500 个初态 | 4000 |
| `cosmos_robocasa` | 同上 8 组 | 24 个已支持任务，每任务 50 个环境种子；固定 layout/style=1/1 | 9600 |

`paper_async` 默认 `n′=2`。这是历史观测抽象，不启动后台并发推理，也不把串行评估墙钟时间解释为异步执行收益。

LIBERO 的 500 是整个 suite 的总数，不是每个任务 500。入口按已有任务初态分配并审计实际覆盖；显式减少 `--episodes` 后属于子集实验。RoboCasa 的 50 是每任务回合数，种子为 `env_seed + episode_index`。其“完整”指完成声明的任务×种子矩阵，不表示穷举厨房场景或复现论文采样；layout/style 的兼容性最终由模拟器初始化检查。

π0.5 默认模型 seed=42、batch=10、每次执行5条动作、10次flow采样、不编译。Cosmos 默认模型 seed=195、环境 seed=0、单环境、每次执行16条动作、5次采样。不会将 Cosmos 配方套到 π0.5。两种模型的动作时长估计也沿用各自 case 的默认值，不更改物理步长。

## 配置环境

先按[环境指南](environment_setup.md)准备对应 case。驱动需要 Linux、Python 3.10+ 标准库（文件锁使用 `fcntl`）；子进程通过以下变量选择模型环境：

| Case | 子进程 Python | 路径模板 |
| --- | --- | --- |
| π0.5＋LIBERO | `ROBOTICS_PI05_PYTHON` | [paths.env.example](../benchmarks/static/pi05_libero/paths.env.example) |
| Cosmos＋LIBERO | `ROBOTICS_COSMOS_PYTHON` | [paths.env.example](../benchmarks/static/cosmos_libero/paths.env.example) |
| Cosmos＋RoboCasa | `ROBOTICS_COSMOS_ROBOCASA_PYTHON` | [paths.env.example](../benchmarks/static/cosmos_robocasa/paths.env.example) |

将模板复制到自己的 `.local/`，填写已有资源路径，运行前显式 `source`。如果在独立 worktree 中运行，原检出目录的 `.local/` 不会随 Git 自动复制；可以直接 `source /absolute/path/to/existing/case.env` 复用已有路径配置。脚本不会自动寻找其他人的资源，也不会自动加载 env 文件。没有 Python 变量时使用驱动自身解释器；也可以用 `--python /absolute/path/to/python` 覆盖。驱动解释器可通过 `ROBOTICS_DRIVER_PYTHON` 配置。

Cosmos 低精度运行需要可用的 CUDA 开发工具链和仓内 CUTLASS 头文件。已有性能配方验证于 RTX 6000 Ada；该设备可显式设置：

```bash
export CUDA_HOME=/path/to/cuda-12.8
export TORCH_CUDA_ARCH_LIST=8.9
```

其他 GPU 需要匹配自己的架构和工具链，不能直接照搬 `8.9`。首次低精度运行可能触发扩展编译。CPU preflight 不证明 CUDA 扩展、渲染或 checkpoint 推理可运行。

## 预览和预检

从当前分支的仓库根目录执行。下面的 GPU 0 是示例，实际运行前选择空闲 GPU；不要在同一张卡上同时启动多组矩阵。

```bash
source .local/pi05-libero.env

# 只显示两组完整命令，不读取 checkpoint、不创建输出目录。
bash benchmarks/static/full_evaluation.sh \
  --case pi05_libero --gpu 0 \
  --output-dir runs/full/pi05-object-001 --dry-run

# 只调用第一组的 CPU 资源检查；不启动模型、GPU 或模拟器。
bash benchmarks/static/full_evaluation.sh \
  --case pi05_libero --gpu 0 \
  --output-dir runs/full/pi05-object-001 --preflight
```

`--preflight` 复用已有 case 检查，可能读取大 checkpoint 计算哈希。它检查第一组代表性的本地资源与源码；每个实际子运行仍会执行自己的参数和资源检查。所有任务/场景能否初始化、所有指令是否命中 embedding 缓存，以及整数后端能否编译，需要实际运行验证。

## 正式运行

三个 case 分别启动。下面均为实际执行命令；不要同时复制执行到同一张 GPU。

```bash
# π0.5：500 回合同步 + 500 回合 paper_async n'=2。
source .local/pi05-libero.env
bash benchmarks/static/full_evaluation.sh \
  --case pi05_libero --gpu 0 \
  --output-dir runs/full/pi05-object-001
```

```bash
# Cosmos LIBERO：4 种配方 × 2 种调度，每组 500 回合。
source .local/cosmos-libero.env
bash benchmarks/static/full_evaluation.sh \
  --case cosmos_libero --gpu 0 \
  --output-dir runs/full/cosmos-object-001
```

```bash
# Cosmos RoboCasa：24 任务，每任务 50 回合，固定场景设置。
source .local/cosmos-robocasa.env
bash benchmarks/static/full_evaluation.sh \
  --case cosmos_robocasa --gpu 0 \
  --layout-id 1 --style-id 1 --episodes 50 \
  --output-dir runs/full/cosmos-robocasa-001
```

建议在 `tmux` 中运行，以免 SSH 断线中止实验。每个 cell 使用独立进程和输出目录，按顺序执行；Cosmos LIBERO 每组加载一次模型，RoboCasa 每任务每组加载一次。当前设计优先复用已验证的运行入口，未实现跨任务的常驻模型服务。

RoboCasa 每个任务先完成原始 BF16 同步组，其余七组自动通过 `--reference-run` 比对对应初态、场景 XML、观测和指令。LIBERO 使用固定任务初态与覆盖审计；该矩阵不会声称额外完成了 LIBERO 像素级初态比对。

## 先跑小规模检查或缩小矩阵

小规模运行使用新的目录，不能直接作为完整实验继续累加：

```bash
# π0.5：每组 10 回合，batch=1，两个调度均执行。
bash benchmarks/static/full_evaluation.sh \
  --case pi05_libero --gpu 0 --episodes 10 --batch-size 1 \
  --output-dir runs/full/pi05-smoke-001

# Cosmos LIBERO：仅原始 BF16 和 INT8，每组 10 回合。
bash benchmarks/static/full_evaluation.sh \
  --case cosmos_libero --gpu 0 --episodes 10 \
  --recipes original-bf16,int8 \
  --output-dir runs/full/cosmos-smoke-001

# RoboCasa：一个任务，每组一个种子。
bash benchmarks/static/full_evaluation.sh \
  --case cosmos_robocasa --gpu 0 --tasks TurnOffMicrowave --episodes 1 \
  --output-dir runs/full/robocasa-smoke-001
```

运行哪个 case，先加载哪个 env 文件。`--recipes` 必须包含 `original-bf16`；同步基线始终保留。π0.5 当前矩阵仅支持原始 BF16，量化实验不在这个预设中。改变 `--overlap-actions`、`--seed`、采样步数、动作执行长度或场景参数时使用新目录。

Cosmos 优化 BF16 包含 `modulation`、`gated_residual`、`cuda_graph`、`vae_norm_fusion`、`vae_silu_fusion`、`vae_condition_prefix`。INT8/INT4 在相同共享优化上增加 `shared_quant`、`activation_quant_fusion`、`integer_pack_reuse` 和 `modulation_quant`，默认整数 tactic=1。原始 BF16 不开启这些开关。配方与数值边界见 [Cosmos 量化指南](cosmos-quantization.md)。

## 输出、视频与中断恢复

```text
runs/full/<experiment>/
├── matrix.json                         # 完整命令、代码身份和各组状态
├── matrix-summary.md / .csv / .json    # 已完成配置的汇总
└── <suite-or-task>/<recipe>/<schedule>/
    ├── attempt-001.launcher.log         # 从进程启动开始记录 stdout/stderr
    ├── attempt-001/
    │   ├── case-manifest.json           # case 参数、资源身份、运行环境
    │   ├── episodes.jsonl / coverage.json
    │   ├── episode-summary.md / .json
    │   ├── requests.jsonl              # Cosmos 的请求记录
    │   ├── initializations/            # RoboCasa 的初始化记录
    │   ├── videos/                     # 开启录制时保存
    │   └── failure.json                # case 执行异常时保存
    └── attempt-002/                    # 失败重试，保留之前产物
```

终端打印当前组和日志路径。查看某组进度：

```bash
tail -f runs/full/pi05-object-001/libero_object/original-bf16/paper_async/attempt-001.launcher.log
```

加 `--record-video` 可以保存所有实际执行 episode 的完整 MP4，包括初始和终止帧。默认关闭，录制会增加耗时与磁盘占用；这不会更改控制频率。所有组使用相同视频设置。无需为了补视频重复已经完成的实验。

发生进程错误、初始化不匹配或统计校验失败时，矩阵停止并返回非零退出码，异常不记为策略失败。恢复时使用**原命令、原参数**并添加 `--resume`：

```bash
source .local/pi05-libero.env
bash benchmarks/static/full_evaluation.sh \
  --case pi05_libero --gpu 0 \
  --output-dir runs/full/pi05-object-001 --resume
```

恢复以 cell 为单位：完成且通过重新审计的组会跳过，未完成组在新 attempt 中从头运行，不能从中间 episode 接着跑。失败 attempt 不进入汇总。完成产物损坏时明确报错，不默默跳过。代码、资源内容或实验参数发生变化时拒绝续跑；恢复原设置，或使用新目录。运行过程中不要切换代码、替换资源或升级 case 环境。同一个输出目录只允许一个驱动进程；非阻塞文件锁会拒绝重复启动或运行中的重新汇总。锁位于输出目录旁的 `<experiment>.lock`，进程退出后自动释放；不要在运行中删除它。

## 统计口径

每个 cell 正常结束时复用原 case 的自动统计。矩阵按 `recipe × schedule` 汇总；RoboCasa 跨任务先汇总实际 episode，再计算加权均值。每项给出：

- Episodes、Successes、Success rate（比例 0～1）。
- Failure-budget total/mean steps：成功用实际步数，失败用该任务预算。
- Success-only total/mean steps：只使用成功回合；零成功时均值为空。

汇总检查配置间的任务/初态/环境 seed 集合。某配置尚有缺失 cell 时，只显示已完成组数，其总体成功率和均值留空，不把任务子集标成全量结果。预算不同的任务分别使用自己的预算；不统一替换成500步。

无需启动模型即可重新汇总：

```bash
python benchmarks/static/full_evaluation.py \
  --case pi05_libero \
  --output-dir runs/full/pi05-object-001 --summarize-only
```

此模式不需要 `--gpu`，不会访问设备。矩阵未全部完成时仍写出明确标记的进度汇总，并返回非零退出码。单组的逐任务明细仍可用：

```bash
python tools/summarize_experiment.py \
  --input runs/full/pi05-object-001/libero_object/original-bf16/sync/attempt-001 \
  --per-task
```

## 加速比与验证边界

本入口先提供闭环质量与控制步数矩阵，不把 `run.log` 的运行耗时、视频编码耗时或步数比值自动标成论文加速比。量化的完整调用时延需要单独使用[推理 benchmark](../benchmarks/inference/README.md)，在同模型、配方、采样步数、GPU和输入口径下测量。随后按[加速比协议](protocols/speedup-metrics.md)关联质量结果与周期模型。

Cosmos INT8/INT4 完整任务质量仍是待运行实验，不能根据脚本完成或 CPU 检查通过就宣称已验证。公开 CPU CI 只检查计划、统计和执行管理，不下载模型、不运行模拟器，也不调用实验室 GPU。
