# Cosmos Policy＋RoboCasa 单环境实验

[run.sh](run.sh) 运行一个固定任务和厨房布局，每次只推进一个环境。模型由共享 [CosmosEngine](../../../src/robotics_bench/engines/cosmos.py) 的 `suite=robocasa` 分支加载，环境由 [RoboCasa adapter](../../../src/robotics_bench/simulators/robocasa.py) 管理，同步和论文历史观测异步复用 [static_runner](../../../src/robotics_bench/protocols/static_runner.py)。

策略输入是左、右外部相机及腕部相机的原始 RGB，加上9维 proprio。模型预测 H=32 个7维动作，默认执行前 n=16 个再请求下一块。`--quant` 仅支持 `none`，模型采用原生 BF16 推理。此入口提供动作生成，不启用 best-of-N 搜索或额外规划模型。

## 环境和资源

CPU工具与各case环境的安装边界、版本表和故障排查见 [环境指南](../../../docs/environment_setup.md)。

使用 [Cosmos 官方说明](https://github.com/NVlabs/cosmos-policy/blob/main/ROBOCASA.md) 指定的 [robocasa-cosmos-policy fork](https://github.com/moojink/robocasa-cosmos-policy)。已验证源码为 [edd9a328b3ec98050f42d194c1419307a79c4d87](https://github.com/moojink/robocasa-cosmos-policy/commit/edd9a328b3ec98050f42d194c1419307a79c4d87)，配合 robosuite 1.5.1、MuJoCo 3.2.6、Python 3.10 和 PyTorch 2.7.0+cu128。环境与 LIBERO 所需 robosuite 1.4.0 分开配置；一个 Python 进程只加载一种 Cosmos case，避免原生全局平台常量串用。

所需文件：RoboCasa 专用 Policy `.pt`、`robocasa_dataset_statistics.json`、`robocasa_t5_embeddings.pkl`、配套 VAE，以及 Cosmos 源码内的 `robocasa_controller_configs.pkl`。提供完整路径模板 [paths.env.example](paths.env.example)，也可以用资源工具下载或复用已有文件：

```bash
python -m pip install -r tools/requirements-download.txt
python tools/prepare_resources.py --case cosmos_robocasa \
  --source /path/to/cosmos-policy \
  --robocasa-source /path/to/robocasa-cosmos-policy \
  --python /path/to/robocasa-env/bin/python
source ~/.cache/robotics/env/cosmos_robocasa.env
```

默认 checkpoint 和 VAE 位于 `~/.cache/robotics/hub/`，可以通过 `--root` 更改。`--reuse policy=DIR --reuse vae=DIR` 复用本地资源；需要训练/校准轨迹时才加 `--with-dataset`，数据集是 `nvidia/RoboCasa-Cosmos-Policy`。模型资源工具不安装运行环境或下载厨房资产。

厨房资产使用兼容 fork 的下载脚本，放在该源码目录的 `robocasa/models/assets/` 下。应在可写的独立源码目录运行；源码和资产不放进本仓 Git：

```bash
"$ROBOTICS_COSMOS_ROBOCASA_PYTHON" \
  "$ROBOTICS_ROBOCASA_SOURCE/robocasa/scripts/download_kitchen_assets.py"
```

该命令要求 fork 已在所选环境安装。运行入口会显式绑定 `--robocasa-source`；模型和环境路径可全部通过 CLI 覆盖环境变量。资产预检检查场景、固定设施、纹理及 Objaverse 目录，完整 mesh/texture 可加载性由环境初始化验证。本机已有 packages 的复用已验证，完整新机依赖安装尚未验证。

## 启动

从本仓库根目录运行；每次实验使用新输出目录：

```bash
# CPU preflight: no model, simulator, GPU or output directory is created.
bash benchmarks/static/cosmos_robocasa/run.sh \
  --task TurnOffMicrowave --layout-id 1 --style-id 1 \
  --episodes 1 --schedule sync \
  --output-dir runs/static/cosmos_robocasa/sync-001 --dry-run

# Synchronous baseline, with complete three-view video.
bash benchmarks/static/cosmos_robocasa/run.sh \
  --task TurnOffMicrowave --layout-id 1 --style-id 1 --env-seed 0 \
  --episodes 1 --schedule sync --record-video --gpu 3 \
  --output-dir runs/static/cosmos_robocasa/sync-001

# Paper async n-prime=2; verify that initialization matches the baseline.
bash benchmarks/static/cosmos_robocasa/run.sh \
  --task TurnOffMicrowave --layout-id 1 --style-id 1 --env-seed 0 \
  --episodes 1 --schedule paper_async --overlap-actions 2 --record-video --gpu 3 \
  --reference-run runs/static/cosmos_robocasa/sync-001 \
  --output-dir runs/static/cosmos_robocasa/async-001
```

`--task` 接受该 checkpoint 对应的24个旧版任务名。入口不把不同任务自动映射到新版名称；当前 GPU 验证限于 `TurnOffMicrowave`。一次命令选择一个任务和一个 layout/style 对；`--episodes N` 在该组合运行 N 回合，环境种子依次为 `env_seed + episode_index`。这套种子协议不声称复现原论文全量采样。

`--max-steps` 默认使用任务预算，例如 `TurnOffMicrowave` 为500。`--seed` 默认195，控制模型采样；`--n-action-steps` 默认16，允许1..32；去噪步数 `--num-inference-steps` 默认5。上述配置先形成明确的 episode 清单，再执行模型和模拟器。

## 初始化、异步与动作

每个 episode 创建新的 seeded 环境，调用一次公开 reset，在 reset 后读取 `get_ep_meta()["lang"]`。随后执行10步 settling，记为准备阶段；首次策略控制从第0步计数。模型加载一次，预计算 T5 缓存保留在 CPU，根据实际指令取对应 embedding；缺少指令缓存时明确失败，不在推理中联网下载或临时加载 T5 模型。

7维策略动作只在确认原生 `HYBRID_MOBILE_BASE` 控制器、机械臂/夹爪索引和剩余底盘/躯干区间后映射到12维动作，附加 `[0,0,0,0,-1]`。控制器不匹配则报错。成功使用原生 `_check_success()` 判断，真实完成的 `env.step` 才计入控制步数。

`paper_async` 在每次请求时选择 `t−n′` 的三相机和 proprio 快照；历史不足使用当前观测，`0 ≤ n′ ≤ n`。这是串行环境推进中的论文静态抽象，不是后台推理线程。`--paper-action-time-ms` 默认50，仅作为声明的周期估计参数，环境实际控制步长单独记录；没有提供同配置 `--paper-inference-time-ms` 和来源时不生成加速比。

`--reference-run` 要求对应运行已完成、任务及布局/种子清单相同，并在首次推理前比较初始物理状态、场景 XML、三相机观测及指令指纹。起点不同会使运行失败，不会算作策略失败。保存的状态用于审计，当前没有通用跨版本状态恢复入口。

## 输出与验证范围

| 文件 | 内容 |
| --- | --- |
| `run.log` | 自动保存控制台输出 |
| `case-manifest.json` / `checkpoint-load.json` | 源码、资源身份、参数及真实模型加载审计 |
| `initializations/000000/` | 本回合的 `state.npy`、`model.xml`、场景元数据与指纹 |
| `requests.jsonl` | 请求时的控制步、观测步、实际历史偏移 |
| `episodes.jsonl` / `coverage.json` | 成功、控制步数、模型调用数、预算和覆盖记录 |
| `episode-summary.md` / `.json` | 成功率、全体失败预算惩罚口径和仅成功步数统计 |
| `videos/TASK/` | 可选三视角横排 MP4，包含初始和终止帧 |
| `failure.json` | 初始化、模型或其他运行异常；不伪装成有效失败 episode |

视频默认关闭；`--record-video` 记录所有实际评估回合，`--no-record-video` 关闭。分辨率672×224，默认30 FPS 播放，播放帧率不改变仿真控制频率。

已在 RTX 6000 Ada 上完成同一任务、layout/style=1/1、环境种子0、模型种子195、H=32/n=16/5步采样的两次 GPU 闭环：同步277步成功、18次请求；论文异步 n′=2 在292步成功、19次请求。加载与覆盖审计通过，异步初态比对通过，请求偏移为首轮0、其后2；视频分别为278和293帧。这些是单回合连通性结果，尚无24任务全量结果或模型推理加速比。

本入口也接受 `--model-runtime owned|native`、`--enable`、`--precision` 与
`--integer-tactic`，量化范围固定为 `--quant-scope dit`。默认优化关闭。
与 LIBERO 的 kernel/模型代码共享不代表两个场景的性能或任务质量可以互相替代；
RoboCasa 的新优化配置需要单独验证。推理工具见 [inference benchmark](../../inference/README.md)。

固定输入计时支持 `--case cosmos_robocasa`，使用RoboCasa专用资源变量和H=32；
可通过 `benchmarks/inference/capture_robocasa.py` 捕获真实三相机观测，再比较
BF16、INT8、INT4以及独立卷积、VAE逐元素融合、DiT attention后端开关。
这些开关默认关闭，完整调用计时不含模拟器推进；模型任务质量需单独闭环验证。
