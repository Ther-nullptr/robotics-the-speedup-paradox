# Cosmos-Policy＋LIBERO 静态 case

[run.sh](run.sh) 调用本仓库的 [run.py](run.py)，每次只推进一个原生 LIBERO 环境。模型加载与动作推理由 [CosmosEngine](../../../src/robotics_bench/engines/cosmos.py) 负责，[LIBERO adapter](../../../src/robotics_bench/simulators/libero.py) 负责任务、初态和环境生命周期，[静态 runner](../../../src/robotics_bench/protocols/static_runner.py) 负责动作执行与历史观测选择。外部 Cosmos 源码、模型权重、统计量、文本 embedding 和模拟器资产均由调用者显式提供，本仓库不下载或 vendor 这些资源。

当前接入原精度推理，`--quant` 仅接受 `none`。模型固定预测 H=16 个七维动作，`--n-action-steps` 指每块实际执行的前 n 个动作，默认 n=16，范围为 1..16。遇到成功、环境终止或步数预算时立即停止。

## 配置路径与运行环境

从仓库根目录复制 [路径模板](paths.env.example)，填入已有资源后显式加载：

```bash
mkdir -p .local
cp -n benchmarks/static/cosmos_libero/paths.env.example .local/cosmos-libero.env
# 编辑 .local/cosmos-libero.env，将 PATH/TO 替换为实际路径
source .local/cosmos-libero.env
```

| 环境变量 | 对应 CLI / 内容 |
| --- | --- |
| `ROBOTICS_COSMOS_PYTHON` | `run.sh` 使用的 Python 解释器，须已有兼容模型与仿真依赖 |
| `ROBOTICS_COSMOS_SOURCE` | `--cosmos-source`，包含 `cosmos_policy/` 的外部源码目录 |
| `ROBOTICS_COSMOS_CHECKPOINT` | `--checkpoint`，`Cosmos-Policy-LIBERO-Predict2-2B.pt` 文件 |
| `ROBOTICS_COSMOS_DATASET_STATS` | `--dataset-stats`，配套 `libero_dataset_statistics.json` |
| `ROBOTICS_COSMOS_TEXT_EMBEDDINGS` | `--text-embeddings`，配套 `libero_t5_embeddings.pkl` |
| `ROBOTICS_COSMOS_VAE_CHECKPOINT` | `--vae-checkpoint`，配套视频 tokenizer 的 `tokenizer.pth` |
| `ROBOTICS_COSMOS_LIBERO_CONFIG_DIR` | `--libero-config-dir`，包含 `config.yaml` 的目录 |

CLI 路径优先于环境变量；shell 入口不会自动加载 `.env` 文件。也可使用选定的解释器直接运行 `python benchmarks/static/cosmos_libero/run.py`。`config.yaml` 中的 BDDL、初态、资产等路径须指向已准备好的 LIBERO 资源；运行入口设置 `LIBERO_CONFIG_PATH`。

CPU 开发依赖足以运行配置和接口测试。实际模型运行还需要兼容的 PyTorch/CUDA、Cosmos 依赖、LIBERO、MuJoCo/robosuite 及 EGL；启用录像需要 imageio 和 MP4/FFmpeg 后端。本机测试使用普通用户的Python 3.10.12环境，复用既有Cosmos packages，并单独安装robosuite 1.4.0；实际组合为PyTorch 2.7.0+cu128、MuJoCo 3.2.6、LIBERO 0.1.1。RoboCasa使用的robosuite 1.5.x不满足本LIBERO包的SingleArmEnv接口。这个本地复用环境不是完整的新机器安装流程。

入口没有自动下载功能：缺少checkpoint、统计量、T5或VAE文件会报错，运行时启用Hugging Face离线模式。当前LIBERO包先找包内assets，再使用 `~/.cache/libero/assets`；只有config.yaml中的assets字段不保证它能找到资产。原生Cosmos配置注册还可能检查源码目录内的相对checkpoint路径；本地验证的配置包含 `checkpoints/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt`。引擎只在原生加载阶段临时使用源码工作目录并恢复，不修改外部源码，也不会为这些依赖创建替代文件。

## 预检与小规模运行

先运行标准库预检，无需选择 GPU：

```bash
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 \
  --schedule sync --quant none \
  --output-dir runs/static/cosmos_libero/sync-001 --dry-run
```

`--dry-run` 校验参数、所需文件、统计量维度和有限数值，并用 AST 检查外部 `get_action` 调用签名。它读取文件计算 SHA-256，不反序列化 checkpoint/T5，不导入模型、Torch 或模拟器，不创建输出目录。大型 checkpoint 的哈希仍需要文件读取时间。预检通过不代表模型加载、数值或 GPU 闭环已验证。

实际运行必须显式指定一个 GPU 索引或完整 GPU UUID。以下用 GPU 0 示意，执行前替换为可用设备；每次使用新输出目录。

```bash
# 同步单 episode，保存完整执行视频
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 \
  --schedule sync --quant none --record-video \
  --gpu 0 --output-dir runs/static/cosmos_libero/sync-video-001

# 论文历史观测抽象：n′=2，仍由单环境串行执行器推进
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 \
  --schedule paper_async --overlap-actions 2 --quant none \
  --gpu 0 --output-dir runs/static/cosmos_libero/paper-async-001
```

程序同时将 Python 控制台输出写入 `<output-dir>/run.log`，无需手动 `tee`。已有输出目录会被拒绝。`COSMOS_SMOKE` 须关闭；外部该开关可能跳过 checkpoint 加载，因此即使将本次运行标记为 `--run-kind smoke`，也不能启用它。

## 任务选择与总预算

`--episodes` 是整个运行的总 episode 预算，默认 1。`--task-ids` 默认为 `all`，也接受 `0,2,9` 等不重复的逗号列表；四个可选 suite 的任务 ID 为 0..9。入口在选定任务间分配总预算，每个任务从初态 0 开始，不重复使用初态；预算超过选定任务的可用初态总数会失败。默认 `all` 配合少量 episode 不保证覆盖全部任务。

当前 LIBERO-object 资源为 10 个任务、每任务 50 个初态。完整覆盖应显式选择全部任务和总预算 500：

```bash
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids all --episodes 500 \
  --schedule sync --quant none \
  --gpu 0 --output-dir runs/static/cosmos_libero/object-all-sync-001
```

这条命令表达完整评估范围，当前文档不声明该评估已完成。与同步比较论文异步时，保持相同资源、任务/初态、seed、n、去噪步数和预算，仅调整调度及 n′。

| 参数 | 默认值 / 约束 |
| --- | --- |
| `--suite` | `libero_object`；另支持 `libero_spatial`、`libero_goal`、`libero_10` |
| `--task-ids` / `--episodes` | `all` / 总预算 `1` |
| `--seed` / `--env-seed` | 推理采样 `195` / 环境 `0` |
| `--n-action-steps` | `16`，范围 1..16；模型预测 horizon 固定 16 |
| `--num-inference-steps` | `5`，正整数 |
| `--max-steps` | 按 suite：spatial 220、object 280、goal 300、10 为 520；可覆盖为正整数 |
| `--schedule` / `--overlap-actions` | 默认 `sync` / `0`；n′ 范围 0..n |
| `--model-config` | `cosmos_predict2_2b_480p_libero__inference_only` |
| `--quant` | 仅 `none`，未接入的量化选项会被拒绝 |
| `--run-kind` | 默认单 episode 为 `smoke`，多 episode 为 `evaluation`；标签不证明验证范围 |

每个 episode 先显式 reset，再应用指定初态并进行 10 步 settling；settling 不计入 rollout 控制步数预算。环境不会在终止后自动开始另一个 episode。

## 论文异步与计时

`sync` 在需要生成动作块时使用当前观测。`paper_async` 在控制步 t 回取 t−n′ 的历史观测，主相机、腕相机和 proprio 来自同一份快照；历史不足时使用当前观测。历史按原始控制步保存，每次 episode reset 清空。省略 `--schedule` 时，正的 `--overlap-actions` 自动选择 `paper_async`；显式 `sync` 与正 n′ 冲突会失败。

这是串行执行中的历史观测实验，不包含推理与环境步进的真实后台并发。`requests.jsonl` 记录调用的 `control_step`、`observation_step`、`history_offset_steps` 和 `inference_index`，可核对实际采用的观测。

论文周期复用 [paper_async.py](../pi05_libero/paper_async.py) 的合同：

$$
C(n')=T_{\mathrm{inf}}+nT_{\mathrm{act}}-\min(T_{\mathrm{inf}},n'T_{\mathrm{act}}).
$$

`--paper-action-time-ms` 默认 50 ms，仅用于 `paper_model` 估算，不修改模拟器控制或 physics dt。需要周期估算时，同时传 `--paper-inference-time-ms` 和非空 `--paper-inference-time-source`，声明同配置 profile 的时延与来源；入口不验证来源文件。缺少推理时延时，周期保持 `null`、状态为 `requires_inference_profile`。`eval_s` 是串行评估墙钟时间，不能替代论文周期；控制步数也不能直接充当模型时延或加速比。

## 视频与自动产物

默认关闭视频。`--record-video` 录制本次实际评估的全部 episode；`--video-episodes-per-task N` 只录每任务前 N 条；`--no-record-video` 显式关闭，三者互斥。视频包含 settling 后的初始画面和成功、失败或超时的终止画面。`--video-fps` 默认 30，只影响播放帧率；录制/编码会增加墙钟开销，写入失败会使运行报错。

| 输出 | 内容 |
| --- | --- |
| `case-manifest.json` | 解析选项、计划任务/初态、资源与代码身份、case 指纹、运行依赖和完成状态 |
| `run.log` | 入口自动保存的 Python 控制台日志 |
| `checkpoint-load.json` | 实际加载覆盖审计；审计未通过则不开始 rollout |
| `paper-async.json` | 调度、历史/state 规则及可选论文周期估算 |
| `episodes.jsonl` | 逐 episode 任务/初态、成功、控制步数、推理调用数、终止原因和控制 dt |
| `requests.jsonl` | 逐请求的任务/初态及观测来源控制步 |
| `coverage.json` | 预期/完成 episode 数、初态覆盖、步数预算和运行状态 |
| `eval_results.json` | 正常结束时的总体/逐任务成功统计与 `eval_s`；启用视频时含路径及元数据 |
| `episode-summary.md` / `episode-summary.json` | 正常结束时的控制步数汇总，保留成功统计与失败预算惩罚口径 |
| `videos/<task>/*.mp4` | 启用录制时的完整 episode 视频 |
| `failure.json` | 实际执行异常及 traceback；失败发生前尚未生成的产物可能缺失 |

可用 `python tools/summarize_experiment.py --input runs/static/cosmos_libero/sync-video-001 --per-task` 在 CPU 上重新汇总。未完成的运行须显式加 `--allow-partial`；汇总不会将失败或不完整运行改成已完成。

manifest 使用独立 `cosmos-libero-run-v1` 格式，`requests.jsonl` 是本 case 的请求记录，不是通用逐 action trace。实验输出和本机配置留在 Git 忽略的本地目录。

## 验证与版本边界

已完成586项CPU测试及RTX 6000 Ada上的以下GPU检查：固定 `sample_libero_10_observation.pkl`、seed195时，新引擎与原生动作接口输出完全相等（16×7、float64、最大绝对误差0）；LIBERO-object任务0、初态0、环境seed0、H=n=16、5步采样下，同步episode在137个控制步成功，`paper_async n′=2` 在154步成功。两次加载与覆盖审计通过；异步请求记录确认首轮使用当前观测，后续回看2个原始控制步；录像为256×256、30 FPS、155帧，含初始和终止画面。

这些是固定输入对齐和单episode闭环连通性证据，不是完整任务集成功率或性能结论。未采集可比较的推理profile，因此不报告论文加速比。量化、RoboCasa以及其他任务的闭环尚未验证。

Cosmos 接入在基线 `1a4983e` 之后扩展，保留该基线已有的 π0.5 功能及其独立入口。模型实现依然来自调用者指定的外部源码。当前本机检查使用的 Cosmos 源码副本没有 `.git`，不能标为某个纯上游 commit：计划记录 Python 源码树 SHA-256 与 `git_available=false`，有可用 Git 信息时才记录真实 commit/dirty 身份。资源文件、本仓执行代码和选项共同进入 case 指纹。

来源与条款边界见 [第三方记录](../../../THIRD_PARTY_NOTICES.md)；模型、任务、调度和时间口径分别遵循 [模型接入](../../../docs/protocols/model-adapters.md)、[仿真协议](../../../docs/protocols/simulation.md) 与 [加速比协议](../../../docs/protocols/speedup-metrics.md)。
