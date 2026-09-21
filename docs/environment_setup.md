# 环境配置与首次运行

[English](environment_setup.en.md) · [项目首页](../README.md) · [资源准备参数](../tools/RESOURCE_PREPARATION.md)

本仓库将CPU工具、资源下载与模型/模拟器运行分开。`requirements-dev.txt` 用于CPU开发；`pip install -e .` 仅安装本仓Python包。GPU环境需要对应case的兼容外部源码和依赖，当前没有统一的一键新机安装器。

LingBot优先通过 [RoboTwin独立case](../benchmarks/static/lingbot_robotwin/README.md) 接入，复用已有模型端和仿真端环境。该入口仅接受本地资源，未加入通用下载工具的case列表；具体版本、路径和Transformers共享embedding兼容处理见case说明。

## 1. CPU工具环境

建议Python 3.11或3.12；Python包元数据的最低版本为3.10。在仓库根目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/validate_contracts.py --examples
python tools/compare_speedups.py --input examples/speedups/synthetic.json --markdown
python tools/embodied/trajectory_metrics.py --input tools/embodied/examples/smooth.csv
```

CPU开发依赖为NumPy、jsonschema、pytest和Ruff，不包含Torch、CUDA或模拟器。绘图按需安装：

```bash
python -m pip install -r tools/embodied/requirements-plot.txt
```

运行测试和提交检查见 [贡献指南](../CONTRIBUTING.md)。文档修改无需重新运行模型实验。

## 2. 为每个case配置独立GPU环境

以下是已有实验验证过的组合，用于判断依赖兼容性，不是完整依赖锁文件。

| 依赖 | π0.5＋LIBERO | Cosmos＋LIBERO | Cosmos＋RoboCasa |
| --- | --- | --- | --- |
| Python | 3.10 | 3.10 | 3.10 |
| PyTorch | 2.7.1+cu126 | 2.7.0+cu128 | 2.7.0+cu128 |
| NumPy | 1.26.4 | 2.2.6 | 2.2.6 |
| MuJoCo | 3.8.0 | 3.2.6 | 3.2.6 |
| robosuite | 1.4.0 | 1.4.0 | 1.5.1 |
| 特有依赖 | LeRobot 0.4.1、Transformers 4.53.3及兼容evaluator改动 | LIBERO 0.1.1、Transformer Engine、NATTEN | 指定RoboCasa fork、Transformer Engine、NATTEN |

Cosmos本机组合使用Transformer Engine `2.2+cu128.torch27`、NATTEN `0.21.0+cu128.torch27`。这些编译扩展必须与实际Torch/CUDA匹配；不同Python、GPU或CUDA组合需按对应项目安装和验证。模型实验目前使用RTX 6000 Ada。

主要模型执行代码现位于本仓 `src/robotics_bench/models/`，默认使用 `--model-runtime owned`，但公共框架、加载器和模拟器依赖仍需配置。可选融合需要模型环境中的Triton；INT后端另外需要兼容CUTLASS checkout和CUDA开发工具链，本次Ada构建使用CUDA 12.8。路径与目标架构必须显式指定，见 [算子构建说明](../src/robotics_kernels/README.md)。FP4/FP8尚待Blackwell实机验证。逐轮图表通过外部profile-visualizer skill生成，CairoSVG/Cairo放在独立渲染环境，见 [推理实验入口](../benchmarks/inference/README.md)。

- **π0.5**：需要支持本仓参数与历史观测协议的VLASH sim evaluator，并使用普通LeRobot π0.5 LIBERO权重。现有兼容源码含本地改动，仅安装相同版本号的LeRobot不足以证明接口可用。入口通过AST预检检查evaluator字段，实际源码身份写入manifest。参见 [case依赖边界](../benchmarks/static/pi05_libero/README.md#已验证范围与外部依赖)。
- **Cosmos**：按 [官方安装说明](https://github.com/NVlabs/cosmos-policy/blob/main/SETUP.md) 配置模型依赖，分别安装LIBERO和RoboCasa的环境栈。当前普通用户环境复用了本机已有Cosmos packages，未验证从空机器完整重建；这些复用路径不会写入公共模板。
- **RoboCasa**：使用 [官方指定fork](https://github.com/moojink/robocasa-cosmos-policy)，已验证commit为 `edd9a328b3ec98050f42d194c1419307a79c4d87`。控制器配置使用Cosmos源码附带的文件。详细安装入口见 [官方RoboCasa说明](https://github.com/NVlabs/cosmos-policy/blob/main/ROBOCASA.md)。

系统需要可用的NVIDIA驱动和无窗口渲染所需的EGL/OpenGL环境；启用视频需要imageio与FFmpeg编码后端。在将要运行实验的解释器中检查：

```bash
nvidia-smi
/path/to/case-env/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

`--gpu` 必须显式指定空闲设备的索引或完整GPU UUID。入口在导入运行栈前设置设备和EGL选项。不要将多个case的环境安装到同一个已有环境里升级覆盖；尤其要保持两种robosuite版本隔离。

## 3. 准备模型、任务与资产

在CPU环境安装可选下载客户端：

```bash
python -m pip install -r tools/requirements-download.txt
python tools/prepare_resources.py --case cosmos_libero --dry-run
```

资源准备支持三种case：`pi05_libero`、`cosmos_libero`、`cosmos_robocasa`。`--dry-run` 是完全离线的计划预览，实际下载在去掉该参数后执行。

| 资源 | 默认位置与准备方式 |
| --- | --- |
| Policy checkpoint、VAE、tokenizer、统计量和T5 | 资源工具下载到 `~/.cache/robotics/hub/` |
| 训练/校准轨迹 | 仅 `--with-dataset` 时下载到相同Hub缓存 |
| LIBERO BDDL和初态 | 随兼容LIBERO包提供；用 `--libero-root` 指定含 `bddl_files/`、`init_files/` 的目录 |
| LIBERO模型/纹理资产 | 资源工具下载到原生解析器使用的 `~/.cache/libero/assets/` |
| RoboCasa厨房资产 | 兼容fork的脚本下载到其 `robocasa/models/assets/`；资源工具不代为下载 |
| 生成的env配置 | `~/.cache/robotics/env/<case>.env`，需要显式 `source` |

模型和数据缓存可用 `--root /data/robotics` 或 `ROBOTICS_RESOURCE_ROOT` 改变；LIBERO/RoboCasa模拟器资产位置仍遵循上表。`HF_HOME` 不覆盖本工具显式选择的Hub缓存目录。

闭环评估从任务初态开始，不读取整套训练轨迹。归一化统计和预计算T5是Cosmos推理所需资源，随模型准备。受限Hub资源需先通过对应项目的授权和登录流程，凭据不写入env配置或Git。

已有本地资源时，使用 `--reuse policy=DIR`、`--reuse vae=DIR` 等参数直接复用，不复制模型。全部资源都复用时无需联网。修改资源计划使用新的 `--env-file`，工具拒绝覆盖内容不同的已有配置。完整参数见 [资源工具说明](../tools/RESOURCE_PREPARATION.md)。

## 4. 绑定运行路径

从仓库根目录复制对应模板，填写已有资源和解释器：

```bash
mkdir -p .local
cp -n benchmarks/static/cosmos_robocasa/paths.env.example .local/cosmos-robocasa.env
# Replace PATH/TO values with your runtime and resource paths.
source .local/cosmos-robocasa.env
```

另外两个模板位于 `benchmarks/static/pi05_libero/` 和 `benchmarks/static/cosmos_libero/`。模板中的解释器路径必须是case虚拟环境的 `bin/python`，不能将它替换为软链接指向的基础Python。

也可以在资源准备时一次生成完整路径配置，例如：

```bash
python tools/prepare_resources.py --case cosmos_robocasa \
  --source /path/to/cosmos-policy \
  --robocasa-source /path/to/robocasa-cosmos-policy \
  --python /path/to/robocasa-env/bin/python
source ~/.cache/robotics/env/cosmos_robocasa.env
```

Cosmos＋LIBERO选择 `--case cosmos_libero`，并把 `--robocasa-source` 换成 `--libero-root /path/to/site-packages/libero/libero`。π0.5使用 `--case pi05_libero`，`--source` 指向兼容VLASH sim源码，同样提供其LIBERO包目录。RoboCasa厨房资产仍需按case说明单独准备。

CLI路径参数优先于环境变量。脚本不自动加载任意 `.env` 文件，且 `--dry-run` 不导入模型/模拟器、不创建结果目录。下载客户端所在Python可以与case运行Python不同。

## 5. 预检并启动

### Cosmos＋RoboCasa

```bash
bash benchmarks/static/cosmos_robocasa/run.sh \
  --task TurnOffMicrowave --layout-id 1 --style-id 1 --env-seed 0 \
  --episodes 1 --schedule sync \
  --output-dir runs/static/cosmos_robocasa/sync-001 --dry-run
```

### Cosmos＋LIBERO

```bash
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 --schedule sync \
  --output-dir runs/static/cosmos_libero/sync-001 --dry-run
```

### π0.5＋LIBERO

```bash
bash benchmarks/static/pi05_libero/run.sh \
  --suite libero_object --episodes 1 --batch-size 1 --schedule sync --quant none \
  --output-dir runs/static/pi05_libero/sync-001 --dry-run
```

### LingBot＋RoboTwin

按 [case说明](../benchmarks/static/lingbot_robotwin/README.md) 配置模型和仿真两个Python环境，在env模板中填入已有源码、checkpoint和RoboTwin资产路径。该入口使用本地资源，不自动下载。

```bash
source .local/lingbot-robotwin.env
bash benchmarks/static/lingbot_robotwin/run.sh \
  --task adjust_bottle --episodes 1 --start-seed 10000 --model-seed 0 \
  --schedule paper_async --overlap-actions 2 \
  --output-dir runs/static/lingbot_robotwin/paper-async-001 --dry-run
```

LingBot的n′范围0..16，以控制指令计数；延迟同时应用于KV/VAE观测历史。默认是sync/0，切换回同步时同时去掉两个异步参数。

先显式加载所选case的env文件，再运行对应命令。预检成功后去掉 `--dry-run`，添加 `--gpu 3`；按需添加 `--record-video`。切换论文异步使用 `--schedule paper_async --overlap-actions 2`。所有case的 `--episodes` 都是本次运行的总回合数；当前RoboCasa一次命令只选择一个任务和布局组合。

全量π0.5 `libero_object` 使用 `--episodes 500 --batch-size 10`。RoboCasa对照可添加 `--reference-run` 检查与已完成baseline相同的初始化。完整参数和结果口径以各case文档为准。

## 6. 日志、视频与结果

结果写入显式 `--output-dir`；已有目录会被拒绝，避免不同实验混写。Cosmos自动保存 `run.log`。π0.5如需保存控制台日志：

```bash
mkdir -p logs
set -o pipefail
bash benchmarks/static/pi05_libero/run.sh \
  --schedule sync --quant none --gpu 3 \
  --output-dir runs/static/pi05_libero/trial-001 \
  2>&1 | tee logs/pi05-trial-001.log
```

每个成功结束的运行生成 `episode-summary.md` 和 `.json`。视频默认关闭，开启后保存在 `videos/<task>/`。失败预算统计、已有结果再分析和每种case额外产物见 [首页](../README.md#实验协议与输出) 与case文档。

## 常见问题

| 现象 | 检查方式 |
| --- | --- |
| checkpoint、tokenizer或T5缺失 | 先运行资源工具或提供 `--reuse` 路径；实验入口不会临时联网补资源 |
| `SingleArmEnv` 导入失败 | LIBERO可能加载了robosuite 1.5.x；使用自己的1.4.0环境 |
| RoboCasa提示资产缺失或采样异常 | 检查兼容fork的fixtures、textures和Objaverse是否完整解压 |
| RoboCasa实际加载旧容器路径 | 通过 `--robocasa-source` 提供有效源码；检查选定解释器和editable安装位置 |
| 初始场景与reference不同 | 使用相同任务、layout/style、环境种子和资产版本；查看 `initializations/` 指纹 |
| 无法渲染或录制 | 检查case环境中的EGL、MuJoCo、imageio和FFmpeg；GPU选项与模型设备一致 |
| Conda中XML/Matplotlib导入报 `XML_SetReparseDeferralEnabled` | 检查Python与libexpat是否来自匹配环境；本机验证曾临时预加载配套库，这不是项目默认启动要求 |

环境检查通过、资源完整和真实模型闭环成功是不同层级的验证。当前提供已验证组合与显式路径绑定，跨机器完整安装仍需要按外部项目依赖检查。
