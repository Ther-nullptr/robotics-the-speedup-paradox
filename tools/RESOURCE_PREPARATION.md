# 模型和数据资源准备

[prepare_resources.py](prepare_resources.py) 为 `cosmos_libero` 和 `pi05_libero` 提供显式资源准备命令。实验入口仍按本地路径运行，并启用离线模式；缺少文件时会报错，不在推理或评测途中下载。

## 默认下载位置

| 资源 | 默认目录 |
| --- | --- |
| 模型、VAE、文本 tokenizer | `~/.cache/robotics/hub/models--ORG--REPO/snapshots/COMMIT/` |
| 可选训练/校准轨迹 | `~/.cache/robotics/hub/datasets--ORG--REPO/snapshots/COMMIT/` |
| LIBERO mesh、texture、场景资产 | `~/.cache/libero/assets/` |
| 生成的路径配置 | `~/.cache/robotics/env/CASE.env` |
| 下载来源记录 | 与 env 同目录的 `CASE.env.resources.json` |
| 生成的 LIBERO 配置 | `~/.cache/robotics/configs/CASE/config.yaml` |

`--root /data/robotics` 优先于环境变量 `ROBOTICS_RESOURCE_ROOT`，两者都未设置时使用 `~/.cache/robotics`。本工具显式传递 Hugging Face 的 `cache_dir`，所以 `HF_HOME` 不改变上表模型和数据集的下载位置。具体文件路径由下载结果返回并写入 env；[Hub 缓存与 local_dir 行为](https://huggingface.co/docs/huggingface_hub/guides/download) 由 `huggingface_hub` 管理。

**LIBERO assets 是例外，不随 `--root` 移动。** 当前验证的 LIBERO wheel 在包内资产缺失时直接查找 `~/.cache/libero/assets`，忽略 YAML 中的 assets 路径。工具沿用这个目录，避免下载到了另一个位置却无法加载。原生包内已有 assets 时仍按该包的解析优先级使用。自定义 fork 的资产布局可继续使用 case 的手工路径配置。

模型、下载缓存、数据和生成配置均不进入 Git。放在仓库内时，使用已忽略的 `.local/`、`artifacts/` 等目录。

## 新机器准备命令

从仓库根目录执行。`--dry-run` 只需 Python 标准库，不联网、不创建文件，也不检查占位路径是否存在：

```bash
python tools/prepare_resources.py --case cosmos_libero --dry-run
python tools/prepare_resources.py --case pi05_libero --dry-run

# Install only the optional download client.
python -m pip install -r tools/requirements-download.txt

# Prepare the default Cosmos inference resources.
python tools/prepare_resources.py --case cosmos_libero
source ~/.cache/robotics/env/cosmos_libero.env
```

上面的简短命令只配置模型资源。若同时提供已安装的运行环境与源码位置，工具会生成完整的启动路径配置：

```bash
python tools/prepare_resources.py --case cosmos_libero \
  --root /data/robotics \
  --source /path/to/cosmos-policy \
  --python /path/to/cosmos-env/bin/python \
  --libero-root /path/to/cosmos-env/lib/python3.10/site-packages/libero/libero

source /data/robotics/env/cosmos_libero.env
bash benchmarks/static/cosmos_libero/run.sh \
  --task-ids 0 --episodes 1 --schedule sync --quant none \
  --output-dir runs/static/cosmos_libero/demo-001 --dry-run
# After preflight, remove --dry-run and add --gpu with your selected GPU.
```

π0.5 使用同一工具，提供它自己的运行环境和兼容 sim 源码：

```bash
python tools/prepare_resources.py --case pi05_libero \
  --source /path/to/compatible-vlash-sim \
  --python /path/to/pi05-env/bin/python \
  --libero-root /path/to/pi05-env/lib/python3.10/site-packages/libero/libero

source ~/.cache/robotics/env/pi05_libero.env
bash benchmarks/static/pi05_libero/run.sh \
  --schedule paper_async --overlap-actions 2 --quant none \
  --output-dir runs/static/pi05_libero/demo-001 --dry-run
```

`--libero-root` 指向含 `bddl_files/` 和 `init_files/` 的已安装包目录；这些任务定义与初态随兼容 LIBERO 包提供。工具检查四个 suite 的 40 个 BDDL/初态文件是否成对存在，生成配置以避免首次 import 的交互提示。不提供该参数时保留现有 LIBERO 配置变量，终端会提示尚未生成配置。

本工具不安装 CUDA、Torch、Cosmos、LeRobot、LIBERO 或量化 kernel，也不克隆外部源码。`--source` 和 `--python` 只绑定已有位置。`resources_prepared` 仅表示资源文件准备完成；运行环境兼容性和 GPU 可用性仍由 case 预检与实际运行验证。

## 下载内容与已有资源复用

| case / 名称 | 来源与内容 |
| --- | --- |
| Cosmos `policy` | [NVIDIA Cosmos Policy LIBERO](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)：Policy `.pt`、数据统计 JSON、预计算 T5 embedding |
| Cosmos `vae` | [NVIDIA Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)：仅 `tokenizer/tokenizer.pth` |
| π0.5 `policy` | [LeRobot π0.5 LIBERO](https://huggingface.co/lerobot/pi05_libero_finetuned_v044)：模型、配置及配套 processor 状态；不使用 VLASH 微调权重 |
| π0.5 `tokenizer` | [Google PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224)：仅 tokenizer/processor 配置和词表，不下载 PaliGemma 模型 |
| 两者 `libero-assets` | 当前 LIBERO wheel 使用的 [libero-assets](https://huggingface.co/jadechoghari/libero-assets)：场景、网格与纹理 |
| Cosmos `dataset`，可选 | [LIBERO-Cosmos-Policy](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy)：包含成功与失败轨迹的原生 HDF5 数据 |
| π0.5 `dataset`，可选 | [LeRobot LIBERO](https://huggingface.co/datasets/lerobot/libero)：LeRobot 格式轨迹；不是 Cosmos HDF5 格式 |

只有 `--with-dataset` 才下载最后两项。LIBERO 全任务闭环评估使用任务初态和模拟器，不读取训练轨迹；预计算 T5 和归一化统计属于推理必需文件，始终准备。数据集路径导出为 `ROBOTICS_COSMOS_DATASET` 或 `ROBOTICS_DATASET`，供后续校准工具使用，当前评测器不消费这两个变量，也不自动转换数据格式。

```bash
# Dataset downloads are opt-in. Use a new env filename for a changed plan.
python tools/prepare_resources.py --case cosmos_libero --with-dataset \
  --env-file ~/.cache/robotics/env/cosmos_libero_with_data.env

# Reuse existing local directories without copying or downloading them.
python tools/prepare_resources.py --case cosmos_libero \
  --reuse policy=/path/to/Cosmos-Policy-LIBERO-Predict2-2B \
  --reuse vae=/path/to/Cosmos-Predict2-2B-Video2World \
  --reuse libero-assets="$HOME/.cache/libero/assets" \
  --env-file .local/cosmos-resources.env
```

`--reuse NAME=PATH` 可以重复指定；未复用的资源才联网下载。VAE 复用目录必须含 `tokenizer/tokenizer.pth`。本地复用校验必需文件及非空/LFS 占位状态，并记录 `origin=local`，不会声称验证了远端版本或完整内容哈希。π0.5 还会检查 processor JSON 引用的状态文件。需要离线准备时复用所有资源。

下载采用固定的 checkpoint/VAE 版本，保持与已有设置一致；`--revision NAME=COMMIT` 可显式覆盖。没有固定默认版本的资源在下载前解析为具体 commit，一次资源下载使用同一 commit，并记录文件数、字节数和版本。失败后可重跑同一命令，利用 Hub 缓存续传/复用；不会把缺文件、空文件、LFS pointer 或与远端大小不符的文件当作完成。

若 Hub 要求权限，例如 PaliGemma，先在模型页面接受相应条款，再按 Hugging Face 官方流程登录。凭据由 Hub 客户端读取，不写入 env 或来源记录。离线模式变量 `HF_HUB_OFFLINE=1` 须在在线准备前取消。

生成的 env、来源记录和 LIBERO 配置只接受新文件或完全相同内容，拒绝覆盖已有不同配置。修改资源计划时选择新的 `--env-file`；如需更换 LIBERO 包位置，同时使用新的 `--root`。下载完成后显式 `source`，CLI 路径参数仍优先于环境变量。

## Cosmos 的额外基础权重引用

原生 Cosmos 注册训练配置时会解析 Video2World 基础模型引用，之后推理加载器才覆盖为 Policy checkpoint。本仓库在加载期间只将这一已知引用绑定到显式传入的 Policy 文件，退出时恢复原生解析函数；最终 `checkpoint.load_path` 和 VAE 路径必须与请求相符。`checkpoint-load.json` 记录绑定与实际加载审计，因此无需为这个未参与推理的基础引用额外下载权重或在源码树放置占位文件。
