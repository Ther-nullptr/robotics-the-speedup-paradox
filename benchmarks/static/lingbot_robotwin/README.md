# LingBot-VA＋RoboTwin 同步实验

本case优先适配RoboTwin双臂任务。模型在独立Python worker中运行，仿真端通过本机WebSocket调用；保留LingBot的条件帧、动作块和KV/VAE缓存更新语义。当前支持单环境同步评估，默认任务为 `adjust_bottle`，不提供 `paper_async` 或量化选项。

## 本地资源与环境

入口只使用显式提供的本地资源，模型加载强制离线，不下载checkpoint、资产或运行依赖。模型目录必须包含 `transformer/`、`text_encoder/`、`vae/`、`tokenizer/`，预检核对分片索引并记录文件SHA-256。仿真源目录需要已解压的 `assets/objects`、`assets/embodiments/aloha-agilex` 和 `assets/background_texture`。

已有任务资产足以进行闭环评估，不需要下载LeRobot训练轨迹。调用者的源码、环境和checkpoint作为只读资源使用；结果写入自己的 `--output-dir`。Torch/cuRobo本地编译缓存仍由所选运行环境管理。

已验证的本机组合：

| 内容 | 模型环境 | 仿真环境 |
| --- | --- | --- |
| Python | 3.10.16 | 3.10.19 |
| Torch | 2.9.0+cu126 | 2.9.0+cu126 |
| NumPy | 2.2.6 | 1.26.4 |
| 模型依赖 | Diffusers 0.36.0、Transformers 5.0.0、FlashAttention 2.8.3 | 无需安装模型运行栈 |
| 仿真依赖 | 无需安装SAPIEN | SAPIEN 3.0.0b1、mplib 0.2.1、cuRobo及Vulkan |

模型来源为 [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va)，当前接入检查的提交为 `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`，权重为其RoboTwin专用posttrain版本。所用本地RoboTwin源码HEAD为 `125db35a88cb89d859da7691a8909d25cd9cbacf`；实际代码身份以每次manifest中的文件哈希为准。这是已验证的本地资源组合，不是新机安装器或50任务完整复现声明。

Transformers 5可为此UMT5 checkpoint额外初始化 `encoder.embed_tokens`，而checkpoint只保存 `shared.weight`。模型worker仅在这种存储结构下恢复原有共享输入embedding关系，然后核对实际参数覆盖；不会修改原始checkpoint或环境。修正记录在 `checkpoint-load.json`。源代码模块仍导入FlashAttention，但此case实际使用原生Torch SDPA。

## 启动

从仓库根目录复制模板并填写已有路径：

```bash
mkdir -p .local
cp -n benchmarks/static/lingbot_robotwin/paths.env.example .local/lingbot-robotwin.env
# Replace PATH/TO values with existing local resources.
source .local/lingbot-robotwin.env

# Standard-library preflight; no model or simulator is imported.
bash benchmarks/static/lingbot_robotwin/run.sh \
  --task adjust_bottle --episodes 1 --start-seed 10000 \
  --output-dir runs/static/lingbot_robotwin/trial-001 --dry-run

# Actual evaluation: starts and stops its own local model worker.
bash benchmarks/static/lingbot_robotwin/run.sh \
  --task adjust_bottle --episodes 1 --start-seed 10000 --model-seed 0 \
  --gpu 3 --record-video \
  --output-dir runs/static/lingbot_robotwin/trial-001
```

`--model-python`、`--lingbot-source`、`--robotwin-source`、`--checkpoint` 可以覆盖env中的路径。shell入口使用 `ROBOTICS_ROBOTWIN_PYTHON` 启动仿真客户端。两个进程使用同一个显式GPU索引，通信端口自动选择；不连接未经声明的外部模型服务。

每次真实实验使用新的输出目录。默认启用CPU offload，将VAE和文本编码器留在CPU；可以显式使用 `--no-cpu-offload`。`--startup-timeout` 默认为600秒。当前GPU验证仅覆盖 `demo_clean` 下的 `adjust_bottle`；其他任务需逐项验证，不能将多卡调度脚本中重复的任务计为新增任务。

## 控制与缓存协议

模型返回 `[16, 2, 16]` 动作数组：16个双臂末端/夹爪通道、2个帧块、每帧16条控制指令。首块的第一帧为条件帧，不执行，所以首轮执行16条，后续完整块执行32条。

```text
选择原生expert能够完成的场景种子
→ 重建该场景并获取指令、相机和初始末端位姿
→ reset模型缓存与本回合模型随机种子
→ 推理动作块
→ 逐条执行双臂末端动作，检查成功和预算
→ 完整块每4条指令取一组真实观测，更新KV/VAE缓存
→ 下一块；成功或预算耗尽立即结束
```

首块更新使用4组真实观测，后续完整块使用8组；cache里的action条件保持原生模型动作表示，包含首块未执行的条件帧。末端位姿转换与执行由模拟器适配器负责，不能将预测动作直接当作测量到的机器人状态。终止或预算截断的块不会把未执行动作灌入下一次cache更新。

`primitive_steps` 在这个case中指原生 `take_action` 接受的控制指令数。每条指令可能推进多个250Hz物理步，其时长取决于运动规划，不能把该计数当作物理tick或固定50ms控制周期。`adjust_bottle` 原生预算为400条；`--max-steps` 可设置更小预算。模型、仿真或初始化错误使运行失败，不计作有效策略失败episode。

场景选择沿用RoboTwin的expert筛选，记录每个候选与接受的种子；这不筛掉模型失败回合。`--episodes` 为总评估回合数，下一回合从上一个接受种子的后一位继续搜索。最多搜索 `--max-initialization-attempts` 个候选，默认32，避免初始化无限重试。

`paper_async` 暂不开放：后续延迟协议必须同时限定KV/VAE缓存可见的观测，不能只延迟当前输入。当前原生WebSocket server的async命名也不表示论文的n′实验已接入。

## 输出与已验证结果

| 文件 | 内容 |
| --- | --- |
| `case-manifest.json` | 配置、资源及源码身份、状态 |
| `run.log` / `server.log` | 仿真/调度日志与独立模型worker日志 |
| `checkpoint-load.json` | Transformer参数覆盖、文本embedding共享关系和模型运行信息 |
| `initialization-000000.json` | 原生种子筛选、指令、初始末端位姿、观测指纹 |
| `requests.jsonl` | 推理请求与缓存更新，分别记录RPC墙钟耗时 |
| `episodes.jsonl` / `coverage.json` | 成功、实际控制指令数、预算、请求和缓存更新次数 |
| `episode-summary.md` / `.json` | 成功率、失败预算惩罚的总体步数、仅成功步数 |
| `videos/<task>/` | 可选完整控制指令录像，三视角横排 |

视频默认关闭。`--record-video` 每条控制指令记录一帧并保留初始/终止画面，独立于模型的4步关键帧采样。默认播放15FPS，不声称与物理仿真时间等速。worker由case在结束时清理，torchrun可能在 `server.log` 输出收到SIGTERM的退出信息；以manifest与加载审计判断运行结果。

已完成 RTX 6000 Ada 上的单回合同步闭环：`adjust_bottle`、`demo_clean`、原生种子10000、模型种子0，115条控制指令成功，5次模型请求、4次缓存更新；Transformer和文本编码器参数覆盖审计通过。视频为960×240、116帧。尚无50任务全量结果或推理加速比。

## English summary

This case prioritizes RoboTwin and runs LingBot in a separate local model environment. Resources are local-only and model loading is offline. Configure the five paths in `paths.env.example`, run preflight, then select an idle GPU explicitly.

The native synchronous protocol executes 16 commands from the initial conditional chunk and 32 from subsequent full chunks, then updates KV/VAE caches from observed keyframes. Control-command counts are distinct from physics ticks. `paper_async` is intentionally unavailable until cache visibility is defined.

One `adjust_bottle` episode succeeded in 115 commands, with 5 inference requests, 4 cache updates and 116 recorded frames. This is a single-episode integration check, not a 50-task evaluation or speedup claim.
