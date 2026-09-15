# 架构图源码

Git保留以下Mermaid源码及生成配置/脚本；SVG、HTML和生成清单在本地重建，不随代码版本保存。

| 图稿 | 内容 |
| --- | --- |
| [01-architecture.mmd](01-architecture.mmd) | 绑定 case 的静态/动态轨道、DOM 与 Kinetix 独立执行分支及共享底层 |
| [02-async-loop.mmd](02-async-loop.mmd) | 按块交接的异步示例、等待与观测年龄 |
| [03-inference-path.mmd](03-inference-path.mmd) | VLA动作块推理与执行后端的边界；Kinetix保留原生RL路径 |
| [04-contribution-flow.mmd](04-contribution-flow.mmd) | 主仓与模拟器fork的协作流程 |

绿色节点表示已有入口文档、契约、CPU工具及协作文件；蓝色模型/runtime节点全部是 planned，灰色为其他项目调用方。入口文档已建立不代表模型、模拟器或调度器已经可运行。实线表示调用或数据流，虚线表示契约约束或可选复用。总体说明见 [架构文档](../architecture.md)。

## 两条轨道与绑定 case

[静态实验入口](../../benchmarks/static/README.md) 与 [动态实验入口](../../benchmarks/dynamic/README.md) 分开组织。每个 case 先绑定模型、checkpoint、processor、任务、环境版本和执行协议，再选择该 case 已声明兼容的精度或调度变体；图中不展开模型与模拟器的任意组合。

静态轨道的候选是 π0.5＋LIBERO、Cosmos 对应的 LIBERO/RoboCasa case，LingBot 的具体 case 待确认。动态轨道保留两个独立分支：DOM 的 DynamicVLA 模型进程与 Isaac ClockDriver，以及 [Kinetix 独立入口](../../benchmarks/dynamic/kinetix/README.md) 下的原生 RL policy 与 JAX rollout。Kinetix 保留自己的动作空间、state/PRNG、编译和批量执行，不要求使用 VLA PolicyAdapter、进程 Executor 或输出 action chunk。

共享层主要是输入输出与状态契约、后端能力描述、计时 trace 和指标定义。各分支按兼容性选择执行组件，保留自己的 Runner/ClockDriver；任务成功判据和预算仍由 case 固定。静态任务可以 async，动态任务可以 sync；静态不等于离线 forward 或冻结世界。轨道划分来自任务条件，不是 LIBERO、RoboCasa 等引擎的永久属性。

## 按块交接示例的范围

图02只说明“完成旧 chunk 再执行新 chunk”的局部时间线：t=0 采样，80 ms 输出可用，120 ms 交接，故 boundary wait 为0而 observation age 为120 ms。这是待实现协议的手算例子，不是性能数据，也不代表 DOM 按时刻替换位姿目标或 Kinetix 原生 JAX 控制循环。

图04说明主仓与按需模拟器 fork 的协作。具体任务范围、认领和共享交接信息写入 Issue/PR，版本化模板用于复用格式；远端强制 review 和 CI 状态需单独验证。

## 本地生成

在独立文档环境中安装Mermaid CLI；本轮使用版本11.15.0和Noto Sans CJK SC字体。进入本目录，使用 [render-config.json](render-config.json)：

```bash
mkdir -p rendered
for source in ./*.mmd; do
  stem=$(basename "$source" .mmd)
  mmdc -i "$source" -o "rendered/$stem.svg" \
    -I "fig-$stem" -c render-config.json -b white -w 1800
done
python build_gallery.py
```

如需指定浏览器路径，给mmdc添加自己的Puppeteer配置。渲染依赖不加入核心工具环境。

[build_gallery.py](build_gallery.py) 从源码和已渲染SVG生成 `architecture-atlas.html`、`gallery.md` 与 `render-manifest.json`。打开本地HTML可以切换、缩放和下载图稿；这些产物均被忽略。生成器不会覆盖此README。
