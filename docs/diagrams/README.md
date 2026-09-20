# 架构图源码

Git保留Mermaid源码及生成脚本，SVG、HTML和生成清单在本地重建。当前实现的完整说明见 [代码架构](../architecture.md)。

| 图稿 | 内容 |
| --- | --- |
| [01-architecture.mmd](01-architecture.mmd) | 三个已实现静态case、外部依赖、CPU工具与规划中的动态case |
| [02-async-loop.mmd](02-async-loop.mmd) | 当前 `paper_async` 的串行历史观测选择与动作执行 |
| [03-inference-path.mmd](03-inference-path.mmd) | Cosmos加载审计、本仓执行代码、可选算子与动作块交付 |
| [04-contribution-flow.mmd](04-contribution-flow.mmd) | 主题分支、中英文提交/PR、相关验证和维护者合并 |

绿色为已有模块或采用的协作步骤，灰色为外部资源，蓝色为明确标注的后续方向。实线表示调用、数据传递或流程推进；虚线表示配置约束或可选依赖。协作图不表示GitHub服务端保护已启用。

图02描述当前Cosmos静态runner：每步保存观测，在需要新动作块时读取t−n′快照，图像与proprio一起延迟。它不绘制后台并发推理，也不把声明的论文周期等同于测得的宿主运行耗时。π0.5采用相同case语义，但由外部evaluator桥接实现。

图03只描述Cosmos已接入路径。可选融合与量化已接入，但量化全量任务质量仍待验证。DOM时钟驱动和Kinetix原生JAX执行按各自case扩展，不在图中冒充现有backend。

## 本地生成

在独立文档环境安装Mermaid CLI，使用 [render-config.json](render-config.json)。进入本目录后：

```bash
mkdir -p rendered
for source in ./*.mmd; do
  stem=$(basename "$source" .mmd)
  mmdc -i "$source" -o "rendered/$stem.svg" \
    -I "fig-$stem" -c render-config.json -b white -w 1800
done
python build_gallery.py
```

如需指定浏览器路径，向mmdc传入自己的Puppeteer配置。渲染依赖不进入CPU工具或GPU模型环境。

[build_gallery.py](build_gallery.py) 从源码与SVG生成 `architecture-atlas.html`、`gallery.md` 和 `render-manifest.json`，支持在本地查看、缩放和下载图稿。所有生成产物均被忽略；公共文档直接引用可审阅的文本源码。
