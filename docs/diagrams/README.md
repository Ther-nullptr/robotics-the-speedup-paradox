# 架构图源码

Git保留以下Mermaid源码及生成配置/脚本；SVG、HTML和生成清单在本地重建，不随代码版本保存。

| 图稿 | 内容 |
| --- | --- |
| [01-architecture.mmd](01-architecture.mmd) | 总体模块、公开边界与后续调用方 |
| [02-async-loop.mmd](02-async-loop.mmd) | 异步闭环时序、等待与观测年龄 |
| [03-inference-path.mmd](03-inference-path.mmd) | 模型语义与执行后端的边界 |
| [04-contribution-flow.mmd](04-contribution-flow.mmd) | 主仓与模拟器fork的协作流程 |

模型/runtime节点是目标设计；已有schema、CPU工具及协作文件单独标注。总体说明见 [架构文档](../architecture.md)。

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
