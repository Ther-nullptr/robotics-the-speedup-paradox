"""Build the offline atlas from Mermaid sources and their rendered SVGs."""

from pathlib import Path
import hashlib
import html
import json
import re

root = Path(__file__).resolve().parent
info = [
    (
        "01-architecture",
        "总体架构",
        "公共核心、机器人 adapter 与外部仿真环境怎样分工。",
        "实线表示调用或依赖，虚线表示数据/开发约束。绿色为已有本地文件，蓝色为目标代码，灰色为外部调用方或环境。",
        "architecture",
    ),
    (
        "02-async-loop",
        "异步闭环",
        "推理与环境推进重叠，输出完成后按队列规则交接动作。",
        "这是逻辑并行的手算例子，不是实测：t=0 采样，80 ms 完成，120 ms 执行动作；等待为 0，观测年龄为 120 ms。环境执行侧可由 runner、独立后端循环或编译执行实现。",
        "sequence",
    ),
    (
        "03-inference-path",
        "模型推理",
        "模型专用语义与通用算子执行后端的边界。",
        "全部推理节点均待实现。实线是请求/数据流，虚线是状态或加载依赖及离线取证；验收不进入每次推理热路径，融合只在 profile 证明有益时选择。",
        "inference",
    ),
    (
        "04-contribution-flow",
        "协作与 PR",
        "一项需求如何进入主仓和必要的模拟器 fork。",
        "已有的是模板与 CPU 校验文件。fork 按需建立；先合并 fork 取得可获取 SHA，再更新主仓依赖。远端 CI 与强制 review 尚未启用。",
        "workflow",
    ),
]
data = []
for stem, title, subtitle, note, key in info:
    source = (root / (stem + ".mmd")).read_text()
    svg = (root / "rendered" / (stem + ".svg")).read_text()
    dims = re.search(r'viewBox="([^"]+)"', svg).group(1).split()
    data.append(
        dict(
            stem=stem,
            title=title,
            subtitle=subtitle,
            note=note,
            key=key,
            source=source,
            svg=svg,
            width=float(dims[2]),
            height=float(dims[3]),
        )
    )

page = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robotics · 架构与执行逻辑</title>
<style>
:root{color-scheme:light;--ink:#172d47;--muted:#5b6e83;--line:#dbe4ee;--blue:#2563eb;--paper:#f3f6fa}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}button,a,input{font:inherit}button{cursor:pointer}button:focus-visible,a:focus-visible{outline:3px solid #93c5fd;outline-offset:3px}
header{background:#142a43;color:white;padding:28px max(24px,calc((100vw - 1440px)/2)) 26px}.eyebrow{font-size:11px;font-weight:700;letter-spacing:.17em;color:#9ec5f5;margin:0 0 10px}h1{font-size:clamp(23px,3vw,34px);letter-spacing:-.025em;margin:0 0 9px;font-weight:650}header p{margin:0;color:#c5d4e4;font-size:14px;line-height:1.7}.status{display:inline-block;padding:3px 10px;margin-left:10px;background:#234261;border:1px solid #45627c;border-radius:20px;font-size:11px;vertical-align:middle}
main{max-width:1488px;margin:auto;padding:22px 24px 40px}nav{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:18px}.tab{border:1px solid var(--line);background:white;border-radius:10px;padding:14px 16px;text-align:left;color:var(--muted);font-weight:600;display:flex;gap:12px;align-items:center}.tab .num{font-family:monospace;font-size:12px;opacity:.65}.tab[aria-selected=true]{color:#174fbc;border-color:#8db2f6;background:#eaf2ff;box-shadow:inset 0 -3px #3b82f6}
.card{background:white;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 6px 24px #243c5b07}.card-head{padding:19px 22px 15px;display:flex;align-items:flex-start;justify-content:space-between;gap:24px}h2{font-size:20px;margin:0 0 6px}.subtitle{margin:0;font-size:13px;color:var(--muted);line-height:1.7}.tools{display:flex;flex-wrap:wrap;gap:6px;align-items:center;justify-content:flex-end}.tools button{background:white;border:1px solid var(--line);border-radius:6px;padding:7px 10px;font-size:12px;color:var(--ink)}.tools button:hover{background:#eef4fc}.zoom{font-variant-numeric:tabular-nums;font-size:12px;color:var(--muted);min-width:43px;text-align:center}
.viewport{height:70vh;min-height:400px;max-height:1000px;overflow:auto;background-color:#fff;background-image:radial-gradient(#dce5ef .65px,transparent .65px);background-size:15px 15px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.stage{width:max-content;min-width:100%;padding:24px}.stage svg{display:block;margin:auto;max-width:none!important;background:white;border-radius:6px}.note{margin:0;padding:14px 22px;font-size:13px;color:var(--muted);line-height:1.8;background:#fafcfe}.foot{display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;margin:16px 2px;font-size:12px;color:var(--muted)}.source{margin-top:17px;border:1px solid var(--line);border-radius:10px;background:#fff}.source summary{padding:14px 18px;cursor:pointer;font-size:13px;font-weight:600}.source pre{padding:0 18px 18px;margin:0;overflow:auto;white-space:pre;font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;color:#334b66}.legend{display:flex;gap:15px;align-items:center;flex-wrap:wrap}.dot{display:inline-block;width:9px;height:9px;margin-right:5px;border:1px solid;border-radius:3px}.green{background:#dcfce7;border-color:#15803d}.blue{background:#dbeafe;border-color:#2563eb}.gray{background:#f1f5f9;border-color:#64748b}a{color:#245bb4;text-decoration:none}a:hover{text-decoration:underline}
@media(max-width:740px){header{padding:24px 18px}.status{margin-left:0;margin-top:8px}main{padding:16px 12px}nav{grid-template-columns:repeat(2,1fr);gap:8px}.tab{padding:12px}.card-head{flex-direction:column;gap:12px;padding:16px}.tools{justify-content:flex-start}.viewport{height:67vh;min-height:360px}.note{padding:12px 16px}.foot{line-height:1.8}}
@media print{header{background:white;color:#172d47;padding:10px 0}header p,.eyebrow{color:#456}nav,.tools,.source,.foot,.status{display:none}main{padding:0}.card{border:0;box-shadow:none}.viewport{height:auto;max-height:none;overflow:visible;border:0;background:none}.stage{padding:0;width:100%}.stage svg{width:100%!important;height:auto!important}.note{background:white}}
</style>
</head>
<body>
<header><p class="eyebrow">ROBOTICS / ARCHITECTURE ATLAS</p><h1>代码架构与执行逻辑</h1><p>从公共核心到模型、仿真与团队协作。<span class="status">目标设计 · 已实现部分单独标注</span></p></header>
<main>
<nav role="tablist" aria-label="选择架构图">__TABS__</nav>
<section class="card" role="tabpanel" id="panel" aria-labelledby="tab-architecture">
<div class="card-head"><div><h2 id="title"></h2><p class="subtitle" id="subtitle"></p></div><div class="tools" aria-label="图形工具"><button id="fit-width">适合宽度</button><button id="fit-all">查看全图</button><button id="minus" aria-label="缩小">−</button><span class="zoom" id="zoom" aria-live="polite">100%</span><button id="plus" aria-label="放大">+</button><button id="download-svg">下载 SVG</button><button id="download-source">下载 Mermaid</button></div></div>
<div class="viewport" id="viewport" tabindex="0" aria-label="图形区域，可滚动查看放大的内容"><div class="stage" id="stage"></div></div><p class="note" id="note"></p>
</section>
<div class="foot"><div class="legend"><span><i class="dot green"></i>已有本地基座</span><span><i class="dot blue"></i>目标实现 / 流程</span><span><i class="dot gray"></i>外部环境 / 数据</span></div><span>每张图的箭头和配色含义见图下注释 · 可离线查看</span></div>
<details class="source"><summary>查看当前图的 Mermaid 源码</summary><pre><code id="source"></code></pre></details>
<p class="foot">当前已有 schema、validator、CPU tests 与协作文件；模型推理、仿真运行时和业务 viewer 尚未实现。此页面只展示架构图。</p>
<noscript>请使用同目录 README.md 或 rendered/ 中的 SVG 查看完整图稿。</noscript>
</main>
<script id="diagram-data" type="application/json">__DATA__</script>
<script>
const diagrams=JSON.parse(document.getElementById('diagram-data').textContent);
const stage=document.getElementById('stage'),viewport=document.getElementById('viewport');
const tabs=[...document.querySelectorAll('[role=tab]')];let active=0,scale=1;
function applyScale(value){scale=Math.min(3,Math.max(.15,value));const d=diagrams[active],svg=stage.querySelector('svg');svg.style.width=`${d.width*scale}px`;svg.style.height=`${d.height*scale}px`;svg.setAttribute('width',d.width*scale);svg.setAttribute('height',d.height*scale);document.getElementById('zoom').textContent=`${Math.round(scale*100)}%`;}
function fit(all=false){const d=diagrams[active];applyScale(Math.min((viewport.clientWidth-48)/d.width,all?(viewport.clientHeight-48)/d.height:1.5));viewport.scrollTo(0,0);}
function show(index){active=index;const d=diagrams[index];document.getElementById('title').textContent=`${String(index+1).padStart(2,'0')} · ${d.title}`;document.getElementById('subtitle').textContent=d.subtitle;document.getElementById('note').textContent=d.note;document.getElementById('source').textContent=d.source;stage.innerHTML=d.svg;tabs.forEach((t,i)=>{t.setAttribute('aria-selected',String(i===index));t.tabIndex=i===index?0:-1;});document.getElementById('panel').setAttribute('aria-labelledby',`tab-${d.key}`);fit();}
function download(content,type,name){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
tabs.forEach((tab,i)=>{tab.onclick=()=>show(i);tab.onkeydown=e=>{let j=i;if(e.key==='ArrowRight')j=(i+1)%tabs.length;else if(e.key==='ArrowLeft')j=(i+tabs.length-1)%tabs.length;else if(e.key==='Home')j=0;else if(e.key==='End')j=tabs.length-1;else return;e.preventDefault();show(j);tabs[j].focus();};});
document.getElementById('fit-width').onclick=()=>fit();document.getElementById('fit-all').onclick=()=>fit(true);document.getElementById('minus').onclick=()=>applyScale(scale/1.2);document.getElementById('plus').onclick=()=>applyScale(scale*1.2);document.getElementById('download-svg').onclick=()=>{const d=diagrams[active];download(d.svg,'image/svg+xml',d.stem+'.svg');};document.getElementById('download-source').onclick=()=>{const d=diagrams[active];download(d.source,'text/plain;charset=utf-8',d.stem+'.mmd');};window.addEventListener('resize',()=>fit());show(0);
</script>
</body></html>
"""
tabs = "".join(
    f'<button class="tab" role="tab" id="tab-{d["key"]}" aria-controls="panel" aria-selected="false"><span class="num">{i + 1:02}</span>{html.escape(d["title"])}</button>'
    for i, d in enumerate(data)
)
page = page.replace("__TABS__", tabs).replace(
    "__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
)
(root / "architecture-atlas.html").write_text(page)
parts = [
    "# 架构与代码逻辑图\n",
    "这四张图描述目标架构与执行方式。当前已有的是 schema、validator、CPU tests 和协作文件；模型、算子运行时、模拟器和业务 viewer 均待实现。\n",
    "[打开离线交互图册](architecture-atlas.html)：切换图、缩放、查看源码和下载 SVG。HTML 内嵌全部图形，可以单独分享，无需联网。\n",
]
for i, d in enumerate(data, 1):
    parts.extend(
        [
            f"## {i}. {d['title']}\n",
            d["subtitle"] + "\n",
            f"![{d['title']}](rendered/{d['stem']}.svg)\n",
            d["note"] + "\n",
            f"[Mermaid 源码]({d['stem']}.mmd) · [SVG](rendered/{d['stem']}.svg)\n",
        ]
    )
parts.extend(
    [
        "## 维护方式\n",
        "以 `.mmd` 为编辑源，SVG 和 HTML 为导出快照。修改后重新渲染并更新图册；不要把图中的计划节点标为已支持。\n",
        "本轮使用 Mermaid CLI 11.15.0、Noto Sans CJK SC 字体和 `render-config.json`。渲染依赖只用于文档导出，不加入模型/CPU 契约运行环境。\n",
        "```bash\n# 已安装 Mermaid CLI 的文档环境中，从本目录运行\nmmdc -i 01-architecture.mmd -o rendered/01-architecture.svg \\\n  -I fig-01-architecture -c render-config.json -b white -w 1800\n```\n",
        "其余图使用对应文件名与唯一 SVG ID。浏览器路径按自己的导出环境配置。更新全部 SVG 后运行 `python build_gallery.py`，重新生成 HTML、索引与来源 hash。图册是架构说明，不是模拟器状态可视化的实现。\n",
        "对应规范：[架构](../architecture.md)、[仿真](../protocols/simulation.md)、[多模拟器后端](../protocols/simulator-backends.md)、[模型 adapter](../protocols/model-adapters.md)、[协作](../../CONTRIBUTING.md)。\n",
    ]
)
(root / "gallery.md").write_text("\n".join(parts))
manifest = {
    "mermaid_cli": "11.15.0",
    "source_of_truth": "*.mmd",
    "files": {
        d["stem"]: {
            "mermaid_sha256": hashlib.sha256(d["source"].encode()).hexdigest(),
            "svg_sha256": hashlib.sha256(d["svg"].encode()).hexdigest(),
        }
        for d in data
    },
}
(root / "render-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Wrote offline atlas, local gallery index, and source/render hashes.")
