# LingBot-VA KV read-window 剪枝设计

本文档定义 [The Speedup Paradox](https://arxiv.org/abs/2606.28529v2) 中 LingBot-VA＋RoboTwin KV-cache pruning 的公开CPU契约。当前 PR 只提供确定性的参考选择器和论文配置，不修改 `lingbot_server.py` 或外部 LingBot 源码，不包含新的GPU时延、动作一致性或闭环成功率证据。

## 方法边界

原生RoboTwin配置使用 `attn_window=72`，最多保留36个历史视觉latent chunk和36个历史action chunk。Read-window pruning不改变这个cache的分配、写入、替换或生命周期，只在action denoise self-attention读取历史K/V前，分别选择视觉和动作流中最近的token：

```text
R_t = Recent_W_lat(valid latent slots) ∪ Recent_W_act(valid action slots)
```

因此它减少的是attention可读K/V集合，而不是已分配cache容量。不得把选择后的token数称为显存降幅；也不能仅凭CPU选择器宣称GPU或任务级加速。

论文Table 5的操作点为：

| 配置 | 启用read window | 视觉token上限 | 动作token上限 | 相对原生可读KV | 视觉历史chunk | 动作历史chunk |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | 否 | 8640 | 1152 | 100.0% | 36.0 | 36.0 |
| p1 | 是 | 1024 | 256 | 13.1% | 4.3 | 8.0 |
| p2 | 是 | 512 | 128 | 6.5% | 2.1 | 4.0 |
| p3 | 是 | 256 | 64 | 3.3% | 1.1 | 2.0 |
| p4 | 是 | 128 | 32 | 1.6% | 0.5 | 1.0 |

这里每个视觉latent chunk为240个KV token，每个action chunk为32个。比例是 `(视觉上限 + 动作上限) / (8640 + 1152)`；它描述可读预算，不是实测时延或FLOPs比例。

## CPU参考契约

[kv_read_window.py](kv_read_window.py) 接收cache slot元数据，不读取张量。每个slot包含：

- `slot`：非负且唯一的物理slot编号；
- `source`：`latent`、`action` 或兼容旧cache的 `unknown`；
- `generation`：该slot的单调写入代次；已知来源必须非负；
- `current`：可选安全标记，表示本步新写入或必须保留的slot。

选择规则如下：

1. `latent` 和 `action` 分别应用自己的token预算；预算0表示该来源不设上限。
2. 按 `generation` 从新到旧选择；同代次按较小 `slot` 确定性打破平局。
3. `unknown` 和 `current` 始终保留，因此安全slot可能使最终数量超过预算。
4. 输出slot升序排列，便于运行时稳定gather；选择器不修改输入和cache。
5. `base` 关闭选择，保留原生cache中的全部有效slot。

运行合成例子：

```bash
python benchmarks/static/lingbot_robotwin/kv_read_window.py \
  --input benchmarks/static/lingbot_robotwin/kv_read_window.synthetic.json
```

输入也可用 `"preset": "p1"` 到 `"p4"` 代替 `enabled` 和两个显式预算；preset与显式字段同时出现会被拒绝。输出携带 `synthetic`、完整配置、保留/丢弃slot和分来源计数。仓内例子仅验证边界，不是模型实测。

## 运行时接入与验收条件

后续GPU runtime应作为单独PR接入，并满足以下边界：

- 默认关闭；原生baseline关闭时逐元素保持动作输出一致。
- cache写入时显式记录来源和generation；缺少来源的旧slot采用保留策略。
- 仅限制action denoise self-attention读取的历史K/V；当前query对应的K/V仍参与本步attention。
- 在进入transformer blocks前选择一次indices，并在各层复用；不得每层重复排序后把选择开销隐藏在profile之外。
- manifest记录实际启用状态、视觉/动作预算、原生 `attn_window`、dtype、backend和模型源码身份。
- 固定observation/noise/seed检查动作向量和多步误差；闭环使用相同episode seeds比较SR、成功回合chunk数、稳定性和任务时间。
- 性能验证同时报告选择/gather开销、完整action inference和任务层结果；attention kernel变快不能单独作为端到端加速结论。

当前仓库的LingBot worker从调用者显式提供的外部源码加载模型，因此本契约不自动修改该源码，也不把本地实验分支、checkpoint或结果带入公共仓库。

## English summary

This is a CPU reference contract for the paper's LingBot-VA KV read-window pruning. It preserves native cache allocation and writes while restricting the historical K/V visible to action-denoise self-attention. Latent and action streams have independent recent-token budgets; unknown and current slots are preserved. The paper presets are built in, but the runtime remains unchanged and the feature is not presented as GPU-, memory-, or task-level acceleration. A later runtime PR must be opt-in and provide fixed-input numerical checks, full-inference timing, and matched-seed closed-loop evidence.
