# 统一实验记录

[study_records.py](../tools/study_records.py) 将多个批次登记到一个本地索引。它重新读取已完成运行的原始episode、coverage和请求记录，计算成功率、成功回合平均chunk及控制步数；推理耗时通过显式的测量目录和variant ID关联。工具只使用CPU，不启动模型、模拟器或下载。

每个批次保留自己的任务、精度、调度、场景、seed、代码身份与原始目录。工具不自动合并不同批次的平均值；复用端点必须标记来源，避免将同一批episode当成新增实验。只有实际记录了请求计数才输出成功平均chunk，零成功时为空，不从控制步数推造调用次数。Cosmos会额外核对请求序列及控制边界。

## 批次描述

在实验输出目录维护一个 `study.json`；相对路径以该文件所在目录解析，绝对路径也可使用。示例中的路径需替换为实际产物：

```json
{
  "format": "robotics-study-v1",
  "id": "robocasa-progressive-002",
  "case_id": "cosmos_robocasa",
  "description": "Additional RoboCasa tasks, synchronous progressive quantization",
  "code_commit": "actual-runtime-git-commit",
  "cells": [
    {
      "id": "TurnOnMicrowave/tier-00",
      "task": "TurnOnMicrowave",
      "role": "candidate",
      "tier": 0,
      "run_dir": "TurnOnMicrowave/tier-00/attempt-001",
      "expected_episodes": 10,
      "timing": {
        "run_dir": "inference-timing/TurnOnMicrowave",
        "variant_id": "tier-00"
      }
    }
  ]
}
```

`cells`列出完整计划，包括尚未启动的组。基线使用 `role=reference`，不指定tier。`timing`可省略，表示该组没有请求计时；提供但尚未测量时标为planned，不填0。复用旧质量结果时设置 `quality_reused_from`，原始运行路径保持指向旧产物。若同一批次引用多个代码版本，可逐cell提供 `code_commit`；缺少历史提交号时留空，源manifest和case指纹仍保留。

Cosmos tier会按固定协议核对280个候选Linear的实际backend和逐层位宽，不能只修改tier标签。计时必须来自已完成的cohort，匹配checkpoint路径、模型runtime、采样步数、预测horizon、优化开关和档位；整数配置还核对scope与tactic；中位数/P95从原始samples_ms重新计算。有保存的初始观测时，只接受该回合NPZ作为第一个实际计时输入，其余验证输入不能代替。推理时间的起止范围由[inference benchmark](../benchmarks/inference/README.md)定义，仿真、录像和整组wall time不进入此表。

## 登记和刷新

```bash
python tools/study_records.py \
  --study runs/static/cosmos_robocasa/progressive_quantization_002/study.json \
  --registry runs/records
```

同一命令可在组完成后再次执行。登记时同时刷新所有已登记批次；已有ID不能被另一个描述文件占用，同一个运行不能在单一批次重复出现。跨批次复用必须指向已登记且使用同一原始运行的首个批次，不允许将同一来源登记为两份新增数据。已完成标记与原始覆盖不一致、请求计数错误或计时配置不匹配会报错。未完成/失败的组保留状态，指标留空。

输出：

| 文件 | 内容 |
| --- | --- |
| `index.json` | 批次目录、描述文件哈希和完整性状态 |
| `records.json` | 完整统一记录，包括来源文件哈希 |
| `records.csv` | 按批次、任务、档位展开的比较表 |
| `README.md` | 每批质量与计时的完成进度 |

这是记录层，不代替case的GPU验收，也不自动解释跨批次差异。初始观测快照协议、资源复用和硬件差异仍须保留在每次运行的manifest中。Linux/macOS文件锁保护索引写入；描述文件应由同一个调度器原子更新。代码、测试和本说明进入Git；`runs/`中的索引、视频、图表、checkpoint及实验数据留在本地。
