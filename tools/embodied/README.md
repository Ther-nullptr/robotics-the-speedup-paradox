# 具身轨迹与运动指标小工具

这个目录提供独立小脚本，直接分析单条末端位置轨迹，不依赖模型、GPU、模拟器或主实验框架。数值指标使用 Python 标准库；绘图额外使用 Matplotlib。

| 脚本 | 用途 |
| --- | --- |
| [trajectory_metrics.py](trajectory_metrics.py) | 时长、路径长度、位移、速度、加速度及 jerk 指标 |
| [plot_trajectory.py](plot_trajectory.py) | 单条/多条轨迹的3D或平面投影，附速度/加速度/jerk幅值曲线 |
| [examples](examples/README.md) | 两份可复现的解析合成轨迹，非真实模型实验结果 |

## 输入约定

CSV 必须包含 `t,x,y,z` 列，每行一个笛卡尔位置样本。例如：

```csv
t,x,y,z
0.00,0.40,0.10,0.25
0.02,0.41,0.10,0.25
0.04,0.42,0.10,0.25
0.06,0.43,0.10,0.25
```

默认时间单位秒、位置单位米；可用 `--time-unit ms` 与 `--position-unit cm/mm` 转换，输出统一为 SI 单位。所有值必须有限、时间严格递增；不自动排序、插值、滤波或补零。空单元格/NaN不是丢帧处理策略，需在上游明确处理后再输入。

每个CSV是一条轨迹。若包含 `episode_id`，必须为同一非空ID；多个episode分文件处理。其他额外列不参与计算。轨迹应采用一致的坐标系，建议时间相对于采样起点，避免大epoch浮点时间损失精度。

当前只处理三维**平移位置**，不把Euler角、四元数或混合单位的关节量当成xyz。工具不会自动做坐标变换；`--frame` 用于标记输入坐标系，不会把坐标转换为world。

## 计算指标

以下命令均从 `robotics/` 根目录运行：

```bash
# 直接输出JSON摘要
python tools/embodied/trajectory_metrics.py \
  --input tools/embodied/examples/smooth.csv --frame synthetic

# 保存摘要，按需附上各阶导数的向量和有效时间点
python tools/embodied/trajectory_metrics.py \
  --input tools/embodied/examples/perturbed.csv --frame synthetic \
  --include-series --output artifacts/perturbed-metrics.json
```

输出以 `summary`、`missing_reasons` 和方法/单位元数据组织；`--include-series` 才输出各阶时间序列。数值缺失时使用null并给原因。写出文件时不覆盖输入轨迹。

## 绘图与比较

```bash
python -m pip install -r tools/embodied/requirements-plot.txt

python tools/embodied/plot_trajectory.py \
  --input tools/embodied/examples/smooth.csv tools/embodied/examples/perturbed.csv \
  --labels smooth perturbed --frame synthetic \
  --title "Synthetic trajectories" \
  --output artifacts/trajectory-comparison.png
```

使用 `--projection xy`、`xz`、`yz` 选择平面投影；默认3D。将输出后缀改为 `.svg` 可保存矢量图；`--dpi` 控制PNG分辨率。使用无GUI后端，适合服务器与批处理。

绘图输出保存在 `artifacts/` 等本地目录，按需重建，不提交生成图片。

左侧路径采用相同物理比例显示坐标，圆点/X分别标识开始/结束。右侧画导数向量的幅值，每条轨迹的显示时间单独从起点归零；这是展示对齐，不会重新采样数据。导数有自己的有效时间点，不沿用原始位置数组的索引。不可计算的曲线明确标为Unavailable，不画一条假的零线。

## jerk 的口径

对于等间隔位置样本 `x[k]`，间隔为h：

| 量 | 差分估计 | 有效时刻 | 最少点数 |
| --- | --- | --- | ---: |
| 速度 | `(x[k+1]-x[k])/h` | `t[k]+h/2` | 2 |
| 加速度 | `(x[k+2]-2*x[k+1]+x[k])/h²` | `t[k]+h` | 3 |
| jerk | `(x[k+3]-3*x[k+2]+3*x[k+1]-x[k])/h³` | `t[k]+3*h/2` | 4 |

均匀性相对首个间隔判断，默认相对容差 `1e-5`、绝对容差 `1e-9 s`。满足容差时加速度/jerk使用首个间隔h；速度始终使用各个实际时间差，因此也是区间平均速度，不宣称为某个时刻的精确瞬时速度。

非均匀数据仍可计算路径、位移、时长和区间速度；本工具不给它套等间隔的加速度/jerk公式，也不暗中用平均dt替代。点数不足时相应指标为null；单点轨迹的路径和时长为0，但速度、加速度和jerk不可用。

论文 [The Speedup Paradox，Eq.32](https://arxiv.org/pdf/2606.28529v2#page=23) 使用：

$$
\mathrm{MSJ}=\frac{1}{K-3}\sum_{k=0}^{K-4}\|j_k\|_2^2.
$$

先对xyz坐标求平方和，再对有效jerk样本求平均；不把坐标维也平均。各指标区别如下：

| 输出 | 含义 | 单位 |
| --- | --- | --- |
| `path_length_m` | 相邻位置欧氏距离之和，离散路径长度 | m |
| `displacement_m` | 起终点欧氏距离 | m |
| `mean_speed_m_s` | 路径长度/总时长，按时间加权的平均速率 | m/s |
| `peak_speed_m_s` | 最大区间速度向量幅值 | m/s |
| `acceleration_rms_m_s2` | 等间隔差分加速度的RMS幅值 | m/s² |
| `jerk_mean_squared_m2_s6` | 论文Eq.32的MSJ | m²/s⁶ |
| `jerk_rms_m_s3` | MSJ的平方根 | m/s³ |
| `jerk_peak_m_s3` | 最大差分jerk幅值 | m/s³ |

这里不计算 integrated squared jerk，也不以末端差分指标代替任务SR或控制稳定性结论。高阶差分对噪声、采样率和量化误差敏感；比较不同模型方案时，保持采样/单位/坐标及处理流程一致，并保留原始轨迹。用户先自行滤波的，应另记滤波方法与参数，不与原始轨迹指标混报。

## 小规模正确性检查

对 `t=0,1,2,3,4`：常速轨迹jerk应为0；`x=t²`加速度应为2、jerk为0；`x=t³`的jerk为6、MSJ为36。多轴 `x=(t³,2t³,0)` 的MSJ为180，可检测误除以坐标数的错误。

```bash
python -m pytest -q tests/test_trajectory_metrics.py
# 绘图测试需先安装可选绘图依赖
python -m pytest -q tests/test_plot_trajectory.py
```

未安装Matplotlib时仅跳过绘图测试，核心指标测试仍可运行。现有加速比工具仍位于 [compare_speedups.py](../compare_speedups.py)，配合 [统一指标协议](../../docs/protocols/speedup-metrics.md) 使用；轨迹平滑度和任务加速比分别记录。
