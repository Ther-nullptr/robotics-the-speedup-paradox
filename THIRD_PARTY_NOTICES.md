# 第三方来源与发布状态

本仓库未复制 TurboVLA、Jetson-PI、Jetson-PI-Edge、OxyGen 或第三方模型/模拟器实现，未包含第三方模型权重与数据集。研究参考不是源码引入。仓内的模型/环境适配与runner调用显式指定的外部资源，相关依赖和资产仍遵守各自条款。

## Cosmos＋LIBERO 外部调用边界

[Cosmos case](benchmarks/static/cosmos_libero/README.md) 通过本仓适配器调用 [NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy) 兼容源码及原生LIBERO接口，没有vendor外部实现。当前检查的Cosmos源码根许可证文件声明Apache-2.0；该声明不覆盖模型权重、数据或所有嵌套依赖。checkpoint、配套统计量/T5和VAE文件的来源与条款须分别核对，不能将源码许可证套用于资产。

当前可用Cosmos源码副本缺少 `.git`，无法证明对应某个纯上游commit。运行计划记录实际Python源码树哈希；只有存在可用Git信息时才记录commit与dirty状态。已检查的LIBERO环境来自 [lerobot-libero](https://github.com/huggingface/lerobot-libero) 安装包，须连同实际包版本、任务/初态和资产配置记录，不默认等同于其他 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) checkout。

普通用户Python当前复用已有外部环境的packages；此接入不分发该环境，也不声明从新机器可独立重建。外部源码、checkpoint、T5、VAE、模拟器任务/资产与本机配置均不进入Git；公开模板只提供 `PATH/TO` 占位路径。

## 来源记录与发布

后续引入第三方组件时，在对应 PR 记录：上游 URL、完整 commit SHA、实际文件、引入方式（依赖/子模块/vendor/重写）、原文件许可证与版权、修改说明，以及模型权重或数据的单独条款。保留原始许可文件，不能根据根 LICENSE 推断全部嵌套内容的授权。

本项目自有代码的开源许可证尚待维护者确定；当前没有自动选择或添加 LICENSE。对外发布前完成该项并更新 README。本文件只记录来源流程，不代表已审查尚未引入的资产。
