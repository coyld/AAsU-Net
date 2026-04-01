# 训练说明

## 默认超参数
- Optimizer: SGD
- LR: 0.01
- Scheduler: Poly
- Batch size: 2
- Patch size: 64x128x128
- Epochs: 1000
- Iterations per epoch: 250
- Early stopping patience: 20

## Deep supervision
默认权重：
`[1.0, 0.5, 0.25, 0.125, 0.0625]`

如果你更强调浅层辅助头影响，可以把前两层权重调高；如果更希望和主头保持一致，可以只保留前 2-3 层。

## AMP
训练代码默认在 CUDA 可用时启用 AMP；CPU 环境下会自动关闭。

## 验证指标
当前实现支持：
- Dice
- IoU
- ASD
- HD95

## 常见建议
1. 先跑 KiTS19，确定训练流程没问题。
2. 再跑 KiTS21，并确认 cyst 标签策略。
3. 如果显存不够，优先降低 batch size，而不是修改网络结构。
4. 如果 patch 太小影响肿瘤完整性，可以适当提高 foreground 采样比率。
