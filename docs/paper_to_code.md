# 论文到代码的映射

## 1. AAs-conv

论文描述：
- 一个空间可分离卷积分支：`3x1x1 -> 1x3x3`
- 一个标准 3D 卷积分支：`3x3x3`
- 两个分支先相加得到融合描述 `xu`
- 对 `xu` 做 GAP，得到通道描述符
- 经共享 squeeze 和两个独立 excitation 得到两组分支权重
- 使用 softmax 对两个分支的权重归一化
- 最终输出 `y = δa * xa + δb * xb`

代码实现：
- `aasunet/models/layers.py::AAsConv`

## 2. 6-stage encoder / 5-stage decoder

论文描述：
- 编码器 6 层，每层 2 个 AAs-conv
- 解码器 5 层，每层 1 个 AAs-conv
- 编码器通道数：24, 48, 96, 192, 320, 320
- 解码器通道数：320, 192, 96, 48, 24
- 第 2 个编码阶段只在平面方向下采样，后续阶段在全部方向下采样
- 最后一个解码阶段只在平面方向上采样

代码实现：
- `aasunet/models/aasunet.py::AAsUNet`

## 3. CSFF

论文描述：
- 从第一层编码器输出 `En1` 取全分辨率细节特征
- 对每个更深层编码器输出 `En_i`：
  1. 对 `En1` 做 average pooling 对齐空间尺寸
  2. 用 `1x1x1` 卷积对齐通道
  3. 与 `En_i` 做逐元素相加

代码实现：
- `aasunet/models/layers.py::CSFFProjector`

## 4. Deep supervision

论文描述：
- 除最深层外，每个解码阶段均输出辅助分割图
- 用于改善梯度传播和多层表示学习

代码实现：
- `aasunet/models/layers.py::SegmentationHead`
- `aasunet/losses/deep_supervision.py::DeepSupervisionLoss`

## 5. 训练与推理

论文描述：
- Patch size: `64 x 128 x 128`
- Batch size: `2`
- SGD
- 初始学习率 `0.01`
- Poly decay
- 最多 `1000` epoch
- 每个 epoch `250` iteration
- patience `20`
- Dice + CE
- online augmentation
- sliding-window 风格全卷积推理（论文虽未展开写，但 3D patch 训练下工程实现通常需要该能力）

代码实现：
- `aasunet/engine/trainer.py`
- `aasunet/optim/schedulers.py`
- `aasunet/engine/inferer.py`


## 消融支持

为了贴合论文中的模块对比，工程额外暴露了 `conv_mode` 和 `use_csff` 两个配置项，用于切换标准 3D 卷积、可分离卷积、并联直接求和以及完整 AAs-conv，并控制是否启用 CSFF。
