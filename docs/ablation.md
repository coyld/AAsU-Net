# 消融实验支持

为了更贴近论文的实验组织方式，仓库提供了五种结构开关，用于快速构建一组与论文叙述一致的对比模型：

- `standard`：标准 3D 卷积块。
- `separable`：仅使用 `3x1x1 -> 1x3x3` 的空间可分离卷积块。
- `sum`：标准 3D 卷积与可分离卷积并联后直接相加，不使用自适应权重。
- `aas`：AAs-conv，自适应加权融合双分支。
- `use_csff`：控制是否启用跨尺度特征融合。

## 推荐对应关系

- `configs/ablation/model1_standard3d.yaml`
- `configs/ablation/model2_separable.yaml`
- `configs/ablation/model3_parallel_sum.yaml`
- `configs/ablation/model4_aas_no_csff.yaml`
- `configs/ablation/model5_aasunet_full.yaml`

## 命令示例

```bash
python scripts/train.py --config configs/ablation/model5_aasunet_full.yaml
python scripts/train.py --config configs/ablation/model1_standard3d.yaml
```

这些配置不会偏离论文主方法，只是把论文中用于说明模块贡献的结构差异显式参数化，方便在 GitHub 仓库中复现实验组织过程。
