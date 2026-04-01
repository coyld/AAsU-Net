# 数据说明

## KiTS19
推荐目录：

```text
data/kits19_raw/
└─ case_xxxxx/
   ├─ imaging.nii.gz
   └─ segmentation.nii.gz
```

标签通常为：
- 0: background
- 1: kidney
- 2: tumor

## KiTS21
推荐目录：

```text
data/kits21_raw/
└─ case_xxxxx/
   ├─ imaging.nii.gz
   └─ segmentation.nii.gz
```

标签通常为：
- 0: background
- 1: kidney
- 2: tumor
- 3: cyst

## 预处理流程
本仓库默认使用论文中的预处理流程：
1. intensity clipping `[-75, 293]`
2. spacing resample `[3.22, 1.62, 1.62]`
3. z-score
4. 训练阶段随机裁块

## 关于 KiTS21 cyst 的处理
论文正文没有完全写死 KiTS21 的训练标签策略，因此仓库提供两种方式：
1. 保留 4 类标签，按官方数据集方式训练
2. 使用 `data.label_map` 把 cyst 映射到 kidney 或其它类别

你可以根据自己的论文实验口径进行配置。
