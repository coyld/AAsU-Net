from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from .overlap import binary_dice, binary_iou
from .regions import region_mask
from .surface import binary_asd, binary_hd95


def evaluate_regions(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    regions: Mapping[str, Sequence[int]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, region_classes in regions.items():
        pred_mask = region_mask(prediction, region_classes)
        tgt_mask = region_mask(target, region_classes)
        metrics[f"{name}/dice"] = binary_dice(pred_mask, tgt_mask)
        metrics[f"{name}/iou"] = binary_iou(pred_mask, tgt_mask)
        metrics[f"{name}/asd"] = binary_asd(pred_mask, tgt_mask, spacing)
        metrics[f"{name}/hd95"] = binary_hd95(pred_mask, tgt_mask, spacing)
    return metrics
