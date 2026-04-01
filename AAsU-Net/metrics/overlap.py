from __future__ import annotations

import numpy as np


def binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = float(np.logical_and(pred, target).sum())
    denom = float(pred.sum() + target.sum())
    if denom == 0:
        return 1.0
    return (2.0 * intersection + eps) / (denom + eps)


def binary_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = float(np.logical_and(pred, target).sum())
    union = float(np.logical_or(pred, target).sum())
    if union == 0:
        return 1.0
    return (intersection + eps) / (union + eps)
