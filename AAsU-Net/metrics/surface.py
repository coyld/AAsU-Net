from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _surface(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    return np.logical_xor(mask, eroded)


def _surface_distances(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    pred = pred.astype(bool)
    target = target.astype(bool)

    pred_surface = _surface(pred)
    target_surface = _surface(target)

    if pred_surface.sum() == 0 and target_surface.sum() == 0:
        return np.array([0.0], dtype=np.float32)
    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return np.array([np.inf], dtype=np.float32)

    dt_target = distance_transform_edt(~target_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    pred_to_target = dt_target[pred_surface]
    target_to_pred = dt_pred[target_surface]
    return np.concatenate([pred_to_target, target_to_pred]).astype(np.float32)


def binary_asd(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, float, float]) -> float:
    distances = _surface_distances(pred, target, spacing)
    if np.isinf(distances).any():
        return float("inf")
    return float(distances.mean())


def binary_hd95(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, float, float]) -> float:
    distances = _surface_distances(pred, target, spacing)
    if np.isinf(distances).any():
        return float("inf")
    return float(np.percentile(distances, 95))
