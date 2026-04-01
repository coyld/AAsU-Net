from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from scipy.ndimage import zoom

from .io import load_nifti, save_npz_case
from ..utils.misc import ensure_dir


def clip_intensity(image: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(image, lower, upper)


def zscore_normalize(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = float(image.mean())
    std = float(image.std())
    return (image - mean) / max(std, eps)


def remap_labels(label: np.ndarray, mapping: Mapping[int, int]) -> np.ndarray:
    if not mapping:
        return label
    out = label.copy()
    for source, target in mapping.items():
        out[label == int(source)] = int(target)
    return out


def _compute_zoom_factors(current_spacing: tuple[float, float, float], target_spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(c) / float(t) for c, t in zip(current_spacing, target_spacing))


def resample_image(
    image: np.ndarray,
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    order: int = 1,
) -> np.ndarray:
    factors = _compute_zoom_factors(current_spacing, target_spacing)
    return zoom(image, zoom=factors, order=order)


def preprocess_case(
    image_path: str | Path,
    label_path: str | Path | None,
    *,
    intensity_clip_range: tuple[float, float] = (-75.0, 293.0),
    target_spacing: tuple[float, float, float] = (3.22, 1.62, 1.62),
    zscore: bool = True,
    label_map: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    image, affine, spacing = load_nifti(image_path)
    label = None
    if label_path is not None:
        label, _, label_spacing = load_nifti(label_path)
        if tuple(label_spacing) != tuple(spacing):
            raise ValueError(f"Image/label spacing mismatch: {spacing} vs {label_spacing}")

    image = clip_intensity(image, *intensity_clip_range)
    image = resample_image(image, spacing, target_spacing, order=1)
    if label is not None:
        label = resample_image(label, spacing, target_spacing, order=0).astype(np.int16)
        label = remap_labels(label, label_map or {})
    if zscore:
        image = zscore_normalize(image)

    return {
        "image": image.astype(np.float32),
        "label": label.astype(np.int16) if label is not None else None,
        "affine": affine.astype(np.float32),
        "spacing": tuple(float(v) for v in target_spacing),
        "case_id": Path(image_path).parent.name if Path(image_path).parent.name else Path(image_path).stem,
        "source_image": str(image_path),
        "source_label": str(label_path) if label_path is not None else None,
        "shape": tuple(int(v) for v in image.shape),
    }


def preprocess_and_save_case(
    image_path: str | Path,
    label_path: str | Path | None,
    output_dir: str | Path,
    *,
    intensity_clip_range: tuple[float, float],
    target_spacing: tuple[float, float, float],
    zscore: bool,
    label_map: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    case = preprocess_case(
        image_path,
        label_path,
        intensity_clip_range=intensity_clip_range,
        target_spacing=target_spacing,
        zscore=zscore,
        label_map=label_map,
    )
    output_dir = ensure_dir(output_dir)
    case_id = case["case_id"]
    npz_path = output_dir / f"{case_id}.npz"
    save_npz_case(
        npz_path,
        case["image"],
        case["label"],
        case["affine"],
        case["spacing"],
        case_id,
    )
    return {
        "case_id": case_id,
        "npz_path": str(npz_path),
        "image_path": str(image_path),
        "label_path": str(label_path) if label_path is not None else None,
        "spacing": list(case["spacing"]),
        "shape": list(case["shape"]),
    }
