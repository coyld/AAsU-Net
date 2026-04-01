from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

try:
    import nibabel as nib
except Exception:  # pragma: no cover - optional dependency
    nib = None


def _require_nibabel() -> None:
    if nib is None:  # pragma: no cover - tested indirectly through usage
        raise ImportError(
            "nibabel is required for NIfTI IO. Please install it with `pip install nibabel`."
        )


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    _require_nibabel()
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    # nibabel returns xyz; convert to zyx to match array order after transpose below
    spacing_xyz = img.header.get_zooms()[:3]
    spacing_zyx = tuple(float(v) for v in spacing_xyz[::-1])
    # convert from xyz to zyx for a more natural [D, H, W] convention
    data = np.transpose(data, (2, 1, 0))
    return data, img.affine, spacing_zyx


def save_nifti(array: np.ndarray, affine: np.ndarray, path: str | Path) -> None:
    _require_nibabel()
    array = np.asarray(array)
    if array.ndim == 3:
        export = np.transpose(array, (2, 1, 0))
    elif array.ndim == 4:
        export = np.transpose(array, (0, 3, 2, 1))
    else:
        raise ValueError(f"Expected 3D or 4D array, got {array.shape}")
    nii = nib.Nifti1Image(export, affine=affine)
    nib.save(nii, str(path))


def load_npz_case(path: str | Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    case = {
        "image": payload["image"].astype(np.float32),
        "label": payload["label"].astype(np.int16) if "label" in payload else None,
        "affine": payload["affine"] if "affine" in payload else np.eye(4, dtype=np.float32),
        "spacing": tuple(payload["spacing"].tolist()) if "spacing" in payload else None,
        "case_id": payload["case_id"].item() if "case_id" in payload else Path(path).stem,
    }
    return case


def save_npz_case(
    path: str | Path,
    image: np.ndarray,
    label: np.ndarray | None,
    affine: np.ndarray,
    spacing: tuple[float, float, float],
    case_id: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "image": image.astype(np.float32),
        "affine": np.asarray(affine, dtype=np.float32),
        "spacing": np.asarray(spacing, dtype=np.float32),
        "case_id": np.asarray(case_id),
    }
    if label is not None:
        arrays["label"] = label.astype(np.int16)
    np.savez_compressed(path, **arrays)
