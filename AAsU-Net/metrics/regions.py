from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_REGION_SETS: dict[str, dict[str, tuple[int, ...]]] = {
    "kits19": {
        "kidney": (1, 2),
        "tumor": (2,),
    },
    "kits21": {
        "kidney_and_masses": (1, 2, 3),
        "kidney_mass": (2, 3),
        "tumor": (2,),
        "cyst": (3,),
    },
    "generic_multiclass": {},
}


def get_region_definitions(dataset: str, num_classes: int) -> dict[str, tuple[int, ...]]:
    dataset = dataset.lower()
    if dataset in DEFAULT_REGION_SETS:
        return DEFAULT_REGION_SETS[dataset]
    if dataset == "generic":
        return {f"class_{idx}": (idx,) for idx in range(1, num_classes)}
    return {f"class_{idx}": (idx,) for idx in range(1, num_classes)}


def region_mask(label: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    return np.isin(label, list(classes))
