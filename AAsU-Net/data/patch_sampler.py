from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def pad_to_shape(array: np.ndarray, target_shape: Sequence[int], constant_value: float = 0) -> np.ndarray:
    pads = []
    for size, target in zip(array.shape[-3:], target_shape):
        diff = max(target - size, 0)
        before = diff // 2
        after = diff - before
        pads.append((before, after))

    if array.ndim == 4:
        pad_width = [(0, 0)] + pads
    else:
        pad_width = pads
    return np.pad(array, pad_width=pad_width, mode="constant", constant_values=constant_value)


def crop_at_center(array: np.ndarray, center: Sequence[int], size: Sequence[int]) -> np.ndarray:
    slices = []
    spatial_shape = array.shape[-3:]
    for c, s, dim in zip(center, size, spatial_shape):
        start = int(round(c - s / 2))
        end = start + int(s)
        if start < 0:
            start = 0
            end = s
        if end > dim:
            end = dim
            start = dim - s
        slices.append(slice(start, end))

    if array.ndim == 4:
        return array[(slice(None), *slices)]
    return array[tuple(slices)]


def random_center(shape: Sequence[int], patch_size: Sequence[int]) -> tuple[int, int, int]:
    center = []
    for dim, patch in zip(shape, patch_size):
        half = patch // 2
        low = half
        high = max(dim - (patch - half), low + 1)
        center.append(random.randint(low, high - 1))
    return tuple(center)


@dataclass
class BalancedPatchSampler:
    patch_size: tuple[int, int, int]
    positive_ratio: float = 0.75
    foreground_labels: tuple[int, ...] = (1, 2)

    def _sample_foreground_center(self, label: np.ndarray) -> tuple[int, int, int] | None:
        mask = np.isin(label, self.foreground_labels)
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return None
        z, y, x = indices[np.random.randint(0, len(indices))]
        return int(z), int(y), int(x)

    def __call__(self, image: np.ndarray, label: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        image = pad_to_shape(image, self.patch_size, constant_value=0)
        if label is not None:
            label = pad_to_shape(label, self.patch_size, constant_value=0)

        shape = image.shape[-3:]
        center = None
        if label is not None and random.random() < self.positive_ratio:
            center = self._sample_foreground_center(label)
        if center is None:
            center = random_center(shape, self.patch_size)

        cropped_image = crop_at_center(image, center, self.patch_size)
        cropped_label = crop_at_center(label, center, self.patch_size) if label is not None else None
        return cropped_image, cropped_label
