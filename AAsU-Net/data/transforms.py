from __future__ import annotations

import math
import random
from typing import Callable, Iterable, List, MutableMapping

import numpy as np
import torch
from scipy.ndimage import rotate, zoom

from .patch_sampler import pad_to_shape


Sample = MutableMapping[str, np.ndarray]


class Compose:
    def __init__(self, transforms: Iterable[Callable[[Sample], Sample]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomFlip3D:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, sample: Sample) -> Sample:
        for axis in range(3):
            if random.random() < self.prob:
                sample["image"] = np.flip(sample["image"], axis=axis + 1).copy()
                if sample.get("label") is not None:
                    sample["label"] = np.flip(sample["label"], axis=axis).copy()
        return sample


class RandomRotate903D:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob
        self.axes = [(1, 2), (1, 3), (2, 3)]

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample
        k = random.randint(0, 3)
        img_axes = random.choice(self.axes)
        sample["image"] = np.rot90(sample["image"], k=k, axes=img_axes).copy()
        if sample.get("label") is not None:
            label_axes = tuple(a - 1 for a in img_axes)
            sample["label"] = np.rot90(sample["label"], k=k, axes=label_axes).copy()
        return sample


class RandomSmallRotation3D:
    def __init__(self, prob: float = 0.2, limit_deg: float = 15.0) -> None:
        self.prob = prob
        self.limit_deg = limit_deg
        self.axes = [(1, 2), (1, 3), (2, 3)]

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample
        angle = random.uniform(-self.limit_deg, self.limit_deg)
        axes = random.choice(self.axes)
        sample["image"] = rotate(
            sample["image"],
            angle=angle,
            axes=axes,
            reshape=False,
            order=1,
            mode="nearest",
        ).astype(np.float32)
        if sample.get("label") is not None:
            label_axes = tuple(a - 1 for a in axes)
            sample["label"] = rotate(
                sample["label"],
                angle=angle,
                axes=label_axes,
                reshape=False,
                order=0,
                mode="nearest",
            ).astype(np.int16)
        return sample


class RandomZoom3D:
    def __init__(self, prob: float = 0.2, zoom_range: tuple[float, float] = (0.9, 1.1)) -> None:
        self.prob = prob
        self.zoom_range = zoom_range

    def _resize_back(self, array: np.ndarray, original_shape: tuple[int, ...], order: int) -> np.ndarray:
        zoom_factors = [o / n for o, n in zip(original_shape, array.shape)]
        return zoom(array, zoom=zoom_factors, order=order).astype(array.dtype)

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample
        factor = random.uniform(*self.zoom_range)
        image = sample["image"]
        label = sample.get("label")
        orig_image_shape = image.shape
        img_zoom = [1.0, factor, factor, factor]
        sample["image"] = zoom(image, zoom=img_zoom, order=1).astype(np.float32)
        sample["image"] = self._resize_back(sample["image"], orig_image_shape, order=1).astype(np.float32)
        if label is not None:
            orig_label_shape = label.shape
            sample["label"] = zoom(label, zoom=[factor, factor, factor], order=0).astype(np.int16)
            sample["label"] = self._resize_back(sample["label"], orig_label_shape, order=0).astype(np.int16)
        return sample


class RandomBrightness:
    def __init__(self, prob: float = 0.15, delta: float = 0.1) -> None:
        self.prob = prob
        self.delta = delta

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.prob:
            shift = random.uniform(-self.delta, self.delta)
            sample["image"] = (sample["image"] + shift).astype(np.float32)
        return sample


class RandomContrast:
    def __init__(self, prob: float = 0.15, contrast_range: tuple[float, float] = (0.9, 1.1)) -> None:
        self.prob = prob
        self.contrast_range = contrast_range

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.prob:
            factor = random.uniform(*self.contrast_range)
            mean = float(sample["image"].mean())
            sample["image"] = ((sample["image"] - mean) * factor + mean).astype(np.float32)
        return sample


class RandomGamma:
    def __init__(self, prob: float = 0.15, gamma_range: tuple[float, float] = (0.7, 1.5)) -> None:
        self.prob = prob
        self.gamma_range = gamma_range

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.prob:
            gamma = random.uniform(*self.gamma_range)
            image = sample["image"]
            image_min, image_max = float(image.min()), float(image.max())
            if image_max - image_min > 1e-8:
                normalized = (image - image_min) / (image_max - image_min)
                adjusted = normalized ** gamma
                sample["image"] = (adjusted * (image_max - image_min) + image_min).astype(np.float32)
        return sample


class ToTensor:
    def __call__(self, sample: Sample) -> dict[str, torch.Tensor | dict]:
        output: dict[str, torch.Tensor | dict] = {
            "image": torch.from_numpy(sample["image"].astype(np.float32)),
            "meta": sample.get("meta", {}),
        }
        if sample.get("label") is not None:
            output["label"] = torch.from_numpy(sample["label"].astype(np.int64))
        return output
