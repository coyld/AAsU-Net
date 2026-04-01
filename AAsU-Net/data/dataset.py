from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import ExperimentConfig
from ..utils.misc import read_jsonl
from .io import load_nifti, load_npz_case
from .patch_sampler import BalancedPatchSampler
from .preprocessing import clip_intensity, remap_labels, resample_image, zscore_normalize
from .transforms import Compose, RandomBrightness, RandomContrast, RandomFlip3D, RandomGamma, RandomRotate903D, RandomSmallRotation3D, RandomZoom3D, ToTensor


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _load_case_from_manifest(entry: Mapping[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    if entry.get("npz_path"):
        case = load_npz_case(entry["npz_path"])
        case["case_id"] = entry.get("case_id", case["case_id"])
        return case

    image, affine, spacing = load_nifti(entry["image_path"])
    label = None
    if entry.get("label_path"):
        label, _, _ = load_nifti(entry["label_path"])

    image = clip_intensity(image, *cfg.data.intensity_clip)
    image = resample_image(image, tuple(spacing), tuple(cfg.data.target_spacing), order=1)
    if cfg.data.zscore:
        image = zscore_normalize(image)
    if label is not None:
        label = resample_image(label, tuple(spacing), tuple(cfg.data.target_spacing), order=0).astype(np.int16)
        label = remap_labels(label, cfg.data.label_map)

    return {
        "image": image.astype(np.float32),
        "label": label.astype(np.int16) if label is not None else None,
        "affine": affine.astype(np.float32),
        "spacing": tuple(cfg.data.target_spacing),
        "case_id": entry.get("case_id", Path(entry["image_path"]).parent.name),
    }


class CTVolumeDataset(Dataset):
    def __init__(
        self,
        manifest: Sequence[Mapping[str, Any]],
        cfg: ExperimentConfig,
        training: bool = True,
    ) -> None:
        self.records = list(manifest)
        self.cfg = cfg
        self.training = training
        self.cache: dict[int, dict[str, Any]] = {}
        self.sampler = BalancedPatchSampler(
            patch_size=tuple(cfg.data.patch_size),
            positive_ratio=cfg.data.positive_sample_ratio,
            foreground_labels=tuple(cfg.data.foreground_labels),
        )

        aug = []
        if training and cfg.augmentation.enabled:
            aug = [
                RandomFlip3D(cfg.augmentation.flip_prob),
                RandomRotate903D(cfg.augmentation.rotate90_prob),
                RandomSmallRotation3D(cfg.augmentation.small_rotate_prob, cfg.augmentation.rotate_limit_deg),
                RandomZoom3D(cfg.augmentation.zoom_prob, tuple(cfg.augmentation.zoom_range)),
                RandomBrightness(cfg.augmentation.brightness_prob, cfg.augmentation.brightness_delta),
                RandomContrast(cfg.augmentation.contrast_prob, tuple(cfg.augmentation.contrast_range)),
                RandomGamma(cfg.augmentation.gamma_prob, tuple(cfg.augmentation.gamma_range)),
            ]
        self.transforms = Compose([*aug, ToTensor()])

    def __len__(self) -> int:
        return len(self.records)

    def _fetch_case(self, idx: int) -> dict[str, Any]:
        if self.cfg.data.cache_in_memory and idx in self.cache:
            return self.cache[idx]
        case = _load_case_from_manifest(self.records[idx], self.cfg)
        if self.cfg.data.cache_in_memory:
            self.cache[idx] = case
        return case

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case = self._fetch_case(idx)
        image = case["image"]
        label = case["label"]
        case_id = case["case_id"]

        if image.ndim == 3:
            image = image[None, ...]
        sample: dict[str, Any] = {"image": image, "label": label}

        if self.training:
            sample["image"], sample["label"] = self.sampler(sample["image"], sample["label"])
            sample["meta"] = {"case_id": case_id}
            return self.transforms(sample)

        sample["meta"] = {
            "case_id": case_id,
            "affine": case["affine"],
            "spacing": case["spacing"],
        }
        return self.transforms(sample)
