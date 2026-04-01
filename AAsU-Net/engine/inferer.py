from __future__ import annotations

import itertools
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F


def _compute_scan_intervals(image_size: Sequence[int], roi_size: Sequence[int], overlap: float) -> list[list[int]]:
    intervals: list[list[int]] = []
    for img_dim, roi_dim in zip(image_size, roi_size):
        if img_dim <= roi_dim:
            intervals.append([0])
            continue
        step = max(int(roi_dim * (1.0 - overlap)), 1)
        starts = list(range(0, img_dim - roi_dim + 1, step))
        if starts[-1] != img_dim - roi_dim:
            starts.append(img_dim - roi_dim)
        intervals.append(starts)
    return intervals


def _gaussian_weight_map(roi_size: Sequence[int], device: torch.device) -> torch.Tensor:
    coords = [torch.arange(size, device=device, dtype=torch.float32) for size in roi_size]
    zz, yy, xx = torch.meshgrid(*coords, indexing="ij")
    center = [(size - 1) / 2.0 for size in roi_size]
    sigmas = [max(size * 0.125, 1.0) for size in roi_size]
    weight = torch.exp(
        -(((zz - center[0]) ** 2) / (2 * sigmas[0] ** 2)
          + ((yy - center[1]) ** 2) / (2 * sigmas[1] ** 2)
          + ((xx - center[2]) ** 2) / (2 * sigmas[2] ** 2))
    )
    return weight.clamp_min(1e-5)


@torch.no_grad()
def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Sequence[int],
    sw_batch_size: int,
    predictor: Callable[[torch.Tensor], torch.Tensor | dict],
    overlap: float = 0.5,
    use_gaussian: bool = True,
) -> torch.Tensor:
    if inputs.ndim != 5:
        raise ValueError(f"Expected input in [B, C, D, H, W], got {inputs.shape}")
    if inputs.shape[0] != 1:
        raise ValueError("This implementation expects batch size 1 for full-volume inference.")

    device = inputs.device
    roi_size = tuple(int(v) for v in roi_size)
    image_size = tuple(int(v) for v in inputs.shape[2:])
    scan_intervals = _compute_scan_intervals(image_size, roi_size, overlap)
    weight_map = _gaussian_weight_map(roi_size, device) if use_gaussian else torch.ones(roi_size, device=device)

    output_acc = None
    count_acc = None

    windows = []
    locations = []

    for start in itertools.product(*scan_intervals):
        slices = tuple(slice(s, s + roi) for s, roi in zip(start, roi_size))
        patch = inputs[(slice(None), slice(None), *slices)]
        if patch.shape[2:] != roi_size:
            patch = F.interpolate(patch, size=roi_size, mode="trilinear", align_corners=False)
        windows.append(patch)
        locations.append(start)

        if len(windows) == sw_batch_size:
            output_acc, count_acc = _accumulate_windows(
                windows, locations, predictor, output_acc, count_acc, weight_map, image_size
            )
            windows, locations = [], []

    if windows:
        output_acc, count_acc = _accumulate_windows(
            windows, locations, predictor, output_acc, count_acc, weight_map, image_size
        )

    return output_acc / count_acc.clamp_min(1e-8)


def _accumulate_windows(
    windows: list[torch.Tensor],
    locations: list[Sequence[int]],
    predictor: Callable[[torch.Tensor], torch.Tensor | dict],
    output_acc: torch.Tensor | None,
    count_acc: torch.Tensor | None,
    weight_map: torch.Tensor,
    image_size: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = torch.cat(windows, dim=0)
    pred = predictor(batch)
    if isinstance(pred, dict):
        pred = pred["logits"]

    if output_acc is None:
        output_acc = pred.new_zeros((1, pred.shape[1], *image_size))
        count_acc = pred.new_zeros((1, 1, *image_size))

    for idx, start in enumerate(locations):
        z, y, x = start
        weighted = pred[idx : idx + 1] * weight_map[None, None]
        output_acc[:, :, z : z + pred.shape[2], y : y + pred.shape[3], x : x + pred.shape[4]] += weighted
        count_acc[:, :, z : z + pred.shape[2], y : y + pred.shape[3], x : x + pred.shape[4]] += weight_map[None, None]

    return output_acc, count_acc
