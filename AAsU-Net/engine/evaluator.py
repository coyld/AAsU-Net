from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..metrics.aggregator import MetricAggregator
from ..metrics.regions import get_region_definitions
from ..metrics.segmentation import evaluate_regions
from ..utils.visualization import save_slice_triplet
from .inferer import sliding_window_inference


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
    output_dir: str | Path | None = None,
) -> dict[str, float]:
    model.eval()
    aggregator = MetricAggregator()
    dataset_name = "kits21" if cfg.data.num_classes >= 4 else "kits19"
    all_regions = get_region_definitions(dataset_name, cfg.data.num_classes)
    regions = {name: all_regions[name] for name in cfg.validation.regions if name in all_regions}
    if not regions:
        regions = all_regions

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        image = batch["image"].to(device)
        label = batch.get("label")
        meta = batch.get("meta", {})
        case_id = meta["case_id"][0] if isinstance(meta["case_id"], list) else meta["case_id"]
        affine = meta.get("affine")
        spacing = meta.get("spacing")
        if isinstance(spacing, list):
            spacing = tuple(float(v) for v in spacing[0])
        elif isinstance(spacing, tuple):
            spacing = tuple(float(v) for v in spacing)
        else:
            spacing = tuple(float(v) for v in cfg.data.target_spacing)

        logits = sliding_window_inference(
            image,
            roi_size=cfg.data.patch_size,
            sw_batch_size=cfg.validation.sliding_window_batch,
            predictor=model,
            overlap=cfg.validation.overlap,
            use_gaussian=cfg.validation.use_gaussian,
        )
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

        if label is None:
            continue
        target = label.cpu().numpy()[0]
        metrics = evaluate_regions(pred, target, spacing, regions)
        aggregator.update(metrics)

        if output_dir is not None and cfg.validation.save_predictions:
            save_slice_triplet(
                image.cpu().numpy()[0],
                target,
                pred,
                output_dir / f"{case_id}_preview.png",
            )

    return aggregator.summary()
