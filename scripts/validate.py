#!/usr/bin/env python
from __future__ import annotations


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aasunet.config import load_config
from aasunet.data.dataset import CTVolumeDataset, load_manifest
from aasunet.engine.evaluator import evaluate_loader
from aasunet.models.factory import build_model
from aasunet.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AAsU-Net checkpoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")

    dataset = CTVolumeDataset(load_manifest(args.manifest), cfg, training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=max(cfg.data.num_workers // 2, 0))

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    metrics = evaluate_loader(model, loader, cfg, device, output_dir=None)
    for k, v in sorted(metrics.items()):
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
