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

from aasunet.config import load_config
from aasunet.models.factory import build_model, count_parameters
from aasunet.utils.misc import human_readable_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Print parameter count and optional FLOPs.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    params = count_parameters(model)
    print(f"Parameters: {params} ({human_readable_count(params)})")

    try:
        from thop import profile
        x = torch.randn(1, cfg.data.input_channels, *cfg.data.patch_size)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        print(f"Approx FLOPs: {flops} ({human_readable_count(flops)})")
    except Exception as exc:
        print("FLOPs not available (install `thop` if needed).")
        print("Reason:", exc)


if __name__ == "__main__":
    main()
