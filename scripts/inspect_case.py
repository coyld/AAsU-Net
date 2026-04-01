#!/usr/bin/env python
from __future__ import annotations


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from aasunet.data.io import load_nifti


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one NIfTI case.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--label", type=Path)
    args = parser.parse_args()

    image, _, spacing = load_nifti(args.image)
    print("image shape:", image.shape)
    print("image spacing (zyx):", spacing)
    print("image min/max:", float(image.min()), float(image.max()))
    print("image mean/std:", float(image.mean()), float(image.std()))

    if args.label:
        label, _, _ = load_nifti(args.label)
        values = {int(v): int((label == v).sum()) for v in sorted(set(label.astype(int).ravel().tolist()))}
        print("label shape:", label.shape)
        print("label histogram:", values)


if __name__ == "__main__":
    main()
