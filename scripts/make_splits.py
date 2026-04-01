#!/usr/bin/env python
from __future__ import annotations


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from aasunet.data.dataset import load_manifest
from aasunet.data.split import save_split, train_val_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val split jsonl files from a manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    train, val = train_val_split(records, val_ratio=args.val_ratio, seed=args.seed)
    save_split(train, val, args.output_dir)
    print(f"[DONE] train={len(train)} val={len(val)} -> {args.output_dir}")


if __name__ == "__main__":
    main()
