#!/usr/bin/env python
from __future__ import annotations


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from aasunet.config import load_config
from aasunet.data.preprocessing import preprocess_and_save_case
from aasunet.utils.misc import write_jsonl


def discover_cases(input_dir: Path) -> list[tuple[Path, Path | None]]:
    pairs = []
    for case_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        image = case_dir / "imaging.nii.gz"
        label = case_dir / "segmentation.nii.gz"
        if image.exists():
            pairs.append((image, label if label.exists() else None))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess KiTS-style folders into manifest + NPZ cases.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cases = discover_cases(args.input_dir)
    if not cases:
        raise FileNotFoundError(f"No KiTS-style cases found under {args.input_dir}")

    rows = []
    for image_path, label_path in cases:
        row = preprocess_and_save_case(
            image_path,
            label_path,
            output_dir=args.output_dir / "cases",
            intensity_clip_range=tuple(cfg.data.intensity_clip),
            target_spacing=tuple(cfg.data.target_spacing),
            zscore=cfg.data.zscore,
            label_map=cfg.data.label_map,
        )
        rows.append(row)
        print(f"[OK] preprocessed {row['case_id']} -> {row['npz_path']}")

    write_jsonl(rows, args.output_dir / "manifest.jsonl")
    print(f"[DONE] wrote manifest with {len(rows)} cases to {args.output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
