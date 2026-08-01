#!/usr/bin/env python3
"""Prepare official GRES files for the current LISAt ReasonSegDataset loader.

Official GRES layout after extract_gres_images.sh:
  lisat_data/gres_images/{train,val,test}/*.jpg
  lisat_data/gres_annotations/{train,val,test}/*.json

Current loader expects:
  dataset/reason_seg/GeoReasonSeg/{train,val,test}/*.jpg
  dataset/reason_seg/GeoReasonSeg/{train,val,test}/*.json

This script creates symlinks by default to avoid duplicating image data.
"""

import argparse
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare GRES for LISAt loader")
    parser.add_argument("--gres-root", default="./dataset/lisat_data")
    parser.add_argument("--output-root", default="./dataset/reason_seg/GeoReasonSeg")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy: bool, overwrite: bool):
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main():
    args = parse_args()
    gres_root = Path(args.gres_root).resolve()
    output_root = Path(args.output_root).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    total_json = 0
    total_img = 0
    missing_images = []
    for split in splits:
        ann_dir = gres_root / "gres_annotations" / split
        img_dir = gres_root / "gres_images" / split
        out_dir = output_root / split
        if not ann_dir.exists():
            raise FileNotFoundError(f"Missing annotation dir: {ann_dir}")
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing image dir: {img_dir}")

        json_files = sorted(ann_dir.glob("*.json"))
        for json_path in json_files:
            image_path = img_dir / json_path.with_suffix(".jpg").name
            if not image_path.exists():
                missing_images.append(str(image_path))
                continue
            link_or_copy(json_path, out_dir / json_path.name, args.copy, args.overwrite)
            link_or_copy(image_path, out_dir / image_path.name, args.copy, args.overwrite)
            total_json += 1
            total_img += 1

        print(f"[{split}] prepared {len(json_files) - len(missing_images)} pairs in {out_dir}")

    if missing_images:
        print("Missing images for annotations, first 20:")
        for path in missing_images[:20]:
            print(path)
        raise SystemExit(f"Missing {len(missing_images)} images; extraction may be incomplete")

    print(f"Done. Prepared {total_json} json files and {total_img} images under {output_root}")


if __name__ == "__main__":
    main()
