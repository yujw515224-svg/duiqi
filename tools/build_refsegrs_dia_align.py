#!/usr/bin/env python3
"""Build DIA-style image-text alignment samples from RefSegRS.

The script keeps the original RefSegRS files unchanged and writes a derived
JSONL dataset plus optional focus/removed image views.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


CONFUSION_GROUPS = [
    ["road", "paved road", "unpaved road", "sidewalk", "runway"],
    ["vehicle", "car", "van", "light-duty vehicle", "truck"],
    ["building", "residential building", "industrial building", "parking area"],
    ["parking area", "impervious surface", "bare land", "building"],
    ["tree", "grass", "farmland", "vegetation"],
    ["water", "river", "pond", "canal"],
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build RefSegRS_DIAAlign samples")
    parser.add_argument("--dataset-dir", default="./dataset")
    parser.add_argument("--output-name", default="RefSegRS_DIAAlign")
    parser.add_argument("--splits", default="train", help="Comma-separated splits, e.g. train,test")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples per split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blur-kernel", type=int, default=31)
    parser.add_argument("--edge-sigma", type=float, default=2.0)
    parser.add_argument("--removed-mode", choices=["blur", "inpaint"], default="blur")
    parser.add_argument("--hard-negatives-per-sample", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def ensure_odd(value):
    value = int(value)
    if value < 3:
        return 3
    return value if value % 2 == 1 else value + 1


def load_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def phrase_lines(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            rows.append((parts[0], parts[1].strip().lower(), line_no))
    return rows


def position_name(cx, cy, width, height):
    x_names = ["left", "central", "right"]
    y_names = ["upper", "middle", "lower"]
    x_idx = min(2, int(3 * cx / max(width, 1)))
    y_idx = min(2, int(3 * cy / max(height, 1)))
    if x_idx == 1 and y_idx == 1:
        return "central"
    return f"{y_names[y_idx]}-{x_names[x_idx]}"


def color_name(rgb):
    r, g, b = [float(x) for x in rgb]
    brightness = (r + g + b) / 3.0
    spread = max(r, g, b) - min(r, g, b)
    if brightness < 55:
        base = "dark"
    elif brightness > 190:
        base = "bright"
    else:
        base = "medium-toned"

    if spread < 18:
        hue = "gray"
    elif g > r + 12 and g > b + 12:
        hue = "green"
    elif b > r + 12 and b > g + 12:
        hue = "blue"
    elif r > g + 12 and r > b + 12:
        hue = "reddish"
    elif r > b + 8 and g > b + 8:
        hue = "yellow-brown"
    else:
        hue = "mixed-color"
    return f"{base} {hue}"


def mask_stats(image_bgr, mask):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    area = float(mask.mean())
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    cx, cy = float(xs.mean()), float(ys.mean())
    bbox_area = max(float(bw * bh), 1.0)
    extent = float(mask.sum()) / bbox_area
    elongation = max(bw, bh) / max(min(bw, bh), 1)

    if area < 0.001:
        size = "tiny"
    elif area < 0.006:
        size = "very small"
    elif area < 0.02:
        size = "small"
    elif area < 0.08:
        size = "medium-sized"
    else:
        size = "large"

    if elongation > 4.0:
        shape = "long and narrow"
    elif elongation > 2.0:
        shape = "elongated"
    elif extent < 0.35:
        shape = "irregular"
    elif extent > 0.65:
        shape = "compact"
    else:
        shape = "moderately compact"

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    target_rgb = image_rgb[mask].mean(axis=0)

    kernel = np.ones((21, 21), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = np.logical_and(dilated, ~mask)
    if ring.any():
        context_rgb = image_rgb[ring].mean(axis=0)
        context = color_name(context_rgb)
    else:
        context = "nearby"

    return {
        "area_ratio": area,
        "area_percent": area * 100.0,
        "bbox": [x1, y1, x2, y2],
        "position": position_name(cx, cy, width, height),
        "size": size,
        "shape": shape,
        "appearance": color_name(target_rgb),
        "context": context,
    }


def detailed_description(phrase, stats):
    return (
        f"The target is a {stats['size']} {stats['shape']} {phrase} "
        f"located in the {stats['position']} part of the remote sensing image. "
        f"It covers about {stats['area_percent']:.2f}% of the image and shows a "
        f"{stats['appearance']} visual appearance, with {stats['context']} surroundings."
    )


def soft_alpha(mask, sigma):
    alpha = mask.astype(np.float32)
    if sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha[..., None]


def make_focus_view(image_bgr, mask, blur_kernel, edge_sigma):
    blur_kernel = ensure_odd(blur_kernel)
    blurred = cv2.GaussianBlur(image_bgr, (blur_kernel, blur_kernel), 0)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    background = cv2.addWeighted(blurred, 0.65, gray, 0.35, 0)
    alpha = soft_alpha(mask, edge_sigma)
    focus = image_bgr.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
    return np.clip(focus, 0, 255).astype(np.uint8)


def make_removed_view(image_bgr, mask, blur_kernel, mode):
    kernel = np.ones((9, 9), np.uint8)
    remove_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    if mode == "inpaint":
        return cv2.inpaint(image_bgr, (remove_mask * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    blur_kernel = ensure_odd(blur_kernel)
    blurred = cv2.GaussianBlur(image_bgr, (blur_kernel, blur_kernel), 0)
    alpha = remove_mask.astype(np.float32)[..., None]
    removed = image_bgr.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(removed, 0, 255).astype(np.uint8)


def choose_hard_negative(phrase, phrase_pool, rng):
    phrase_words = set(phrase.split())
    candidates = []
    for group in CONFUSION_GROUPS:
        if any(term in phrase for term in group):
            candidates.extend([term for term in group if term != phrase and term not in phrase])
    if not candidates:
        candidates = [p for p in phrase_pool if p != phrase and not (set(p.split()) & phrase_words)]
    if not candidates:
        return None
    return rng.choice(sorted(set(candidates)))


def write_image(path, image_bgr, quality):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    else:
        cv2.imwrite(str(path), image_bgr)


def rel(path, dataset_dir):
    return str(path.relative_to(dataset_dir)).replace("\\", "/")


def add_record(records, dataset_dir, image_path, mask_path, image_id, line_no, phrase, desc, stats, sample_type, is_positive):
    records.append({
        "source": "RefSegRS",
        "image_id": image_id,
        "line_no": line_no,
        "phrase": phrase,
        "detailed_text": desc,
        "sample_type": sample_type,
        "is_positive": bool(is_positive),
        "image_path": rel(image_path, dataset_dir),
        "mask_path": rel(mask_path, dataset_dir),
        "area_ratio": stats["area_ratio"],
        "area_percent": stats["area_percent"],
        "bbox": stats["bbox"],
        "position": stats["position"],
        "shape": stats["shape"],
        "appearance": stats["appearance"],
        "context": stats["context"],
    })


def build_split(args, split, rng):
    dataset_dir = Path(args.dataset_dir).resolve()
    src_root = dataset_dir / "RefSegRS"
    out_root = dataset_dir / args.output_name
    phrase_path = src_root / f"output_phrase_{split}.txt"
    rows = phrase_lines(phrase_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    phrase_pool = [phrase for _, phrase, _ in rows]

    records = []
    skipped = 0
    for image_id, phrase, line_no in rows:
        image_path = src_root / "images" / f"{image_id}.tif"
        mask_path = src_root / "masks" / f"{image_id}.tif"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or not mask_path.exists():
            skipped += 1
            continue
        mask = load_mask(mask_path)
        stats = mask_stats(image, mask)
        if stats is None:
            skipped += 1
            continue

        desc = detailed_description(phrase, stats)
        stem = f"{image_id}_{line_no:06d}"
        focus_path = out_root / "views" / split / "focus" / f"{stem}_focus.png"
        removed_path = out_root / "views" / split / "removed" / f"{stem}_removed.png"

        focus = make_focus_view(image, mask, args.blur_kernel, args.edge_sigma)
        removed = make_removed_view(image, mask, args.blur_kernel, args.removed_mode)
        write_image(focus_path, focus, args.jpeg_quality)
        write_image(removed_path, removed, args.jpeg_quality)

        add_record(records, dataset_dir, image_path, mask_path, image_id, line_no, phrase, desc, stats, "original_positive", True)
        add_record(records, dataset_dir, focus_path, mask_path, image_id, line_no, phrase, desc, stats, "focus_positive", True)
        add_record(records, dataset_dir, removed_path, mask_path, image_id, line_no, phrase, desc, stats, "removed_negative", False)

        for neg_idx in range(args.hard_negatives_per_sample):
            neg_phrase = choose_hard_negative(phrase, phrase_pool, rng)
            if not neg_phrase:
                continue
            neg_desc = detailed_description(neg_phrase, stats)
            add_record(
                records,
                dataset_dir,
                image_path,
                mask_path,
                image_id,
                line_no,
                neg_phrase,
                neg_desc,
                stats,
                f"hard_text_negative_{neg_idx}",
                False,
            )

    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / f"alignment_{split}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"[{split}] wrote {len(records)} records to {jsonl_path}; skipped {skipped} source rows")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
        build_split(args, split, rng)


if __name__ == "__main__":
    main()
