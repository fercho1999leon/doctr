# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Audit the annotated field boxes before training a detector.

1. Draws the ground-truth polygons (coloured per class) on a few pages -> `<data>/audit_viz/`
2. Crops every ground-truth box and runs the pretrained recognition model(s) on the crop, then reports the
   normalised exact-match rate per class against the annotated `mentionText`.

This separates recognition errors from detection errors: a field whose crop cannot be read by the
recogniser (multi-line block, > 32 characters, missing vocabulary) will not be extracted correctly even with
a perfect detector.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from field_utils import load_manifest, normalize_text, polygon_to_box, resolve_device
from PIL import Image, ImageDraw

from doctr.models import recognition_predictor

PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (128, 128, 0),
]


def draw_pages(manifest: dict, data_dir: Path, out_dir: Path, max_pages: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(manifest["class_names"])}
    for doc in manifest["documents"][:max_pages]:
        img = Image.open(data_dir / "images" / doc["image"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for box in doc["boxes"]:
            pts = [tuple(p) for p in box["polygon"]]
            draw.polygon(pts, outline=colors[box["class"]], width=3)
            x, y = pts[0]
            draw.text((x + 2, max(0, y - 12)), box["class"], fill=colors[box["class"]])
        img.save(out_dir / doc["image"])
    print(f"{min(max_pages, len(manifest['documents']))} annotated pages written to {out_dir}")


def crop(img: np.ndarray, polygon) -> np.ndarray:
    x0, y0, x1, y1 = polygon_to_box(polygon)
    h, w = img.shape[:2]
    x0, y0 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0)))
    x1, y1 = int(min(w, np.ceil(x1))), int(min(h, np.ceil(y1)))
    return img[y0 : max(y1, y0 + 1), x0 : max(x1, x0 + 1)]


@torch.no_grad()
def audit_recognition(manifest: dict, data_dir: Path, archs: list[str], device: torch.device) -> dict:
    crops, refs = [], []
    for doc in manifest["documents"]:
        img = np.asarray(Image.open(data_dir / "images" / doc["image"]).convert("RGB"))
        for box in doc["boxes"]:
            crops.append(crop(img, box["polygon"]))
            refs.append((doc["id"], box["class"], box["text"]))

    report: dict = {"n_boxes": len(crops), "class_names": manifest["class_names"], "models": {}}
    for arch in archs:
        predictor = recognition_predictor(arch, pretrained=True).to(device)
        preds = predictor(crops)
        per_class = collections.defaultdict(
            lambda: {"n": 0, "exact": 0, "exact_nospace": 0, "exact_single_line": 0, "n_single_line": 0}
        )
        examples = collections.defaultdict(list)
        for (doc_id, cls_name, gt), (pred, conf) in zip(refs, preds):
            stats = per_class[cls_name]
            stats["n"] += 1
            match = normalize_text(pred, cls_name) == normalize_text(gt, cls_name)
            stats["exact"] += int(match)
            # word-level recognisers drop the spaces of a multi-word crop: measure that separately
            stats["exact_nospace"] += int(
                normalize_text(pred, cls_name).replace(" ", "") == normalize_text(gt, cls_name).replace(" ", "")
            )
            if "\n" not in gt.strip():
                stats["n_single_line"] += 1
                stats["exact_single_line"] += int(match)
            if not match and len(examples[cls_name]) < 5:
                examples[cls_name].append({"doc": doc_id, "gt": gt, "pred": pred, "confidence": round(conf, 3)})
        summary = {
            c: {
                **s,
                "exact_match": round(s["exact"] / s["n"], 3) if s["n"] else None,
                "exact_match_nospace": round(s["exact_nospace"] / s["n"], 3) if s["n"] else None,
                "exact_match_single_line": round(s["exact_single_line"] / s["n_single_line"], 3)
                if s["n_single_line"]
                else None,
            }
            for c, s in per_class.items()
        }
        total = sum(s["n"] for s in per_class.values())
        exact = sum(s["exact"] for s in per_class.values())
        report["models"][arch] = {
            "exact_match_overall": round(exact / total, 3) if total else None,
            "per_class": summary,
            "mismatch_examples": examples,
        }
        print(f"\n=== recognition baseline on ground-truth crops: {arch} ===")
        print(f"{'class':24s} {'n':>5s} {'exact':>7s} {'nospc':>7s} {'1-line':>7s}")
        for c in manifest["class_names"]:
            s = summary.get(c)
            if s is None:
                continue
            em = f"{s['exact_match']:.2%}" if s["exact_match"] is not None else "-"
            emn = f"{s['exact_match_nospace']:.2%}"
            em1 = f"{s['exact_match_single_line']:.2%}" if s["exact_match_single_line"] is not None else "-"
            print(f"{c:24s} {s['n']:5d} {em:>7s} {emn:>7s} {em1:>7s}")
        print(f"{'overall':24s} {total:5d} {exact / total if total else 0:7.2%}")
    return report


def main(args):
    data_dir = Path(args.data)
    manifest = load_manifest(data_dir)
    draw_pages(manifest, data_dir, data_dir / "audit_viz", args.viz_pages)
    device = resolve_device(args.device)
    report = audit_recognition(manifest, data_dir, args.reco_archs, device)
    out = data_dir / "audit_recognition.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\nrecognition audit written to {out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise ground-truth field boxes and measure the pretrained recogniser on their crops",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True, help="split folder produced by convert_documentai.py (has manifest.json)"
    )
    parser.add_argument(
        "--reco-archs", nargs="+", default=["parseq", "crnn_vgg16_bn"], help="pretrained recognition models to test"
    )
    parser.add_argument("--viz-pages", type=int, default=10, help="number of pages to draw in audit_viz/")
    parser.add_argument("--device", default=None, help="cpu, mps, cuda, cuda:N or a CUDA index (default: auto)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
