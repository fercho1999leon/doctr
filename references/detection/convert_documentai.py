# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Convert Google Document AI exports (labelled entities + embedded page image) into the docTR
multi-class detection format expected by `references/detection/train.py`.

Each input JSON must contain `pages[0].image.content` (base64 image) and `entities[]`, where every
entity has a `type` (class), a `mentionText` and a `pageAnchor.pageRefs[0].boundingPoly.normalizedVertices`.

Output layout (one folder per split):

    <output>/<split>/images/<id>.<ext>          # split = train, val and (with --test-ratio) test
    <output>/<split>/labels.json    # docTR multi-class labels, every class present in every entry
    <output>/<split>/manifest.json  # provenance + annotated text per box (used by evaluate_fields.py)
    <output>/audit.json             # dataset audit (class counts, multi-line texts, tiny/overlapping boxes, ...)

Documents whose page images are near-duplicates (perceptual hash within --dup-threshold) are grouped and always land in
the same split. The split is stratified on the presence of each class, so rare fields are represented
in both train and val.
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import io
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CONVERTER_VERSION = "1.0"
SUPPORTED_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tiff", "BMP": ".bmp"}


def parse_mapping(items: list[str]) -> dict[str, str]:
    """Parse `old=new` CLI arguments into a dict."""
    mapping = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected old=new, got '{item}'")
        old, new = item.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def perceptual_hash(img: Image.Image, size: int = 16) -> str:
    """Difference hash: grayscale, downscale to (size+1) x size, compare horizontally adjacent pixels.
    Near-duplicate pages (rescaled, recompressed) have a small Hamming distance; different receipts from
    the same bank template differ in dozens of bits."""
    small = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.float32)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def decode_page_image(page: dict) -> tuple[Image.Image, str]:
    """Decode the embedded page image. The format is detected from the bytes (many exports lack mimeType)."""
    raw = base64.b64decode(page["image"]["content"])
    img = Image.open(io.BytesIO(raw))
    img.load()
    fmt = img.format or "PNG"
    if fmt not in SUPPORTED_EXTENSIONS:
        fmt = "PNG"
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img, SUPPORTED_EXTENSIONS[fmt]


def entity_polygon(entity: dict, width: int, height: int) -> np.ndarray | None:
    """Return the entity polygon as absolute pixel coordinates with shape (4, 2), or None if unusable."""
    refs = entity.get("pageAnchor", {}).get("pageRefs", [])
    if not refs:
        return None
    poly = refs[0].get("boundingPoly", {})
    vertices = poly.get("normalizedVertices")
    if vertices:
        pts = [(v.get("x", 0.0) * width, v.get("y", 0.0) * height) for v in vertices]
    else:
        vertices = poly.get("vertices")
        if not vertices:
            return None
        pts = [(v.get("x", 0.0), v.get("y", 0.0)) for v in vertices]
    if len(pts) != 4:
        # Fall back to the axis-aligned box around whatever polygon we were given
        xs, ys = zip(*pts)
        pts = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
    arr = np.asarray(pts, dtype=np.float64)
    arr[:, 0] = arr[:, 0].clip(0, width)
    arr[:, 1] = arr[:, 1].clip(0, height)
    return arr


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0 = a.min(axis=0)
    ax1, ay1 = a.max(axis=0)
    bx0, by0 = b.min(axis=0)
    bx1, by1 = b.max(axis=0)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def load_documents(inputs: list[Path], rename: dict[str, str], drop: set[str], log) -> list[dict]:
    """Read every Document AI JSON under the input folders and normalise it into an internal record."""
    files = sorted(p for root in inputs for p in root.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"no JSON files found under {[str(p) for p in inputs]}")
    docs = []
    seen_ids: dict[str, Path] = {}
    for path in files:
        with open(path, "rb") as f:
            data = json.load(f)
        pages = data.get("pages", [])
        if not pages:
            log(f"[skip] {path}: no pages")
            continue
        if len(pages) > 1:
            log(f"[warn] {path}: {len(pages)} pages, only the first one is used")
        page = pages[0]
        if not page.get("image", {}).get("content"):
            log(f"[skip] {path}: no embedded image")
            continue
        img, ext = decode_page_image(page)
        width, height = img.size
        declared = (page["image"].get("width"), page["image"].get("height"))
        if declared != (None, None) and declared != (width, height):
            log(f"[warn] {path}: declared size {declared} != decoded {img.size}, using decoded size")

        # The parent folder name is the Document AI document id and is unique in the export
        doc_id = path.parent.name if path.parent not in inputs else path.stem
        if doc_id in seen_ids:
            log(f"[warn] duplicate document id '{doc_id}' ({path} and {seen_ids[doc_id]}), suffixing")
            doc_id = f"{doc_id}_{hashlib.sha1(str(path).encode()).hexdigest()[:6]}"
        seen_ids[doc_id] = path

        boxes = []
        for ent in data.get("entities", []):
            cls = rename.get(ent["type"], ent["type"])
            if cls in drop:
                continue
            poly = entity_polygon(ent, width, height)
            if poly is None:
                log(f"[warn] {path}: entity '{ent['type']}' has no usable geometry, skipped")
                continue
            boxes.append({
                "class": cls,
                "polygon": poly,
                "text": ent.get("mentionText", ""),
                "confidence": ent.get("confidence"),
                "entity_id": ent.get("id"),
            })
        docs.append({
            "id": doc_id,
            "source": str(path),
            "source_split": next((r.name for r in inputs if r in path.parents), ""),
            "image": img,
            "ext": ext,
            "width": width,
            "height": height,
            "phash": perceptual_hash(img),
            "pixel_hash": hashlib.sha256(img.tobytes()).hexdigest(),
            "boxes": boxes,
        })
    return docs


def group_documents(docs: list[dict], threshold: int, log) -> list[list[dict]]:
    """Group near-duplicate pages (perceptual hash Hamming distance <= threshold, transitively).
    Groups are never split across train/val. Exact byte-identical pages are reported as well."""
    parent = list(range(len(docs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if hamming(docs[i]["phash"], docs[j]["phash"]) <= threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = collections.OrderedDict()
    for i, doc in enumerate(docs):
        groups.setdefault(find(i), []).append(doc)
    for members in groups.values():
        if len(members) > 1:
            group_id = members[0]["phash"]
            for m in members:
                m["group"] = group_id
            exact = len({m["pixel_hash"] for m in members}) < len(members)
            log(
                f"[info] near-duplicate group ({len(members)} docs{', exact duplicates inside' if exact else ''}): "
                f"{[m['id'] for m in members]}"
            )
        else:
            members[0]["group"] = members[0]["phash"]
    return list(groups.values())


def stratified_group_split(
    groups: list[list[dict]], class_names: list[str], val_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Greedy stratified split at the group level.

    Groups are shuffled with `seed`, then moved to val one at a time, each time picking the group that keeps
    the per-class presence ratio of val closest to the global ratio, until val holds `val_ratio` of the docs.
    """
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)
    n_docs = sum(len(g) for g in groups)
    n_val_target = max(1, math.ceil(val_ratio * n_docs)) if val_ratio > 0 else 0

    def presence(doc: dict) -> np.ndarray:
        present = {b["class"] for b in doc["boxes"]}
        return np.array([c in present for c in class_names], dtype=np.float64)

    global_ratio = sum((presence(d) for g in groups for d in g), np.zeros(len(class_names))) / max(n_docs, 1)
    val: list[dict] = []
    val_presence = np.zeros(len(class_names))
    remaining = groups
    while remaining and len(val) < n_val_target:
        best_idx, best_score = 0, None
        for idx, g in enumerate(remaining):
            if len(val) + len(g) > n_val_target and val:  # do not overshoot once we have something
                continue
            cand_presence = val_presence + sum((presence(d) for d in g), np.zeros(len(class_names)))
            cand_ratio = cand_presence / (len(val) + len(g))
            score = float(np.abs(cand_ratio - global_ratio).sum())
            if best_score is None or score < best_score:
                best_idx, best_score = idx, score
        if best_score is None:
            break
        chosen = remaining.pop(best_idx)
        val.extend(chosen)
        val_presence += sum((presence(d) for d in chosen), np.zeros(len(class_names)))
    train = [d for g in remaining for d in g]
    return train, val


def kfold_group_split(groups: list[list[dict]], folds: int, fold: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Grouped K-fold: fold `fold` is val, the others are train."""
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)
    buckets: list[list[dict]] = [[] for _ in range(folds)]
    for g in sorted(groups, key=len, reverse=True):  # largest groups first, into the smallest bucket
        min(buckets, key=len).extend(g)
    val = buckets[fold]
    train = [d for i, b in enumerate(buckets) if i != fold for d in b]
    return train, val


def write_split(split_name: str, docs: list[dict], class_names: list[str], out_dir: Path, meta: dict) -> dict:
    split_dir = out_dir / split_name
    img_dir = split_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels: dict[str, dict] = {}
    manifest_docs = []
    for doc in docs:
        img_name = f"{doc['id']}{doc['ext']}"
        img_path = img_dir / img_name
        save_kwargs = {"quality": 95} if doc["ext"] == ".jpg" else {}
        doc["image"].save(img_path, **save_kwargs)
        with open(img_path, "rb") as f:
            img_hash = hashlib.sha256(f.read()).hexdigest()
        polygons: dict[str, list] = {c: [] for c in class_names}
        for box in doc["boxes"]:
            polygons[box["class"]].append(np.round(box["polygon"], 2).tolist())
        labels[img_name] = {
            "img_dimensions": [doc["height"], doc["width"]],
            "img_hash": img_hash,
            "polygons": polygons,
        }
        manifest_docs.append({
            "image": img_name,
            "id": doc["id"],
            "source": doc["source"],
            "source_split": doc["source_split"],
            "group": doc["group"],
            "phash": doc["phash"],
            "pixel_hash": doc["pixel_hash"],
            "img_hash": img_hash,
            "width": doc["width"],
            "height": doc["height"],
            "boxes": [
                {
                    "class": b["class"],
                    "polygon": np.round(b["polygon"], 2).tolist(),
                    "text": b["text"],
                    "confidence": b["confidence"],
                    "entity_id": b["entity_id"],
                }
                for b in doc["boxes"]
            ],
        })
    with open(split_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
    manifest = {**meta, "split": split_name, "class_names": class_names, "documents": manifest_docs}
    with open(split_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return manifest


def audit(splits: dict[str, list[dict]], class_names: list[str], max_text_len: int, log) -> dict:
    """Compute and print dataset statistics that matter for detection + recognition of the fields."""
    report: dict = {"class_names": class_names, "splits": {}}
    for split_name, docs in splits.items():
        boxes_per_class = collections.Counter()
        docs_with_class = collections.Counter()
        multiline = collections.Counter()
        too_long = collections.Counter()
        tiny = []
        overlaps = []
        empty_docs = []
        for doc in docs:
            present = set()
            area_img = doc["width"] * doc["height"]
            for i, b in enumerate(doc["boxes"]):
                boxes_per_class[b["class"]] += 1
                present.add(b["class"])
                if "\n" in b["text"].strip():
                    multiline[b["class"]] += 1
                if len(b["text"].replace("\n", " ")) > max_text_len:
                    too_long[b["class"]] += 1
                w = b["polygon"][:, 0].max() - b["polygon"][:, 0].min()
                h = b["polygon"][:, 1].max() - b["polygon"][:, 1].min()
                if w * h < 0.001 * area_img or min(w, h) < 4:
                    tiny.append({"doc": doc["id"], "class": b["class"], "w": round(w, 1), "h": round(h, 1)})
                for other in doc["boxes"][i + 1 :]:
                    if other["class"] != b["class"]:
                        iou = box_iou(b["polygon"], other["polygon"])
                        if iou > 0.5:
                            overlaps.append({
                                "doc": doc["id"],
                                "classes": [b["class"], other["class"]],
                                "iou": round(iou, 3),
                            })
            for c in present:
                docs_with_class[c] += 1
            if not doc["boxes"]:
                empty_docs.append(doc["id"])
        n = len(docs)
        report["splits"][split_name] = {
            "n_docs": n,
            "boxes_per_class": {c: boxes_per_class[c] for c in class_names},
            "docs_with_class": {c: docs_with_class[c] for c in class_names},
            "presence_ratio": {c: round(docs_with_class[c] / n, 3) if n else 0 for c in class_names},
            "multiline_texts": {c: multiline[c] for c in class_names},
            f"texts_longer_than_{max_text_len}": {c: too_long[c] for c in class_names},
            "tiny_boxes": tiny,
            "cross_class_overlaps_iou_gt_0.5": overlaps,
            "docs_without_boxes": empty_docs,
        }
        log(f"\n=== {split_name}: {n} documents ===")
        log(f"{'class':24s} {'boxes':>6s} {'docs':>6s} {'ratio':>6s} {'multi':>6s} {'>' + str(max_text_len):>6s}")
        for c in class_names:
            log(
                f"{c:24s} {boxes_per_class[c]:6d} {docs_with_class[c]:6d} "
                f"{(docs_with_class[c] / n if n else 0):6.2f} {multiline[c]:6d} {too_long[c]:6d}"
            )
        if tiny:
            log(f"[audit] {len(tiny)} tiny boxes (<0.1% of image or <4px side)")
        if overlaps:
            log(f"[audit] {len(overlaps)} cross-class overlaps with IoU>0.5 (possible label+value in one box)")
        if empty_docs:
            log(f"[audit] {len(empty_docs)} documents without any box: {empty_docs}")
    return report


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main(args):
    log = print
    inputs = [Path(p).resolve() for p in args.input]
    for p in inputs:
        if not p.is_dir():
            raise FileNotFoundError(f"input folder not found: {p}")
    rename = parse_mapping(args.rename)
    drop = set(args.drop)

    docs = load_documents(inputs, rename, drop, log)
    class_counter = collections.Counter(b["class"] for d in docs for b in d["boxes"])
    class_names = sorted(class_counter)
    log(f"\n{len(docs)} documents, {sum(class_counter.values())} boxes, classes: {class_names}")
    if args.min_boxes_per_class > 0:
        rare = [c for c in class_names if class_counter[c] < args.min_boxes_per_class]
        if rare:
            log(f"[warn] classes with fewer than {args.min_boxes_per_class} boxes: {rare} (consider --drop)")

    groups = group_documents(docs, args.dup_threshold, log)
    test: list[dict] = []
    if args.folds:
        train, val = kfold_group_split(groups, args.folds, args.fold, args.seed)
        split_desc = f"grouped {args.folds}-fold, fold {args.fold}"
    else:
        if args.test_ratio > 0:
            # Carve the held-out test set first (group-aware, so no near-duplicate of a test page is ever
            # trained on), then split the remaining groups into train/val
            remaining, test = stratified_group_split(groups, class_names, args.test_ratio, args.seed)
            test_ids = {d["id"] for d in test}
            groups = [g for g in groups if g[0]["id"] not in test_ids]
        train, val = stratified_group_split(groups, class_names, args.val_ratio, args.seed + 1)
        split_desc = f"grouped stratified split, val_ratio={args.val_ratio}, test_ratio={args.test_ratio}"
    log(f"\nSplit ({split_desc}, seed={args.seed}): train={len(train)} val={len(val)} test={len(test)}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "converter_version": CONVERTER_VERSION,
        "git_revision": git_revision(),
        "command": " ".join(sys.argv),
        "inputs": [str(p) for p in inputs],
        "rename": rename,
        "drop": sorted(drop),
        "seed": args.seed,
        "split_strategy": split_desc,
    }
    splits = {"train": train, args.val_name: val}
    if test:
        splits["test"] = test
    for name, split_docs in splits.items():
        write_split(name, split_docs, class_names, out_dir, meta)
        log(f"wrote {name}: {len(split_docs)} images -> {out_dir / name}")

    report = audit(splits, class_names, args.max_text_len, log)
    report["meta"] = meta
    with open(out_dir / "audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log(f"\naudit written to {out_dir / 'audit.json'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Google Document AI exports to the docTR multi-class detection format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", nargs="+", required=True, help="folder(s) containing the Document AI JSON files")
    parser.add_argument("--output", required=True, help="output folder (train/ and val/ are created inside)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="fraction of documents used for validation")
    parser.add_argument("--val-name", default="val", help="name of the validation split folder")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="fraction of documents held out in a test/ split (carved before train/val, group-aware)",
    )
    parser.add_argument("--folds", type=int, default=0, help="if >0, use grouped K-fold instead of a single split")
    parser.add_argument("--fold", type=int, default=0, help="index of the fold used for validation (with --folds)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split")
    parser.add_argument(
        "--dup-threshold",
        type=int,
        default=8,
        help="max Hamming distance (out of 256 bits) between page hashes to treat two pages as near-duplicates",
    )
    parser.add_argument("--rename", nargs="*", default=[], help="class renames, e.g. facha=fecha")
    parser.add_argument("--drop", nargs="*", default=[], help="classes to discard, e.g. numero_cuenta")
    parser.add_argument(
        "--min-boxes-per-class", type=int, default=10, help="warn about classes with fewer boxes than this"
    )
    parser.add_argument(
        "--max-text-len", type=int, default=32, help="audit texts longer than this (PARSeq max_length is 32)"
    )
    args = parser.parse_args()
    if args.folds and not 0 <= args.fold < args.folds:
        parser.error("--fold must be in [0, --folds)")
    return args


if __name__ == "__main__":
    main(parse_args())
