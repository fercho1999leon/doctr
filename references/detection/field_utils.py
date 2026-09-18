# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Shared helpers for the field (KIE) detection scripts: manifest loading, text normalisation,
geometry helpers, word-to-region assignment and checkpoint metadata."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

from doctr.models import detection

__all__ = [
    "box_iou",
    "load_manifest",
    "normalize_text",
    "polygon_to_box",
    "read_fields_from_words",
    "load_detection_checkpoint",
    "resolve_device",
    "FieldExtractor",
]


def resolve_device(device: str | int | None) -> torch.device:
    """Turn a CLI device spec (None, "cpu", "mps", "cuda", "cuda:1", 0) into a torch device.

    With `None`, pick CUDA if available, then MPS, then CPU.
    """
    if device is None or device == "":
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        index = int(device)
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if index >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
        return torch.device("cuda", index)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise AssertionError("MPS backend is not available on this machine.")
    return dev


def load_manifest(split_dir: str | Path) -> dict:
    path = Path(split_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run convert_documentai.py first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_MONEY_RE = re.compile(r"[$,\s]")


def normalize_text(text: str, field: str | None = None) -> str:
    """Normalisation used for exact-match scoring.

    - unicode NFKC, upper-case
    - line breaks and repeated whitespace collapsed to a single space
    - for `valor_transferido`: currency symbols, thousands separators and spaces removed
    """
    text = unicodedata.normalize("NFKC", text).upper()
    text = re.sub(r"\s+", " ", text).strip()
    if field == "valor_transferido":
        text = _MONEY_RE.sub("", text)
    return text


def polygon_to_box(poly) -> np.ndarray:
    """(N, 2) polygon or (4,) box -> [xmin, ymin, xmax, ymax]."""
    arr = np.asarray(poly, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:4]
    return np.concatenate([arr.min(axis=0), arr.max(axis=0)])


def box_iou(a, b) -> float:
    a, b = polygon_to_box(a), polygon_to_box(b)
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def read_fields_from_words(
    regions: dict[str, list[np.ndarray]],
    words: list[dict],
    margin: float = 0.01,
) -> dict[str, list[dict]]:
    """Assign OCR words to detected field regions and concatenate them in reading order.

    Args:
        regions: class name -> list of relative boxes/polygons ([xmin, ymin, xmax, ymax] or (4, 2)), optionally with
            a trailing score (the 5th value of a straight box, or a (5, 2) polygon whose last row holds the score)
        words: list of {"value": str, "confidence": float, "box": [xmin, ymin, xmax, ymax]} in relative coordinates
        margin: region boxes are expanded by this relative margin before testing word centres

    Returns:
        class name -> list of {"value", "confidence", "geometry", "words"} sorted by position
    """
    out: dict[str, list[dict]] = {}
    for cls_name, geoms in regions.items():
        fields = []
        for geom in geoms:
            geom = np.asarray(geom, dtype=np.float64)
            score = None
            if geom.ndim == 1 and geom.shape[0] >= 5:
                score, geom = float(geom[4]), geom[:4]
            elif geom.ndim == 2 and geom.shape[0] == 5:
                score, geom = float(geom[4, 0]), geom[:4]
            box = polygon_to_box(geom)
            x0, y0, x1, y1 = box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin
            inside = []
            for w in words:
                cx = (w["box"][0] + w["box"][2]) / 2
                cy = (w["box"][1] + w["box"][3]) / 2
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    inside.append(w)
            # Reading order: group into lines by vertical overlap, then left to right
            lines: list[list[dict]] = []
            for w in sorted(inside, key=lambda w: (w["box"][1] + w["box"][3]) / 2):
                h = w["box"][3] - w["box"][1]
                cy = (w["box"][1] + w["box"][3]) / 2
                for line in lines:
                    ly = np.mean([(lw["box"][1] + lw["box"][3]) / 2 for lw in line])
                    if abs(ly - cy) < 0.6 * max(h, 1e-6):
                        line.append(w)
                        break
                else:
                    lines.append([w])
            ordered = [w for line in lines for w in sorted(line, key=lambda w: w["box"][0])]
            value = " ".join(w["value"] for w in ordered)
            text_conf = float(np.mean([w["confidence"] for w in ordered])) if ordered else 0.0
            fields.append({
                "value": value,
                "confidence": text_conf,
                "detection_score": score,
                "geometry": np.round(box, 4).tolist(),
                "words": [w["value"] for w in ordered],
            })
        fields.sort(key=lambda f: (f["geometry"][1], f["geometry"][0]))
        out[cls_name] = fields
    return out


def load_detection_checkpoint(checkpoint: str | Path, device: torch.device, **overrides):
    """Load a detection model trained with `train.py` from `<name>.pt` + its `<name>.json` sidecar.

    The sidecar (written by train.py) holds the architecture, the ordered class names and the target-building
    options, so that inference never depends on a hand-written class list.
    """
    ckpt = Path(checkpoint)
    sidecar = ckpt.with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"{sidecar} not found. Checkpoints must be trained with references/detection/train.py, which writes "
            "the class names next to the weights."
        )
    with open(sidecar, encoding="utf-8") as f:
        cfg = json.load(f)
    kwargs = {
        "pretrained": False,
        "class_names": cfg["class_names"],
        "assume_straight_pages": cfg.get("assume_straight_pages", True),
        "mask_empty_classes": cfg.get("mask_empty_classes", True),
    }
    kwargs.update(overrides)
    model = detection.__dict__[cfg["arch"]](**kwargs)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if list(model.class_names) != list(cfg["class_names"]):
        raise ValueError(f"class order mismatch: model {model.class_names} vs checkpoint {cfg['class_names']}")
    return model.to(device).eval(), cfg


class FieldExtractor:
    """Detect semantic field regions with a fine-tuned multi-class detector and read them with the standard
    word-level OCR pipeline.

    Two reading modes:
      - "words" (default): run `ocr_predictor` on the page, assign every word whose centre falls inside a detected
        region to that region and concatenate them in reading order. Handles multi-line and long fields.
      - "kie": crop each detected region and feed it to the recogniser as a single item (`kie_predictor`
        behaviour). Only suitable for short single-line fields.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: torch.device,
        reco_arch: str = "parseq",
        det_arch_words: str = "fast_base",
        mode: str = "words",
        input_size: int | None = None,
        margin: float = 0.01,
        bin_thresh: float | None = None,
        box_thresh: float | None = None,
        top_k_per_class: int | None = None,
    ) -> None:
        from doctr.models import detection_predictor, kie_predictor, ocr_predictor

        if mode not in ("words", "kie"):
            raise ValueError("mode must be 'words' or 'kie'")
        self.mode = mode
        self.margin = margin
        # A receipt holds at most one value per field: keeping the k best-scored regions per class removes
        # most false positives without retraining
        self.top_k_per_class = top_k_per_class
        self.device = device
        model, self.cfg = load_detection_checkpoint(checkpoint, device)
        size = input_size or self.cfg.get("input_size") or model.cfg["input_shape"][-1]
        model.cfg = {**model.cfg, "input_shape": (3, size, size)}
        if bin_thresh is not None:
            model.postprocessor.bin_thresh = bin_thresh
        if box_thresh is not None:
            model.postprocessor.box_thresh = box_thresh
        self.class_names = list(self.cfg["class_names"])
        self.field_detector = detection_predictor(
            arch=model,
            pretrained=False,
            assume_straight_pages=self.cfg.get("assume_straight_pages", True),
            preserve_aspect_ratio=True,
            symmetric_pad=True,
        ).to(device)
        if mode == "words":
            self.ocr = ocr_predictor(det_arch_words, reco_arch, pretrained=True, assume_straight_pages=True).to(device)
        else:
            self.kie = kie_predictor(det_arch=model, reco_arch=reco_arch, pretrained=True, assume_straight_pages=True)
            self.kie = self.kie.to(device)

    @torch.no_grad()
    def detect(self, pages: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        """Relative field boxes per page: {class: (N, 5) [xmin, ymin, xmax, ymax, score]}, best scores first."""
        results = []
        for regions in self.field_detector(pages):
            kept = {}
            for cls_name, boxes in regions.items():
                boxes = np.asarray(boxes)
                if boxes.ndim == 2 and boxes.shape[1] >= 5:
                    order = np.argsort(-boxes[:, 4])
                elif boxes.ndim == 3 and boxes.shape[1] == 5:
                    order = np.argsort(-boxes[:, 4, 0])
                else:
                    order = np.arange(len(boxes))
                boxes = boxes[order]
                if self.top_k_per_class is not None:
                    boxes = boxes[: self.top_k_per_class]
                kept[cls_name] = boxes
            results.append(kept)
        return results

    @torch.no_grad()
    def __call__(self, pages: list[np.ndarray]) -> list[dict[str, list[dict]]]:
        if self.mode == "kie":
            doc = self.kie(pages)
            results = []
            for page in doc.pages:
                fields: dict[str, list[dict]] = {c: [] for c in self.class_names}
                for cls_name, preds in page.predictions.items():
                    for pred in preds:
                        (x0, y0), (x1, y1) = (
                            pred.geometry[:2]
                            if len(pred.geometry) == 2
                            else (
                                np.min(pred.geometry, axis=0),
                                np.max(pred.geometry, axis=0),
                            )
                        )
                        fields.setdefault(cls_name, []).append({
                            "value": pred.value,
                            "confidence": float(pred.confidence),
                            "detection_score": None,
                            "geometry": [round(float(v), 4) for v in (x0, y0, x1, y1)],
                            "words": [pred.value],
                        })
                for cls_name, v in fields.items():
                    if self.top_k_per_class is not None:
                        v.sort(key=lambda f: -f["confidence"])
                        fields[cls_name] = v = v[: self.top_k_per_class]
                    v.sort(key=lambda f: (f["geometry"][1], f["geometry"][0]))
                results.append(fields)
            return results

        regions_per_page = self.detect(pages)
        ocr_doc = self.ocr(pages)
        results = []
        for regions, page in zip(regions_per_page, ocr_doc.pages):
            words = [
                {
                    "value": w.value,
                    "confidence": float(w.confidence),
                    "box": [w.geometry[0][0], w.geometry[0][1], w.geometry[1][0], w.geometry[1][1]],
                }
                for block in page.blocks
                for line in block.lines
                for w in line.words
            ]
            results.append(
                read_fields_from_words({c: list(regions.get(c, [])) for c in self.class_names}, words, self.margin)
            )
        return results
