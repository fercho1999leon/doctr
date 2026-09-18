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

from doctr.models import detection, layout

__all__ = [
    "box_iou",
    "load_manifest",
    "normalize_text",
    "polygon_to_box",
    "read_fields_from_words",
    "load_detection_checkpoint",
    "resolve_device",
    "FieldExtractor",
    "extract_key",
    "extract_time",
    "find_fallback",
    "keys_match",
    "merge_fragments",
    "parse_min_score",
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


def parse_min_score(items: list[str] | None) -> dict[str, float]:
    """Parse `--min-score cls=0.6 other=0.4` CLI values."""
    out: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--min-score expects CLASS=VALUE, got '{item}'")
        cls_name, value = item.split("=", 1)
        out[cls_name.strip()] = float(value)
    return out


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
                "normalized": extract_key(cls_name, value),
                "confidence": text_conf,
                "detection_score": score,
                "geometry": np.round(box, 4).tolist(),
                "words": [w["value"] for w in ordered],
            })
        fields.sort(key=lambda f: (f["geometry"][1], f["geometry"][0]))
        out[cls_name] = fields
    return out


def merge_fragments(boxes: np.ndarray, gap_ratio: float = 0.8) -> np.ndarray:
    """Merge straight boxes (N, 5) of one class that are vertically adjacent and horizontally overlapping into
    a single region (union box, max score). Multi-line fields are often detected line by line; merging them
    restores the annotated block so that it can be matched and read as a whole."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or len(boxes) < 2:
        return boxes
    order = np.argsort(boxes[:, 1])
    clusters: list[np.ndarray] = []
    for b in boxes[order]:
        merged = False
        for i, c in enumerate(clusters):
            h = max(min(c[3] - c[1], b[3] - b[1]), 1e-6)
            v_gap = max(b[1] - c[3], c[1] - b[3], 0.0)
            h_overlap = min(c[2], b[2]) - max(c[0], b[0])
            if v_gap <= gap_ratio * h and h_overlap > 0:
                clusters[i] = np.array([
                    min(c[0], b[0]),
                    min(c[1], b[1]),
                    max(c[2], b[2]),
                    max(c[3], b[3]),
                    max(c[4], b[4]) if len(b) > 4 else 1.0,
                ])
                merged = True
                break
        if not merged:
            clusters.append(b.copy() if len(b) > 4 else np.append(b[:4], 1.0))
    return np.stack(clusters)


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
    }
    is_layout = cfg["arch"].startswith("lw_detr")
    if not is_layout:
        kwargs["mask_empty_classes"] = cfg.get("mask_empty_classes", True)
    kwargs.update(overrides)
    model = (layout if is_layout else detection).__dict__[cfg["arch"]](**kwargs)
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
        min_score: dict[str, float] | None = None,
        merge_fragments: bool = False,
        fallback: bool = False,
    ) -> None:
        from doctr.models import detection_predictor, kie_predictor, ocr_predictor

        if mode not in ("words", "kie"):
            raise ValueError("mode must be 'words' or 'kie'")
        self.mode = mode
        self.margin = margin
        # A receipt holds at most one value per field: keeping the k best-scored regions per class removes
        # most false positives without retraining
        self.top_k_per_class = top_k_per_class
        # Per-class minimum detection score (tune it with `evaluate_fields.py --tune-thresholds`)
        self.min_score = dict(min_score or {})
        self.merge_fragments = merge_fragments
        # Pattern search over the page when the detector returns nothing for a field (words mode only)
        self.fallback = fallback
        self.device = device
        from doctr.models import layout_predictor

        model, self.cfg = load_detection_checkpoint(checkpoint, device)
        self.is_layout = self.cfg["arch"].startswith("lw_detr")
        size = input_size or self.cfg.get("input_size") or model.cfg["input_shape"][-1]
        model.cfg = {**model.cfg, "input_shape": (3, size, size)}
        if self.is_layout:
            # LW-DETR: one score threshold on the class logits (the `--box-thresh` CLI value)
            if box_thresh is not None:
                model.postprocessor.score_thresh = box_thresh
        else:
            if bin_thresh is not None:
                model.postprocessor.bin_thresh = bin_thresh
            if box_thresh is not None:
                model.postprocessor.box_thresh = box_thresh
        self.class_names = list(self.cfg["class_names"])
        build = layout_predictor if self.is_layout else detection_predictor
        self.field_detector = build(
            arch=model,
            pretrained=False,
            assume_straight_pages=self.cfg.get("assume_straight_pages", True),
            preserve_aspect_ratio=True,
            symmetric_pad=True,
        ).to(device)
        if mode == "words":
            self.ocr = ocr_predictor(det_arch_words, reco_arch, pretrained=True, assume_straight_pages=True).to(device)
        elif self.is_layout:
            raise ValueError("'kie' reading mode needs a text-detection checkpoint; LW-DETR checkpoints use 'words'")
        else:
            self.kie = kie_predictor(det_arch=model, reco_arch=reco_arch, pretrained=True, assume_straight_pages=True)
            self.kie = self.kie.to(device)

    @torch.no_grad()
    def detect(self, pages: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        """Relative field boxes per page: {class: (N, 5) [xmin, ymin, xmax, ymax, score]}, best scores first."""
        results = []
        raw = self.field_detector(pages)
        if self.is_layout:
            # {"class_names": [...], "boxes": (N, 4), "scores": [...]} -> {class: (N, 5)}
            raw = [
                {
                    c: np.array(
                        [[*b, s] for n, b, s in zip(page["class_names"], page["boxes"], page["scores"]) if n == c],
                        dtype=np.float64,
                    ).reshape(-1, 5)
                    for c in self.class_names
                }
                for page in raw
            ]
        for regions in raw:
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
                thr = self.min_score.get(cls_name)
                if thr is not None and len(boxes):
                    scores = boxes[:, 4] if boxes.ndim == 2 else boxes[:, 4, 0]
                    boxes = boxes[scores >= thr]
                if self.merge_fragments and boxes.ndim == 2 and len(boxes) > 1:
                    boxes = merge_fragments(boxes)
                    boxes = boxes[np.argsort(-boxes[:, 4])]
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
                            "normalized": extract_key(cls_name, pred.value),
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
            words, lines = [], []
            for block in page.blocks:
                for line in block.lines:
                    if not line.words:
                        continue
                    lines.append({
                        "text": " ".join(w.value for w in line.words),
                        "confidence": float(np.mean([w.confidence for w in line.words])),
                        "box": [line.geometry[0][0], line.geometry[0][1], line.geometry[1][0], line.geometry[1][1]],
                    })
                    for w in line.words:
                        words.append({
                            "value": w.value,
                            "confidence": float(w.confidence),
                            "box": [w.geometry[0][0], w.geometry[0][1], w.geometry[1][0], w.geometry[1][1]],
                        })
            fields = read_fields_from_words({c: list(regions.get(c, [])) for c in self.class_names}, words, self.margin)
            for cls_name, entries in fields.items():
                for entry in entries:
                    entry["source"] = "detector"
                # Fallback when the detector found nothing, or only regions whose text yields no usable value
                if self.fallback and not any(e["normalized"] for e in entries):
                    hit = find_fallback(cls_name, lines)
                    if hit is not None:
                        fields[cls_name] = [hit]
            results.append(fields)
        return results


# --- Canonical keys per field -------------------------------------------------------------------------
# Receipts write the same value in many ways ("El 16 de septiembre de 2026", "16/09/2026", "2026/sep./16",
# "No.0000737211", "****** 3861" vs "*** *** 3861"). `extract_key` reduces a field value to the canonical
# form a downstream system actually needs, so that inference returns it and evaluation scores it.

_MONTHS_ES = {
    "ene": 1, "enero": 1, "feb": 2, "febrero": 2, "mar": 3, "marzo": 3, "abr": 4, "abril": 4, "may": 5, "mayo": 5,
    "jun": 6, "junio": 6, "jul": 7, "julio": 7, "ago": 8, "agosto": 8, "sep": 9, "sept": 9, "set": 9,
    "septiembre": 9, "setiembre": 9, "oct": 10, "octubre": 10, "nov": 11, "noviembre": 11, "dic": 12,
    "diciembre": 12, "jan": 1, "apr": 4, "aug": 8, "dec": 12,
}  # fmt: skip
_MONTH_RE = "|".join(sorted(_MONTHS_ES, key=len, reverse=True))
_SEP = r"(?:\s|[-/.,:])*"  # loose separators the OCR leaves between date tokens
_DATE_PATTERNS = [
    # 16 de septiembre de 2026 / 16 sept 2026 / 16-sep-2026 / E116 de septiembre de - 2026 (OCR glued "El")
    re.compile(rf"(\d{{1,2}}){_SEP}(?:de)?{_SEP}({_MONTH_RE}){_SEP}(?:de)?{_SEP}(\d{{4}})", re.I),
    # 2026/ago./26 / 2026-sep-16
    re.compile(rf"(\d{{4}}){_SEP}({_MONTH_RE}){_SEP}(\d{{1,2}})", re.I),
    # 2026/09/16 / 2026-09-16
    re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"),
    # 16/09/2026 / 16-09-2026 / 16.09.26
    re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})"),
]
# 14:22, 14:22:40, 08.23:17 (OCR), possibly glued to the date: searched right after the date first
_TIME_AFTER_RE = re.compile(r"\s*(\d{1,2})[:.](\d{2})(?:[:.]\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?", re.I)
_TIME_ANY_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?", re.I)
_PREFIX_RE = re.compile(r"^(?:n[o°º]\.?|nro\.?|num(?:ero)?\.?|#|comprobante|control|ref\.?)\s*:?\s*", re.I)


def _key_fecha(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    for idx, pat in enumerate(_DATE_PATTERNS):
        m = pat.search(t)
        if not m:
            continue
        a, b, c = m.groups()
        try:
            if idx == 0:
                day, month, year = int(a), _MONTHS_ES[b.lower()], int(c)
            elif idx == 1:
                year, month, day = int(a), _MONTHS_ES[b.lower()], int(c)
            elif idx == 2:
                year, month, day = int(a), int(b), int(c)
            else:
                day, month, year = int(a), int(b), int(c)
                if year < 100:
                    year += 2000
            if not (1 <= day <= 31 and 1 <= month <= 12):
                continue
        except (KeyError, ValueError):
            continue
        # The canonical key is the day only: the time is optional on receipts and often printed on another
        # line, use `extract_time` when you need it
        return f"{year:04d}-{month:02d}-{day:02d}"
    return ""


def extract_time(text: str) -> str:
    """HH:MM found in a date field ("" if none). Handles glued times, OCR '.' for ':' and am/pm."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    tm = None
    for pat in _DATE_PATTERNS:
        m = pat.search(t)
        if m:
            tm = _TIME_AFTER_RE.match(t, m.end())
            break
    tm = tm or _TIME_ANY_RE.search(t)
    if not tm:
        return ""
    hour, minute, ampm = int(tm.group(1)), tm.group(2), (tm.group(3) or "").replace(".", "")
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute}" if 0 <= hour < 24 and int(minute) < 60 else ""


def _key_amount(text: str) -> str:
    m = re.search(r"\d[\d.,]*", text)
    if not m:
        return ""
    raw = m.group(0)
    if "," in raw and "." in raw:
        dec = "," if raw.rfind(",") > raw.rfind(".") else "."
        raw = raw.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif "," in raw:
        head, _, tail = raw.rpartition(",")
        raw = f"{head.replace(',', '')}.{tail}" if len(tail) == 2 else raw.replace(",", "")
    elif raw.count(".") > 1:
        head, _, tail = raw.rpartition(".")
        raw = f"{head.replace('.', '')}.{tail}"
    try:
        return f"{float(raw):.2f}"
    except ValueError:
        return ""


def _key_number(text: str) -> str:
    """Longest alphanumeric token holding at least 3 digits, without a 'No.'-style prefix."""
    t = _PREFIX_RE.sub("", unicodedata.normalize("NFKC", text).strip())
    tokens = re.findall(r"[A-Za-z0-9]+", t)
    tokens = [tok for tok in tokens if sum(ch.isdigit() for ch in tok) >= 3]
    if not tokens:
        # "No.0000737211" with the prefix glued: strip letters before the digits
        m = re.search(r"\d{3,}", t)
        return m.group(0) if m else ""
    return max(tokens, key=len).upper()


def _key_account(text: str) -> str:
    """Last run of digits: masked accounts (****3861, 21XXXXXXX61) are only identified by their suffix."""
    runs = re.findall(r"\d+", text)
    return runs[-1] if runs else ""


def _key_name(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"[^A-Z]", "", t.upper())


_KEY_EXTRACTORS = {
    "fecha": _key_fecha,
    "valor_transferido": _key_amount,
    "numero_comprobante": _key_number,
    "numero_control": _key_number,
    "cuenta_destino": _key_account,
    "nombre_cuenta_origen": _key_name,
}


# Fallback patterns searched over the OCR lines of the whole page when the detector finds nothing for a field.
# Receipts are highly regular, so a pattern hit is almost always the right value.
_FALLBACK_LINE_PATTERNS: dict[str, re.Pattern] = {
    # ****** 3861 / XXXXXX3861 / 21XXXXXXXXX61 / 210XXX3861 / 21X00000X61 (OCR)
    "cuenta_destino": re.compile(r"(?:[*xX•]{2,}\s*\d{2,4}\b|\b\d{2,3}[xX*0]{3,}\d{2,4}\b)"),
    "numero_comprobante": re.compile(
        r"(?:n[o°º]\.?|nro\.?|comprobante|documento|referencia|ref\.?)\s*:?\s*([A-Z0-9]{6,})", re.I
    ),
    "numero_control": re.compile(r"control\W{0,6}([0-9]{5,})", re.I),
    "valor_transferido": re.compile(r"(?:\$|usd)\s*\d[\d.,]*", re.I),
}


def find_fallback(field: str, lines: list[dict]) -> dict | None:
    """Search `lines` ({"text", "box", "confidence"}) for a value of `field` by pattern. Returns a field entry
    (same shape as the detector output, with `source="fallback"`) or None."""
    candidates: list[tuple[int, dict, str]] = []
    if field == "fecha":
        for i, line in enumerate(lines):
            if extract_key("fecha", line["text"]):
                candidates.append((i, line, line["text"]))
    else:
        pat = _FALLBACK_LINE_PATTERNS.get(field)
        if pat is None:
            return None
        for i, line in enumerate(lines):
            m = pat.search(line["text"])
            if m:
                candidates.append((i, line, m.group(0)))
    if not candidates:
        return None
    # Prefer the candidate whose canonical key is non-empty, then the first one in reading order
    for _, line, text in candidates:
        key = extract_key(field, text)
        if key:
            return {
                "value": text,
                "normalized": key,
                "confidence": float(line["confidence"]),
                "detection_score": None,
                "geometry": [round(float(v), 4) for v in line["box"]],
                "words": text.split(),
                "source": "fallback",
            }
    return None


def keys_match(field: str, gt_key: str, pred_key: str) -> bool:
    """Whether two canonical keys denote the same value.

    - masked accounts: one digit suffix must end with the other (`3861` vs `61`, `X` read as `0`)
    - names: fuzzy ratio >= 0.9 on the letters (tolerates one misread character, not a different person)
    - everything else: exact
    """
    if not gt_key or not pred_key:
        return False
    if field == "cuenta_destino":
        short, long_ = sorted((gt_key, pred_key), key=len)
        return len(short) >= 2 and long_.endswith(short)
    if field == "nombre_cuenta_origen":
        from rapidfuzz.distance import Levenshtein

        return Levenshtein.normalized_similarity(gt_key, pred_key) >= 0.9
    return gt_key == pred_key


def extract_key(field: str, text: str) -> str:
    """Canonical value of a field ("" when nothing usable is found). Unknown fields fall back to normalize_text."""
    fn = _KEY_EXTRACTORS.get(field)
    return fn(text or "") if fn else normalize_text(text or "", field)
