# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Field-level evaluation of a multi-class detector (+ OCR reading) on a split produced by convert_documentai.py.

    python references/detection/evaluate_fields.py --checkpoint runs/db_resnet50_invoices.pt \
        --data docs/ENTRENAMIENTO/doctr/val

Reports, per class (matched by name, never by position):
  - detection precision / recall / F1 at a given IoU threshold
  - false positive rate on documents where the field is absent
  - exact match of the read text against the annotated `mentionText` (normalised, see field_utils.normalize_text)
  - the percentage of documents where every required field is read correctly
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from field_utils import FieldExtractor, box_iou, load_manifest, normalize_text, polygon_to_box, resolve_device
from PIL import Image


def match_boxes(gts: list[np.ndarray], preds: list[np.ndarray], iou_thresh: float) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by IoU (highest IoU first). Returns (gt_idx, pred_idx) pairs."""
    pairs = sorted(
        ((box_iou(g, p), gi, pi) for gi, g in enumerate(gts) for pi, p in enumerate(preds)),
        key=lambda t: -t[0],
    )
    used_g, used_p, matches = set(), set(), []
    for iou, gi, pi in pairs:
        if iou < iou_thresh:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi))
    return matches


def evaluate(manifest: dict, data_dir: Path, extractor: FieldExtractor, iou_thresh: float, required: list[str]) -> dict:
    class_names = manifest["class_names"]
    det = {c: {"tp": 0, "fp": 0, "fn": 0, "iou": []} for c in class_names}
    absent = {c: {"docs_absent": 0, "docs_with_fp": 0} for c in class_names}
    text = {c: {"n": 0, "exact": 0, "exact_nospace": 0} for c in class_names}
    docs_all_required, n_docs = 0, 0
    per_doc = []

    for doc in manifest["documents"]:
        n_docs += 1
        img = np.asarray(Image.open(data_dir / "images" / doc["image"]).convert("RGB"))
        h, w = img.shape[:2]
        fields = extractor([img])[0]
        gt_by_class: dict[str, list[dict]] = collections.defaultdict(list)
        for b in doc["boxes"]:
            gt_by_class[b["class"]].append(b)
        doc_ok = True
        doc_report = {"id": doc["id"], "fields": {}}
        for c in class_names:
            gts = gt_by_class.get(c, [])
            preds = fields.get(c, [])
            gt_boxes = [polygon_to_box(np.asarray(b["polygon"]) / [w, h]) for b in gts]
            pred_boxes = [np.asarray(p["geometry"]) for p in preds]
            matches = match_boxes(gt_boxes, pred_boxes, iou_thresh)
            det[c]["tp"] += len(matches)
            det[c]["fp"] += len(preds) - len(matches)
            det[c]["fn"] += len(gts) - len(matches)
            det[c]["iou"].extend(box_iou(gt_boxes[gi], pred_boxes[pi]) for gi, pi in matches)
            if not gts:
                absent[c]["docs_absent"] += 1
                if preds:
                    absent[c]["docs_with_fp"] += 1
            matched_pred = dict(matches)
            field_ok = len(gts) == len(preds)  # no missing, no extra
            entries = []
            for gi, gt in enumerate(gts):
                text[c]["n"] += 1
                pred_value = preds[matched_pred[gi]]["value"] if gi in matched_pred else ""
                gt_norm, pred_norm = normalize_text(gt["text"], c), normalize_text(pred_value, c)
                exact = gt_norm == pred_norm
                text[c]["exact"] += int(exact)
                text[c]["exact_nospace"] += int(gt_norm.replace(" ", "") == pred_norm.replace(" ", ""))
                field_ok &= exact
                entries.append({"gt": gt["text"], "pred": pred_value, "exact": exact})
            doc_report["fields"][c] = {"ok": field_ok, "n_gt": len(gts), "n_pred": len(preds), "entries": entries}
            if c in required:
                doc_ok &= field_ok
        docs_all_required += int(doc_ok)
        doc_report["all_required_ok"] = doc_ok
        per_doc.append(doc_report)

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
        return p, r, f

    summary = {}
    for c in class_names:
        p, r, f = prf(det[c]["tp"], det[c]["fp"], det[c]["fn"])
        summary[c] = {
            "detection": {
                **{k: det[c][k] for k in ("tp", "fp", "fn")},
                "precision": p,
                "recall": r,
                "f1": f,
                "mean_iou": float(np.mean(det[c]["iou"])) if det[c]["iou"] else None,
            },
            "absent_docs": absent[c]["docs_absent"],
            "false_positive_rate_when_absent": absent[c]["docs_with_fp"] / absent[c]["docs_absent"]
            if absent[c]["docs_absent"]
            else None,
            "text": {
                "n": text[c]["n"],
                "exact_match": text[c]["exact"] / text[c]["n"] if text[c]["n"] else None,
                "exact_match_nospace": text[c]["exact_nospace"] / text[c]["n"] if text[c]["n"] else None,
            },
        }
    return {
        "n_docs": n_docs,
        "iou_threshold": iou_thresh,
        "required_fields": required,
        "docs_all_required_ok": docs_all_required,
        "docs_all_required_ok_ratio": docs_all_required / n_docs if n_docs else None,
        "per_class": summary,
        "per_document": per_doc,
    }


def fmt(v, pct=True):
    if v is None:
        return "-"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def to_markdown(report: dict, title: str) -> str:
    lines = [f"# {title}", "", f"Documents: {report['n_docs']}  ·  IoU threshold: {report['iou_threshold']}", ""]
    lines.append("| class | P | R | F1 | mIoU | FP rate (absent) | text exact | text exact (no space) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c, s in report["per_class"].items():
        d, t = s["detection"], s["text"]
        lines.append(
            f"| {c} | {fmt(d['precision'])} | {fmt(d['recall'])} | {fmt(d['f1'])} | {fmt(d['mean_iou'], False)} | "
            f"{fmt(s['false_positive_rate_when_absent'])} ({s['absent_docs']} docs) | "
            f"{fmt(t['exact_match'])} | {fmt(t['exact_match_nospace'])} |"
        )
    lines.append("")
    lines.append(
        f"Documents with every required field ({', '.join(report['required_fields'])}) read correctly: "
        f"**{report['docs_all_required_ok']} / {report['n_docs']}** ({fmt(report['docs_all_required_ok_ratio'])})"
    )
    return "\n".join(lines) + "\n"


def main(args):
    data_dir = Path(args.data)
    manifest = load_manifest(data_dir)
    device = resolve_device(args.device)
    extractor = FieldExtractor(
        args.checkpoint,
        device,
        reco_arch=args.reco_arch,
        det_arch_words=args.word_det_arch,
        mode=args.reco_mode,
        input_size=args.input_size,
        margin=args.margin,
        bin_thresh=args.bin_thresh,
        box_thresh=args.box_thresh,
        top_k_per_class=args.top_k_per_class,
    )
    unknown = [c for c in args.required if c not in manifest["class_names"]]
    if unknown:
        raise ValueError(f"unknown required fields {unknown}, classes are {manifest['class_names']}")
    report = evaluate(manifest, data_dir, extractor, args.iou, args.required)
    report["checkpoint"] = str(args.checkpoint)
    report["reco_mode"] = args.reco_mode
    report["reco_arch"] = args.reco_arch
    title = f"Field evaluation: {Path(args.checkpoint).stem} on {data_dir.name} ({args.reco_mode}/{args.reco_arch})"
    md = to_markdown(report, title)
    print(md)
    out = Path(args.output) if args.output else data_dir / f"eval_{Path(args.checkpoint).stem}_{args.reco_mode}"
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    with open(out.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"report written to {out.with_suffix('.json')} and {out.with_suffix('.md')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-field evaluation of a multi-class detector + OCR reading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="detector weights (.pt) trained with train.py")
    parser.add_argument("--data", required=True, help="split folder produced by convert_documentai.py")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a detection to count as correct")
    parser.add_argument(
        "--required",
        nargs="+",
        default=["valor_transferido", "fecha", "cuenta_destino", "numero_comprobante"],
        help="fields that must all be correct for a document to count as fully extracted",
    )
    parser.add_argument("--reco-arch", default="parseq", help="recognition architecture (pretrained weights)")
    parser.add_argument("--word-det-arch", default="fast_base", help="word detector used in 'words' mode")
    parser.add_argument("--reco-mode", choices=["words", "kie"], default="words", help="see kie_inference.py")
    parser.add_argument("--input-size", type=int, default=None, help="detector input size (default: from sidecar)")
    parser.add_argument("--margin", type=float, default=0.01, help="relative margin added around field regions")
    parser.add_argument("--bin-thresh", type=float, default=None, help="override the detector binarisation threshold")
    parser.add_argument("--box-thresh", type=float, default=None, help="override the detector box score threshold")
    parser.add_argument(
        "--top-k-per-class",
        type=int,
        default=None,
        help="keep only the k best-scored regions per class (use 1 when every field occurs at most once per page)",
    )
    parser.add_argument("--device", default=None, help="cpu, mps, cuda, cuda:N or a CUDA index (default: auto)")
    parser.add_argument("--output", default=None, help="output path without extension (.json and .md are written)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
