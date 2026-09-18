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
  - key match: both texts reduced to the canonical value of the field (date -> YYYY-MM-DD, amount -> 0.00,
    receipt number without "No." prefix, account -> digit suffix, name -> letters), see field_utils.extract_key
  - the percentage of documents where every required field is read correctly (exact and key)
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from field_utils import (
    FieldExtractor,
    box_iou,
    extract_key,
    keys_match,
    load_manifest,
    normalize_text,
    parse_min_score,
    polygon_to_box,
    resolve_device,
)
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
    text = {c: {"n": 0, "exact": 0, "exact_nospace": 0, "key": 0} for c in class_names}
    docs_all_required, docs_all_required_key, n_docs = 0, 0, 0
    per_doc = []
    # Per class and document: (score of the best-scored prediction or None, has ground truth, best pred matches)
    sweep: dict[str, list[tuple[float | None, bool, bool]]] = {c: [] for c in class_names}

    for doc in manifest["documents"]:
        n_docs += 1
        img = np.asarray(Image.open(data_dir / "images" / doc["image"]).convert("RGB"))
        h, w = img.shape[:2]
        fields = extractor([img])[0]
        gt_by_class: dict[str, list[dict]] = collections.defaultdict(list)
        for b in doc["boxes"]:
            gt_by_class[b["class"]].append(b)
        doc_ok, doc_key_ok = True, True
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
            best = max(preds, key=lambda p: p["detection_score"] or 0.0) if preds else None
            best_idx = preds.index(best) if best is not None else None
            sweep[c].append((
                best["detection_score"] if best is not None else None,
                bool(gts),
                best_idx is not None and best_idx in matched_pred.values(),
            ))
            field_ok = field_key_ok = len(gts) == len(preds)  # no missing, no extra
            entries = []
            for gi, gt in enumerate(gts):
                text[c]["n"] += 1
                pred_value = preds[matched_pred[gi]]["value"] if gi in matched_pred else ""
                gt_norm, pred_norm = normalize_text(gt["text"], c), normalize_text(pred_value, c)
                exact = gt_norm == pred_norm
                gt_key, pred_key = extract_key(c, gt["text"]), extract_key(c, pred_value)
                key_ok = keys_match(c, gt_key, pred_key)
                text[c]["exact"] += int(exact)
                text[c]["exact_nospace"] += int(gt_norm.replace(" ", "") == pred_norm.replace(" ", ""))
                text[c]["key"] += int(key_ok)
                field_ok &= exact
                field_key_ok &= key_ok
                entries.append({
                    "gt": gt["text"],
                    "pred": pred_value,
                    "exact": exact,
                    "gt_key": gt_key,
                    "pred_key": pred_key,
                    "key_ok": key_ok,
                })
            doc_report["fields"][c] = {
                "ok": field_ok,
                "key_ok": field_key_ok,
                "n_gt": len(gts),
                "n_pred": len(preds),
                "entries": entries,
            }
            if c in required:
                doc_ok &= field_ok
                doc_key_ok &= field_key_ok
        docs_all_required += int(doc_ok)
        docs_all_required_key += int(doc_key_ok)
        doc_report["all_required_ok"] = doc_ok
        doc_report["all_required_key_ok"] = doc_key_ok
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
                "key_match": text[c]["key"] / text[c]["n"] if text[c]["n"] else None,
            },
        }
    thresholds = tune_thresholds(sweep)
    for c in class_names:
        summary[c]["threshold"] = thresholds[c]

    return {
        "n_docs": n_docs,
        "iou_threshold": iou_thresh,
        "suggested_min_score": {
            c: t["best_threshold"] for c, t in thresholds.items() if t["best_threshold"] is not None
        },
        "required_fields": required,
        "docs_all_required_ok": docs_all_required,
        "docs_all_required_ok_ratio": docs_all_required / n_docs if n_docs else None,
        "docs_all_required_key_ok": docs_all_required_key,
        "docs_all_required_key_ok_ratio": docs_all_required_key / n_docs if n_docs else None,
        "per_class": summary,
        "per_document": per_doc,
    }


def tune_thresholds(sweep: dict[str, list[tuple[float | None, bool, bool]]]) -> dict[str, dict]:
    """For each class, sweep a minimum detection score for the best-scored region of every document and keep the
    value maximising F1 (a document without ground truth where a region survives the threshold counts as a false
    positive). Exact for `--top-k-per-class 1`, an approximation otherwise."""
    out = {}
    for c, rows in sweep.items():
        candidates = sorted({round(s, 3) for s, _, _ in rows if s is not None} | {0.0})
        best = {"best_threshold": None, "f1": None, "precision": None, "recall": None, "f1_at_zero": None}
        for t in candidates:
            tp = sum(1 for s, gt, ok in rows if s is not None and s >= t and gt and ok)
            fp = sum(1 for s, gt, ok in rows if s is not None and s >= t and not (gt and ok))
            fn = sum(1 for s, gt, ok in rows if gt and not (s is not None and s >= t and ok))
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            if t == 0.0:
                best["f1_at_zero"] = f1
            if best["f1"] is None or f1 > best["f1"] + 1e-9:
                best.update({"best_threshold": t, "f1": f1, "precision": p, "recall": r})
        out[c] = best
    return out


def fmt(v, pct=True):
    if v is None:
        return "-"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def recommendations(report: dict, iou_thresh: float) -> tuple[str, list[str]]:
    """Derive a one-line diagnosis and actionable recommendations from the per-class metrics."""
    per_class = report["per_class"]
    n_docs = report["n_docs"]
    recs: list[str] = []
    recalls = [s["detection"]["recall"] for s in per_class.values() if s["detection"]["recall"] is not None]
    precisions = [s["detection"]["precision"] for s in per_class.values() if s["detection"]["precision"] is not None]
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    total_fp = sum(s["detection"]["fp"] for s in per_class.values())
    total_tp = sum(s["detection"]["tp"] for s in per_class.values())

    if mean_recall < 0.3:
        diagnosis = "Under-trained detector: most fields are not localised yet."
        recs.append(
            "Train longer (hundreds of epochs on a dataset this small; keep `--sched cosine`) and check that the "
            "validation loss is still decreasing. Try `--bin-thresh 0.1 --box-thresh 0.05` here to see whether the "
            "fields are emerging below the default thresholds."
        )
    elif mean_recall < 0.8:
        diagnosis = "Detector is learning but misses a sizeable share of the fields."
        recs.append(
            "Continue training or lower `--bin-thresh`; inspect the per-class recall below and add data for the "
            "weakest classes and their templates."
        )
    elif mean_precision < 0.8:
        diagnosis = "Fields are found but too many extra regions are produced."
        recs.append("Raise `--box-thresh`, use `--top-k-per-class 1` and validate the read text with per-field rules.")
    else:
        diagnosis = "Detection is solid; remaining errors come from reading the text."

    if total_fp > 3 * max(total_tp, 1) and report.get("top_k_per_class") is None:
        recs.append(
            f"{total_fp} false positives for {total_tp} true positives: pass `--top-k-per-class 1` "
            "(each field occurs at most once per receipt) and/or raise `--box-thresh`."
        )
    for c, s in per_class.items():
        d, t = s["detection"], s["text"]
        if d["tp"] + d["fn"] == 0:
            continue
        if d["recall"] is not None and d["recall"] < 0.5 and d["tp"] + d["fn"] < 15:
            recs.append(
                f"`{c}`: recall {d['recall']:.0%} with only {d['tp'] + d['fn']} annotated boxes in this split; "
                "collect more examples of this field (target >= 100 per class) or oversample the documents "
                "that contain it."
            )
        fp_absent = s["false_positive_rate_when_absent"]
        if fp_absent is not None and fp_absent > 0.3 and s["absent_docs"] >= 3:
            recs.append(
                f"`{c}`: predicted on {fp_absent:.0%} of the {s['absent_docs']} documents where it is absent; "
                "apply the suggested `--min-score` for this class (see the thresholds table)."
            )
        key = t.get("key_match")
        if d["recall"] is not None and d["recall"] >= 0.7 and key is not None and key < 0.7:
            gap = (t["exact_match_nospace"] or 0) - t["exact_match"]
            if gap > 0.2:
                recs.append(
                    f"`{c}`: detection is fine but the text differs mainly by spaces; use `--reco-mode words` "
                    "(default) and check the reading-order grouping (`--margin`)."
                )
            else:
                recs.append(
                    f"`{c}`: detected ({d['recall']:.0%} recall) but its canonical value is right only {key:.0%} "
                    "of the time; fine-tune the recogniser (Spanish vocabulary, `references/recognition`) or add "
                    "per-field validation."
                )
        if d["mean_iou"] is not None and d["tp"] >= 3 and d["mean_iou"] < 0.7:
            recs.append(
                f"`{c}`: matched boxes have a low mean IoU ({d['mean_iou']:.2f}); the regions are loose or fragmented, "
                f"consider a larger `--margin` or a slightly lower IoU threshold than {iou_thresh} when reading."
            )
    if n_docs < 30:
        recs.append(
            f"Only {n_docs} validation documents: every percentage point above is ~{100 / n_docs:.0f} % of a document. "
            "Use grouped K-fold (`convert_documentai.py --folds 5`) before drawing conclusions."
        )
    return diagnosis, recs


def to_markdown(report: dict, title: str) -> str:
    lines = [f"# {title}", "", f"Documents: {report['n_docs']}  ·  IoU threshold: {report['iou_threshold']}", ""]
    lines.append("| class | P | R | F1 | mIoU | FP rate (absent) | text exact | text exact (no space) | key match |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c, s in report["per_class"].items():
        d, t = s["detection"], s["text"]
        lines.append(
            f"| {c} | {fmt(d['precision'])} | {fmt(d['recall'])} | {fmt(d['f1'])} | {fmt(d['mean_iou'], False)} | "
            f"{fmt(s['false_positive_rate_when_absent'])} ({s['absent_docs']} docs) | "
            f"{fmt(t['exact_match'])} | {fmt(t['exact_match_nospace'])} | {fmt(t['key_match'])} |"
        )
    lines.append("")
    lines.append(
        f"Documents with every required field ({', '.join(report['required_fields'])}) read correctly: "
        f"**{report['docs_all_required_ok']} / {report['n_docs']}** ({fmt(report['docs_all_required_ok_ratio'])}) "
        f"as exact text, **{report['docs_all_required_key_ok']} / {report['n_docs']}** "
        f"({fmt(report['docs_all_required_key_ok_ratio'])}) as canonical keys"
    )
    tuned = {c: s["threshold"] for c, s in report["per_class"].items() if s.get("threshold")}
    if tuned:
        lines += ["", "## Suggested minimum detection score per class (tuned on this split)", ""]
        lines.append("| class | min score | F1 with it | P | R | F1 without |")
        lines.append("|---|---|---|---|---|---|")
        for c, t in tuned.items():
            if t["best_threshold"] is None:
                continue
            lines.append(
                f"| {c} | {t['best_threshold']:.2f} | {fmt(t['f1'])} | {fmt(t['precision'])} | {fmt(t['recall'])} | "
                f"{fmt(t['f1_at_zero'])} |"
            )
        flags = " ".join(f"{c}={t['best_threshold']:.2f}" for c, t in tuned.items() if t["best_threshold"])
        lines += ["", f"Apply with: `--min-score {flags}` (in `evaluate_fields.py` and `kie_inference.py`)."]
    lines += ["", "## Diagnosis", "", report["diagnosis"], "", "## Recommendations", ""]
    lines += [f"{i}. {r}" for i, r in enumerate(report["recommendations"], 1)] or ["Nothing to report."]
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
        min_score=parse_min_score(args.min_score),
        merge_fragments=args.merge_fragments,
    )
    unknown = [c for c in args.required if c not in manifest["class_names"]]
    if unknown:
        raise ValueError(f"unknown required fields {unknown}, classes are {manifest['class_names']}")
    report = evaluate(manifest, data_dir, extractor, args.iou, args.required)
    report["checkpoint"] = str(args.checkpoint)
    report["top_k_per_class"] = args.top_k_per_class
    report["min_score"] = parse_min_score(args.min_score)
    report["merge_fragments"] = args.merge_fragments
    report["diagnosis"], report["recommendations"] = recommendations(report, args.iou)
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
    parser.add_argument(
        "--merge-fragments",
        action="store_true",
        help="merge vertically adjacent regions of the same class into one (multi-line fields detected line by line)",
    )
    parser.add_argument(
        "--min-score",
        nargs="*",
        default=None,
        metavar="CLASS=VALUE",
        help="per-class minimum detection score, e.g. numero_control=0.62 (the report suggests values)",
    )
    parser.add_argument("--device", default=None, help="cpu, mps, cuda, cuda:N or a CUDA index (default: auto)")
    parser.add_argument("--output", default=None, help="output path without extension (.json and .md are written)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
