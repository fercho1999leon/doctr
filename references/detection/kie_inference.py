# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Extract labelled fields from receipt/invoice images with a multi-class detector trained by train.py.

    python references/detection/kie_inference.py --checkpoint runs/db_resnet50_invoices.pt page.jpg [more.png ...]

The class names, architecture and target options are read from the `<checkpoint>.json` sidecar written by
train.py, so nothing has to be typed by hand. Output: one JSON object per image, `{class: [field, ...]}`.
Every field has `value` (text as read), `normalized` (canonical key, see field_utils.extract_key),
`confidence`, `detection_score`, `geometry` ([xmin, ymin, xmax, ymax], relative) and `words`.
Classes without detection map to an empty list, so `result["cuenta_destino"]` is always defined.
"""

from __future__ import annotations

import argparse
import json
import sys

from field_utils import FieldExtractor, parse_min_score, resolve_device

from doctr.io import DocumentFile


def main(args):
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
        fallback=args.fallback,
        straighten=args.straighten,
    )
    results = {}
    for path in args.images:
        pages = DocumentFile.from_pdf(path) if path.lower().endswith(".pdf") else DocumentFile.from_images(path)
        page_results = extractor(pages)
        results[path] = page_results[0] if len(page_results) == 1 else page_results
        if not args.quiet:
            angle = extractor.last_angles[0] if extractor.last_angles else 0
            print(f"\n== {path} ==" + (f" (rotated {angle} deg)" if angle else ""))
            fields = page_results[0] if len(page_results) == 1 else page_results[0]
            for cls_name in extractor.class_names:
                values = fields.get(cls_name, [])
                shown = (
                    " | ".join(
                        f"{f['normalized']!r} <- {f['value']!r} ({f['confidence']:.2f}, {f.get('source', 'detector')})"
                        for f in values
                    )
                    if values
                    else "-"
                )
                print(f"{cls_name:24s} {shown}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"\nwritten to {args.json}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Field extraction with a fine-tuned multi-class detector + standard OCR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("images", nargs="+", help="image (or PDF) files to process")
    parser.add_argument("--checkpoint", required=True, help="detector weights (.pt) trained with train.py")
    parser.add_argument("--reco-arch", default="parseq", help="recognition architecture (pretrained weights)")
    parser.add_argument("--word-det-arch", default="fast_base", help="word detector used in 'words' mode")
    parser.add_argument(
        "--reco-mode",
        choices=["words", "kie"],
        default="words",
        help="'words': OCR words assigned to field regions (handles multi-line fields); "
        "'kie': recognise each field crop as one item (kie_predictor behaviour)",
    )
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
        "--straighten",
        action="store_true",
        help="rotate photos/scans upright before detection (page orientation classifier + text line skew)",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="when the detector finds nothing for a field, search the page OCR lines by pattern (receipt-specific)",
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
        help="per-class minimum detection score, e.g. numero_control=0.62 (tune with evaluate_fields.py)",
    )
    parser.add_argument("--device", default=None, help="cpu, mps, cuda, cuda:N or a CUDA index (default: auto)")
    parser.add_argument("--json", default=None, help="write the results to this JSON file")
    parser.add_argument("--quiet", action="store_true", help="do not print the per-image summary")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
