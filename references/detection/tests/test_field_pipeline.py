"""Tests of the receipt field pipeline (references/detection), on synthetic data only.

pytest references/detection/tests
"""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

DETECTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DETECTION_DIR))

import convert_documentai as conv  # noqa: E402
import evaluate_fields  # noqa: E402
import field_utils as fu  # noqa: E402
import receipt_schema as rs  # noqa: E402

import utils as det_utils  # noqa: E402


def _load_layout_utils():
    spec = importlib.util.spec_from_file_location("layout_utils", DETECTION_DIR.parent / "layout" / "utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- checkpoint metadata -------------------------------------------------------------------------------------------


def test_load_detection_checkpoint_resolves_run_metadata(tmp_path):
    """The metadata written by train.py records the architecture as `architecture` and `args.arch` only: the
    returned cfg must expose `arch`, `input_size` and `assume_straight_pages`, which FieldExtractor reads."""
    from doctr.models import detection

    class_names = ["fecha", "valor_transferido"]
    model = detection.db_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, class_names=class_names)
    det_utils.save_checkpoint(model, tmp_path, "run")
    args = Namespace(arch="db_mobilenet_v3_large", input_size=768, rotation=False)
    det_utils.save_run_metadata(
        tmp_path,
        "run",
        {
            "architecture": args.arch,
            "input_size": args.input_size,
            **det_utils.run_metadata(args, task="detection", class_names=class_names, assume_straight_pages=True),
        },
    )
    # A mid-run checkpoint shares the run metadata
    det_utils.save_checkpoint(model, tmp_path, "run_epoch3")

    for name in ("run.pt", "run_epoch3.pt"):
        loaded, cfg = fu.load_detection_checkpoint(tmp_path / name, torch.device("cpu"), pretrained_backbone=False)
        assert cfg["arch"] == "db_mobilenet_v3_large"
        assert cfg["input_size"] == 768
        assert cfg["assume_straight_pages"] is True
        assert list(loaded.class_names) == class_names


def test_load_detection_checkpoint_falls_back_to_args(tmp_path):
    from doctr.models import detection

    model = detection.db_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, class_names=["a"])
    det_utils.save_checkpoint(model, tmp_path, "old")
    with open(tmp_path / "old.json", "w") as f:
        json.dump({"class_names": ["a"], "args": {"arch": "db_mobilenet_v3_large", "input_size": 640}}, f)
    _, cfg = fu.load_detection_checkpoint(tmp_path / "old.pt", torch.device("cpu"), pretrained_backbone=False)
    assert (cfg["arch"], cfg["input_size"], cfg["assume_straight_pages"]) == ("db_mobilenet_v3_large", 640, True)


# --- canonical keys ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("16/09/2026", "2026-09-16"),
        ("El 16 de septiembre de 2026", "2026-09-16"),
        ("2026/ago./26", "2026-08-26"),
        ("16.09.26", "2026-09-16"),
        # A time read with dots comes first and matches the numeric pattern: the date after it must still be found
        ("14.22.40 16.09.2026", "2026-09-16"),
        ("Hora 23.59.59 Fecha 01/10/2026", "2026-10-01"),
        # Month name with a 2-digit year, month first, English names, "del"
        ("28-SEP-26", "2026-09-28"),
        ("28 sep 26", "2026-09-28"),
        ("Sep 28, 2026", "2026-09-28"),
        ("Septiembre 28 de 2026", "2026-09-28"),
        ("28 September 2026", "2026-09-28"),
        ("September 28, 2026 11:35", "2026-09-28"),
        ("28 de sep. del 2026", "2026-09-28"),
        # A number followed by a month name is not a date when the day is impossible
        ("Total 25.00 mar 2026", ""),
        # No such day, truncated or implausible year: no date
        ("31/02/2026", ""),
        ("16/09/202", ""),
        ("16/09/1899", ""),
        ("sin fecha", ""),
    ],
)
def test_key_fecha(text, expected):
    assert fu.extract_key("fecha", text) == expected


def test_extract_time_follows_the_valid_date():
    assert fu.extract_time("14.22.40 16.09.2026 08:05") == "08:05"
    assert fu.extract_time("16/09/2026 2:15 pm") == "14:15"
    assert fu.extract_time("16/09/2026") == ""


@pytest.mark.parametrize(
    "text, expected",
    [
        ("$ 1,250.00", "1250.00"),
        ("USD 1.250,00", "1250.00"),
        ("$25", "25.00"),
        # Sentence punctuation after the amount is not a separator
        ("USD 25.00.", "25.00"),
        ("25,50,", "25.50"),
        ("1.234.567,89", "1234567.89"),
        ("sin monto", ""),
    ],
)
def test_key_amount(text, expected):
    assert fu.extract_key("valor_transferido", text) == expected


# --- receipt payload -----------------------------------------------------------------------------------------------


def _entry(value, cls_name, score=0.9, conf=0.95, source="detector"):
    return {
        "value": value,
        "normalized": fu.extract_key(cls_name, value),
        "confidence": conf,
        "detection_score": score,
        "geometry": [0.1, 0.1, 0.5, 0.2],
        "words": value.split(),
        "source": source,
    }


def _validate(payload):
    jsonschema = pytest.importorskip("jsonschema")
    with open(DETECTION_DIR / "receipt_schema.json") as f:
        jsonschema.validate(payload, json.load(f))


def test_receipt_payload_types_and_selection():
    fields = {
        "fecha": [_entry("El 16 de septiembre de 2026 14:22", "fecha")],
        "valor_transferido": [_entry("$ 1,250.00", "valor_transferido", conf=0.8)],
        # The best-scored region reads nothing usable: the one with a canonical value wins
        "numero_comprobante": [
            _entry("Comprobante", "numero_comprobante", score=0.95),
            _entry("No. 0000737211", "numero_comprobante", score=0.7),
        ],
        "cuenta_destino": [_entry("Cta. ****** 3861", "cuenta_destino")],
        "nombre_cuenta_origen": [_entry("José  Pérez-López", "nombre_cuenta_origen")],
        "numero_control": [],
    }
    payload = rs.to_receipt_payload(fields, model={"checkpoint": "x.pt"})
    _validate(payload)
    f = payload["fields"]
    assert payload["schema_version"] == rs.SCHEMA_VERSION
    assert (f["date"]["normalized"], f["date"]["time"]) == ("2026-09-16", "14:22")
    assert f["total"]["normalized"] == "1250.00"
    assert f["total"]["confidence"] == 0.8  # min of the reading confidence and the detection score
    assert f["receipt_number"]["normalized"] == "0000737211"
    assert f["receipt_number"]["alternatives"][0]["value"] == "Comprobante"
    assert (f["destination"]["normalized"], f["destination"]["account_suffix"]) == ("****3861", "3861")
    assert f["payer_name"]["normalized"] == "JOSE PEREZ LOPEZ"
    assert f["receipt_control"]["value"] is None and not f["receipt_control"]["valid"]
    assert f["payer_document"]["value"] is None
    assert payload["missing_required"] == []


def test_receipt_payload_account_holder_and_missing_fields():
    fields = {"cuenta_destino": [_entry("MARIA LOPEZ", "cuenta_destino", score=None, source="fallback")]}
    payload = rs.to_receipt_payload(fields)
    _validate(payload)
    dest = payload["fields"]["destination"]
    assert (dest["normalized"], dest["account_suffix"], dest["account_holder"]) == ("MARIA LOPEZ", None, "MARIA LOPEZ")
    assert dest["confidence"] == 0.95 and dest["source"] == "fallback"
    assert payload["missing_required"] == ["receipt_number", "date", "total"]

    empty = rs.to_receipt_payload({})
    _validate(empty)
    assert empty["missing_required"] == list(rs.REQUIRED)


# --- evaluation ----------------------------------------------------------------------------------------------------


class _FakeExtractor:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def __call__(self, pages):
        return [next(self.outputs)]


def test_evaluate_value_accuracy(tmp_path):
    (tmp_path / "images").mkdir()
    docs = []
    for i in range(3):
        Image.new("RGB", (100, 100), "white").save(tmp_path / "images" / f"{i}.png")
        docs.append({"image": f"{i}.png", "id": str(i), "boxes": []})
    box = [[10, 10], [50, 10], [50, 20], [10, 20]]
    # doc 0: two annotated regions for one field (holder name + masked number), doc 1: holder name only, doc 2: absent
    docs[0]["boxes"] = [
        {"class": "cuenta_destino", "polygon": box, "text": "MARIA LOPEZ"},
        {"class": "cuenta_destino", "polygon": [[10, 30], [50, 30], [50, 40], [10, 40]], "text": "****3861"},
    ]
    docs[1]["boxes"] = [{"class": "cuenta_destino", "polygon": box, "text": "MARIA LOPEZ"}]
    manifest = {"class_names": ["cuenta_destino"], "documents": docs}

    def pred(value, score):
        return {**_entry(value, "cuenta_destino", score=score), "geometry": [0.1, 0.3, 0.5, 0.4]}

    outputs = [
        # Two regions returned, the best one holds the right account: one value per field is right
        {"cuenta_destino": [pred("XXXX3861", 0.9), pred("otra 1234", 0.3)]},
        {"cuenta_destino": [pred("****3861", 0.9)]},  # right account, but the annotation is the holder's name
        {"cuenta_destino": []},
    ]
    report = evaluate_fields.evaluate(manifest, tmp_path, _FakeExtractor(outputs), 0.5, ["cuenta_destino"])
    value = report["per_class"]["cuenta_destino"]["value"]
    assert value["by_annotation"]["digits"] == {"n": 1, "correct": 1, "accuracy": 1.0}
    assert value["by_annotation"]["letters"] == {"n": 1, "correct": 0, "accuracy": 0.0}
    assert value["by_annotation"]["absent"] == {"n": 1, "correct": 1, "accuracy": 1.0}
    assert report["docs_all_required_value_ok"] == 2
    # The region-count based key metric fails doc 0 (2 annotations vs 2 regions of which one is wrong)
    assert report["per_document"][0]["all_required_value_ok"] is True
    md = evaluate_fields.to_markdown({**report, "diagnosis": "", "recommendations": []}, "t")
    assert "One value per field" in md


# --- dataset conversion --------------------------------------------------------------------------------------------


def test_entity_polygon_unions_page_refs():
    def ref(x0, y0, x1, y1, page=None):
        r = {
            "boundingPoly": {
                "normalizedVertices": [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}]
            }
        }
        if page is not None:
            r["page"] = page
        return r

    ent = {"pageAnchor": {"pageRefs": [ref(0.1, 0.1, 0.5, 0.2), ref(0.1, 0.2, 0.3, 0.3), ref(0, 0, 1, 1, page="1")]}}
    poly = conv.entity_polygon(ent, 200, 100)
    np.testing.assert_allclose(poly, [[20, 10], [100, 10], [100, 30], [20, 30]])
    assert conv.entity_polygon({"pageAnchor": {"pageRefs": []}}, 200, 100) is None


def _doc(doc_id, phash, number):
    return {
        "id": doc_id,
        "phash": phash,
        "pixel_hash": doc_id,
        "boxes": [{"class": "numero_comprobante", "text": number}],
    }


def test_same_transaction_grouped_and_audited():
    a, b = "0" * 256, "1" * 256  # as far apart as two page hashes can be
    docs = [
        _doc("screenshot", a, "No. 0000737211"),
        _doc("photo", b, "0000737211"),
        _doc("other", b[:-5] + "0" * 5, "12"),
    ]
    logs = []
    groups = conv.group_documents(docs, 8, logs.append)
    assert sorted(len(g) for g in groups) == [1, 2]  # the photo and "other" are near duplicates by hash only
    groups = conv.group_documents(docs, 8, logs.append, key_classes=["numero_comprobante"])
    assert [len(g) for g in groups] == [3]
    # Short identifiers ("12") never group anything
    assert conv.transaction_keys(docs[2], ["numero_comprobante"]) == set()

    shared = conv.shared_transactions({"train": [docs[0]], "test": [docs[1]]}, ["numero_comprobante"])
    assert shared == [
        {"class": "numero_comprobante", "key": "0000737211", "docs": {"train": ["screenshot"], "test": ["photo"]}}
    ]


# --- training ------------------------------------------------------------------------------------------------------


def test_model_ema_update():
    layout_utils = _load_layout_utils()
    model = torch.nn.BatchNorm1d(2)
    ema = layout_utils.ModelEMA(model, decay=0.5, tau=1e-9)  # decay ramp already complete
    with torch.no_grad():
        model.weight.fill_(3.0)
    model.num_batches_tracked.fill_(7)
    ema.update(model)
    torch.testing.assert_close(ema.module.weight, torch.full((2,), 2.0))  # 0.5 * 1 + 0.5 * 3
    assert int(ema.module.num_batches_tracked) == 7
    assert not any(p.requires_grad for p in ema.module.parameters())
