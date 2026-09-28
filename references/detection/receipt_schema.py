# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Versioned, typed receipt payload built from the output of `FieldExtractor`.

`FieldExtractor` returns every candidate region per detector class. A consumer (the ERP endpoint
`POST /api/v1/ocr/receipts`) needs one value per business field, typed and stable across model versions:

    {
      "schema_version": "1.0",
      "fields": {
        "date":  {"value": "16/09/2026 14:22", "normalized": "2026-09-16", "time": "14:22", "confidence": 0.91, ...},
        "total": {"value": "$ 1,250.00", "normalized": "1250.00", ...},
        ...
      },
      "missing_required": ["receipt_control"],
      "model": {...}
    }

Every field always has the same keys (null when nothing was found), so a consumer never needs to test for them.
The JSON Schema of the payload is `receipt_schema.json`, next to this file.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from field_utils import extract_key, extract_time

__all__ = ["SCHEMA_VERSION", "FIELDS", "REQUIRED", "select_entry", "to_receipt_payload"]

SCHEMA_VERSION = "1.0"

# Business field -> (detector class, type). `None`: not produced by the current model, always null
FIELDS: dict[str, tuple[str | None, str]] = {
    "receipt_number": ("numero_comprobante", "identifier"),
    "receipt_control": ("numero_control", "identifier"),
    "date": ("fecha", "date"),
    "total": ("valor_transferido", "amount"),
    "destination": ("cuenta_destino", "account"),
    "payer_name": ("nombre_cuenta_origen", "name"),
    "payer_document": (None, "identifier"),
}
REQUIRED = ("receipt_number", "date", "total", "destination")


def select_entry(entries: list[dict]) -> tuple[dict | None, list[dict]]:
    """Pick the value of a field among the candidate regions of its class.

    A candidate with a usable canonical value beats one without; then the highest detection score (pattern-fallback
    entries have none), then the highest reading confidence. Returns the chosen entry and the others, best first.
    """
    if not entries:
        return None, []
    ranked = sorted(
        entries,
        key=lambda e: (
            bool(e.get("normalized")),
            e.get("detection_score") if e.get("detection_score") is not None else -1.0,
            e.get("confidence") or 0.0,
        ),
        reverse=True,
    )
    return ranked[0], ranked[1:]


def _clean_name(text: str) -> str:
    """Readable upper-case name: accents removed, letters and single spaces only."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).upper()
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z\s]", " ", t)).strip()


def _typed(kind: str, cls_name: str, entry: dict) -> dict[str, Any]:
    """Type-specific part of a field: the canonical value and its validity, plus extras for some types."""
    key = entry.get("normalized") or extract_key(cls_name, entry.get("value", ""))
    out: dict[str, Any] = {"normalized": key or None, "valid": bool(key)}
    if kind == "date":
        out["time"] = extract_time(entry.get("value", "")) or None
    elif kind == "account":
        # Digits: the visible suffix of a masked account. Letters only: the account holder's name was read instead
        if key and key.isdigit():
            out.update(normalized=f"****{key}", account_suffix=key, account_holder=None)
        else:
            holder = _clean_name(entry.get("value", "")) or None
            out.update(normalized=holder, account_suffix=None, account_holder=holder)
    elif kind == "name":
        name = _clean_name(entry.get("value", ""))
        out.update(normalized=name or None, valid=bool(name))
    return out


def _empty_field(kind: str) -> dict[str, Any]:
    field: dict[str, Any] = {
        "value": None,
        "normalized": None,
        "valid": False,
        "confidence": None,
        "ocr_confidence": None,
        "detection_score": None,
        "source": None,
        "geometry": None,
        "alternatives": [],
    }
    if kind == "date":
        field["time"] = None
    elif kind == "account":
        field.update(account_suffix=None, account_holder=None)
    return field


def _confidence(entry: dict) -> float | None:
    """One number a consumer can threshold: the lower of the reading confidence and the detection score."""
    scores = [s for s in (entry.get("confidence"), entry.get("detection_score")) if s is not None]
    return round(float(min(scores)), 4) if scores else None


def to_receipt_payload(
    fields: dict[str, list[dict]],
    model: dict[str, Any] | None = None,
    max_alternatives: int = 2,
) -> dict[str, Any]:
    """Build the versioned receipt payload from the `FieldExtractor` output of one page.

    Args:
        fields: detector class -> candidate entries, as returned by `FieldExtractor.__call__` for one page
        model: description of the model that produced the fields (checkpoint name, architecture, ...)
        max_alternatives: number of runner-up values kept per field

    Returns:
        the payload described in the module docstring
    """
    out_fields: dict[str, dict[str, Any]] = {}
    for name, (cls_name, kind) in FIELDS.items():
        field = _empty_field(kind)
        entry, others = select_entry(fields.get(cls_name, []) if cls_name else [])
        if entry is not None:
            field.update(
                value=entry.get("value"),
                ocr_confidence=entry.get("confidence"),
                detection_score=entry.get("detection_score"),
                confidence=_confidence(entry),
                source=entry.get("source", "detector"),
                geometry=entry.get("geometry"),
            )
            field.update(_typed(kind, cls_name, entry))
            field["alternatives"] = [
                {
                    "value": o.get("value"),
                    "normalized": _typed(kind, cls_name, o)["normalized"],
                    "confidence": _confidence(o),
                }
                for o in others[:max_alternatives]
            ]
        out_fields[name] = field
    return {
        "schema_version": SCHEMA_VERSION,
        "fields": out_fields,
        "missing_required": [name for name in REQUIRED if not out_fields[name]["valid"]],
        "model": dict(model or {}),
    }
