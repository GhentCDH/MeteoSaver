"""Build one self-contained HTML page comparing Tesseract and GLM-OCR on the annotated cell crops.

Reads results.csv (per-cell predictions on the 2,854 test crops, written by evaluate.py), summary.json (latency),
ocr_only_results.csv (whole-sheet cells vs manual transcription) and the crops in common.DATA_DIR.
Writes comparison/index.html (crops embedded as small WebP, no server or network needed). Nothing is re-run.

Usage: python3 build_comparison.py        (needs Pillow)
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from pathlib import Path

from PIL import Image

from common import DATA_DIR, HERE

OUT = HERE / "comparison"

LABELS = {
    "tess_V6_digits": ("Tesseract V6, digits only", "tess"),
    "tess_V6_punct": ("Tesseract V6, digits + , - =", "tess"),
    "tess_V6_punct_pad": ("Tesseract V6, digits + , - =, padded (best Tesseract)", "tess"),
    "tess_V9_405_digits": ("Tesseract V9_405, digits only (MeteoSaver config)", "tess"),
    "tess_V9_405_punct": ("Tesseract V9_405, digits + , - =", "tess"),
    "tess_V9_405_punct_pad": ("Tesseract V9_405, digits + , - =, padded", "tess"),
    "easyocr_digits": ("EasyOCR, digits only", "easy"),
    "easyocr_punct": ("EasyOCR, digits + , - =", "easy"),
    "meteosaver_chain": ("MeteoSaver chain (V9_405, EasyOCR fallback)", "tess"),
    "zeroshot": ("GLM zeroshot", "glm"),
    "n500": ("GLM n500", "glm"),
    "n2000": ("GLM n2000", "glm"),
    "n8000": ("GLM n8000", "glm"),
    "full": ("GLM full", "glm"),
}


def thumb(path: Path) -> str:
    buf = io.BytesIO()
    Image.open(path).convert("L").save(buf, "WEBP", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def family(name: str) -> str:
    n = name.lower()
    return "glm" if n.startswith("glm") else "easy" if n.startswith("easyocr") else "tess"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-grouping", choices=["kmeans", "columnwise"], default="columnwise", help="which whole-sheet run to show (columnwise fixes the row shift on sheet 601)")
    suffix = "" if (rg := ap.parse_args().row_grouping) == "kmeans" else f"_{rg}"
    rows = list(csv.DictReader(open(HERE / "results.csv", encoding="utf-8")))
    summary = json.loads((HERE / "summary.json").read_text())
    systems = [k for k in rows[0] if k not in ("stem", "gt")]
    data = {
        "systems": [{"id": s, "label": LABELS.get(s, (s, family(s)))[0], "family": LABELS.get(s, (s, family(s)))[1],
                     "sec": summary.get(s, {}).get("sec_per_cell")} for s in systems],
        "stems": [r["stem"] for r in rows],
        "gt": [r["gt"] for r in rows],
        "pred": {s: [r[s] for r in rows] for s in systems},
        "img": [thumb(DATA_DIR / f"{r['stem']}.png") for r in rows],
    }
    sheets = list(csv.DictReader(open(HERE / f"ocr_only_results{suffix}.csv", encoding="utf-8")))
    sys2 = [k for k in sheets[0] if k not in ("sheet", "row", "col", "truth")]
    def sheet_crop(r: dict) -> str:  # original-colour ROI of the cell, same box the engines read
        cells = json.loads((HERE / "sheet_runs" / r["sheet"] / f"rois{suffix}" / "cells.json").read_text())
        idx = {tuple(v): k for k, v in cells.items()}[(int(r["row"]), int(r["col"]))]
        return thumb(HERE / "sheet_runs" / r["sheet"] / f"rois{suffix}" / "orig" / f"{idx}.png")

    data["sheet"] = {
        "rc": [[int(r["row"]), int(r["col"])] for r in sheets],
        "img": [sheet_crop(r) for r in sheets],
        "systems": [{"id": s, "label": s, "family": family(s)} for s in sys2],
        "sheets": [r["sheet"] for r in sheets],
        "truth": [r["truth"] for r in sheets],
        "pred": {s: [r[s] for r in sheets] for s in sys2},
    }
    OUT.mkdir(exist_ok=True)
    html = (HERE / "comparison_template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {OUT / 'index.html'} ({len(html) / 1e6:.1f} MB, {len(rows)} cells, {len(systems)} systems)")


if __name__ == "__main__":
    main()
