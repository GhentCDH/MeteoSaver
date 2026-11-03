"""Dump the exact ROI crop MeteoSaver would hand to OCR for every cell, plus which sheet cell it lands in.

Runs the real detection + transcription() with a recorder in place of the OCR call: the recorder
copies the ROI file and returns a unique token ("R000123"). The tokens that survive into the final
pre_QA_QC workbook give each ROI its (row, col). Done twice per sheet with identical detection:
  roi_bin/  = ROIs from the binarized, line-removed table (what Tesseract gets in production)
  roi_orig/ = the same boxes cut from the original colour clip
Output: sheet_runs/<file>/rois/{bin,orig}/<id>.png and sheet_runs/<file>/rois/cells.json {id: [row, col]}.

Run with .venv-easyocr/bin/python.   Usage: extract_rois.py [--only FILE ...] [--row-grouping columnwise]
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import shutil
import types
from pathlib import Path

import openpyxl

import compare_sheet as cs

MANUAL = cs.REPO / "results/06_manually_transcribed_hydroclimate_data"


class Recorder:
    def __init__(self, dest: Path):
        self.dest, self.i = dest, 0
        dest.mkdir(parents=True, exist_ok=True)

    def __call__(self, image_path, *a, **k) -> str:
        shutil.copy(image_path, self.dest / f"{self.i}.png")
        self.i += 1
        return f"R{self.i - 1:06d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--row-grouping", choices=["kmeans", "columnwise"], default="kmeans", help="columnwise writes to rois_columnwise/ (see row_grouping.py)")
    args = ap.parse_args()
    cfg = configparser.ConfigParser()
    cfg.read(cs.REPO / "configuration.ini")
    # configuration.ini's full_datadir may point at a single station (the TUI rewrites it); run_pipeline appends <station>/<file>
    cfg["Directories"]["full_datadir"] = str(cs.REPO / "data/00_post1960_DRC_hydroclimate_datasheet_images")
    os.environ["TESSDATA_PREFIX"] = str(cs.REPO / "OCR_HTR_models")

    for p in sorted(MANUAL.glob("*/*_manually_entered_temperatures.xlsx")):
        station, f = p.parent.name, p.name.split("_manually")[0]
        if args.only and f not in args.only:
            continue
        root = cs.HERE / "sheet_runs" / f / ("rois" if args.row_grouping == "kmeans" else f"rois_{args.row_grouping}")
        if (root / "cells.json").exists():
            continue
        ns = types.SimpleNamespace(station=station, file=f)
        counts = {}
        for src in ("bin", "orig"):
            rec = Recorder(root / src)
            xlsx = cs.run_pipeline(f"roi_{src}" + ("" if args.row_grouping == "kmeans" else f"_{args.row_grouping}"), ns, cfg, patch_ocr=rec, roi_source="binarized" if src == "bin" else "original", row_grouping=args.row_grouping, hoist_borders=True)
            counts[src] = rec.i
            if src == "orig":
                ws = openpyxl.load_workbook(xlsx).active
                cells = {}
                for row in ws.iter_rows():
                    for c in row:
                        m = re.fullmatch(r"R(\d{6})", str(c.value or "").strip())
                        if m:
                            cells[int(m.group(1))] = [c.row, c.column]
        assert counts["bin"] == counts["orig"], counts
        (root / "cells.json").write_text(json.dumps(cells))
        print(f"{f}: {counts['orig']} ROIs, {len(cells)} placed in the sheet", flush=True)


if __name__ == "__main__":
    main()
