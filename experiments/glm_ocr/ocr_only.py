"""OCR-only comparison on the ROI crops dumped by extract_rois.py (detection and placement held fixed).

Every engine sees the same crop for the same manually transcribed cell (cols Max/Min/Mean). Inputs:
  bin  = binarized ROI as MeteoSaver feeds Tesseract    orig = same box from the original colour clip
  +128 = additionally rescaled to 128 px high (the height GLM was fine-tuned on)
Score: digits-only match against the manual value in tenths ('17' accepted for 17.0), and CER on digits.

Run with .venv-easyocr/bin/python (spawns the GLM worker under .venv-glm).
Writes ocr_only_results.csv (per cell, per system) and prints the summary.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import openpyxl
from PIL import Image, ImageOps

from common import edit_distance
import compare_sheet as cs

MANUAL = cs.REPO / "results/06_manually_transcribed_hydroclimate_data"
TESS_DIR = str(cs.REPO / "OCR_HTR_models")


def tesseract(path: str, lang: str, whitelist: str = "0123456789") -> str:
    r = subprocess.run(["tesseract", path, "stdout", "-l", lang, "--oem", "1", "--psm", "7", "-c", f"tessedit_char_whitelist={whitelist}"],
                       capture_output=True, text=True, env={**os.environ, "TESSDATA_PREFIX": TESS_DIR})
    return "".join(r.stdout.split())


def tenths(v):
    return None if v in (None, "") else str(round(float(v) * 10))


def digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def effective(pred: str, truth: str) -> str:
    d = digits(pred)
    return d + "0" if d + "0" == truth else d


def main() -> None:
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    glm = cs.GlmClient(str(cs.HERE / "runs/full/merged"), 0)
    tmp = Path(tempfile.mkdtemp())

    def variant(path: Path, rescale: bool, pad: int = 0) -> str:
        if not rescale and not pad:
            return str(path)
        im = Image.open(path).convert("RGB")
        if rescale:
            im = im.resize((max(1, round(im.width * 128 / im.height)), 128), Image.BICUBIC)
        if pad:
            im = ImageOps.expand(im, pad, fill=(255, 255, 255))
        out = tmp / f"v_{abs(hash((str(path), rescale, pad)))}.png"
        im.save(out)
        return str(out)

    systems = {  # name -> (engine, source, rescale)
        "tess_V9_405 (production) | bin": ("tessV9", "bin", False),
        "tess_V9_405 | orig": ("tessV9", "orig", False),
        "tess_V6 | bin": ("tessV6", "bin", False),
        "tess_V6 | orig": ("tessV6", "orig", False),
        "tess_V6 padded | orig": ("tessV6pad", "orig", False),
        "easyocr | bin": ("easy", "bin", False),
        "easyocr | orig": ("easy", "orig", False),
        "GLM full | bin": ("glm", "bin", False),
        "GLM full | bin +128": ("glm", "bin", True),
        "GLM full | orig": ("glm", "orig", False),
        "GLM full | orig +128": ("glm", "orig", True),
    }

    def run(engine: str, p: str) -> str:
        if engine == "tessV9":
            return tesseract(p, "cobecore-V9_405")
        if engine == "tessV6":
            return tesseract(p, "cobecore-V6")
        if engine == "tessV6pad":
            return tesseract(variant(Path(p), False, 20), "cobecore-V6")
        if engine == "easy":
            return "".join(reader.readtext(p, detail=0, allowlist="0123456789"))
        return glm(p)

    rows_out = []
    per = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # system -> sheet -> [n, exact, edits, chars]
    for xlsx in sorted(MANUAL.glob("*/*_manually_entered_temperatures.xlsx")):
        f = xlsx.name.split("_manually")[0]
        root = cs.HERE / "sheet_runs" / f / "rois"
        if not (root / "cells.json").exists():
            print("missing ROIs for", f)
            continue
        m = openpyxl.load_workbook(xlsx).active
        cells = {tuple(v): int(k) for k, v in json.loads((root / "cells.json").read_text()).items()}
        for r in range(4, 47):
            for c in (4, 5, 6):
                truth = tenths(m.cell(r, c).value)
                if not truth or (r, c) not in cells:
                    continue
                i = cells[(r, c)]
                row = {"sheet": f, "row": r, "col": c, "truth": truth}
                for name, (engine, src, rescale) in systems.items():
                    p = str(root / src / f"{i}.png")
                    if rescale:
                        p = variant(Path(p), True)
                    pred = run(engine, p)
                    eff = effective(pred, truth)
                    row[name] = pred
                    s = per[name][f]
                    s[0] += 1; s[1] += eff == truth; s[2] += edit_distance(truth, eff); s[3] += len(truth)
                rows_out.append(row)
        print(f"{f}: {sum(1 for x in rows_out if x['sheet'] == f)} cells", flush=True)
    glm.close()

    with open(cs.HERE / "ocr_only_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)

    def agg(name, skip=()):
        t = [sum(v[k] for sh, v in per[name].items() if sh not in skip) for k in range(4)]
        return t[1] / t[0], t[2] / t[3], t[0]
    skip601 = ("601_196705_SF.JPG",)
    print(f"\n{'system':<34}{'exact':>8}{'CER':>8}   | {'exact w/o 601':>14}{'CER':>8}")
    for name in systems:
        e, c, n = agg(name); e2, c2, _ = agg(name, skip601)
        print(f"{name:<34}{e:>8.1%}{c:>8.3f}   | {e2:>14.1%}{c2:>8.3f}")
    print(f"cells scored: {n}")


if __name__ == "__main__":
    main()
