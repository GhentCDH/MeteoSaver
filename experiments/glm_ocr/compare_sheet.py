"""Run MeteoSaver's real cell detection + transcription() on one sheet twice -- once with the
shipped OCR (Tesseract, EasyOCR fallback) and once with GLM-OCR swapped in for the Tesseract
call -- and compare the two pre-QA/QC workbooks cell by cell.

Nothing in MeteoSaver/src is modified: the GLM run monkeypatches `pytesseract.image_to_string`
(the single OCR call in transcription.py) to ask a GLM worker process about the same ROI
image file. Detection, cell ordering, Top/Midpoint merge and workbook layout are identical, and
the comparison is taken on `<file>_pre_QA_QC.xlsx`, i.e. right before QA/QC.

Run with experiments/glm_ocr/.venv-easyocr/bin/python (has the MeteoSaver deps).

Usage:
    .venv-easyocr/bin/python compare_sheet.py --station 508 --file 508_198802_SF.JPG
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

GLM_PY = Path.home() / "Documents/Visual Code Repos/BelHisFirm---HisTableFinder/.venv-glm/bin/python"


class GlmClient:
    def __init__(self, model: str, resize_h: int):
        self.proc = subprocess.Popen(
            [str(GLM_PY), "-u", str(HERE / "glm_worker.py"), "--model", model, "--resize-h", str(resize_h)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=HERE, bufsize=1,
        )
        ready = self.proc.stdout.readline()
        assert json.loads(ready).get("ready"), ready
        self.log: list[dict] = []

    def __call__(self, image_path, *args, **kwargs) -> str:
        self.proc.stdin.write(str(image_path) + "\n")
        self.proc.stdin.flush()
        out = json.loads(self.proc.stdout.readline())
        self.log.append(out)
        return out["text"]

    def batch(self, paths: list[str]) -> tuple[list[str], float]:
        self.proc.stdin.write(json.dumps(paths) + "\n")
        self.proc.stdin.flush()
        out = json.loads(self.proc.stdout.readline())
        return out["texts"], out["seconds"]

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=30)


class Collector:
    """OCR stand-in for the collect pass: remember each distinct ROI image (by content hash), return a dummy."""

    def __init__(self, dest: Path):
        self.dest, self.paths, self.calls = dest, {}, 0
        dest.mkdir(parents=True, exist_ok=True)

    def __call__(self, image_path, *a, **k) -> str:
        import hashlib
        data = Path(image_path).read_bytes()
        key = hashlib.md5(data).hexdigest()
        if key not in self.paths:
            p = self.dest / f"{key}.png"
            p.write_bytes(data)
            self.paths[key] = str(p)
        self.calls += 1
        return "0"  # non-empty, so transcription() does not fall back to EasyOCR


class Replay:
    """OCR stand-in for the replay pass: look the text up by ROI content hash."""

    def __init__(self, table: dict[str, str]):
        self.table = table

    def __call__(self, image_path, *a, **k) -> str:
        import hashlib
        return self.table[hashlib.md5(Path(image_path).read_bytes()).hexdigest()]


def load_transcription(hoist_borders: bool):
    """Return the transcription module. With hoist_borders, load a runtime-transformed copy of src/transcription.py in
    which the whole-sheet border loop (re-run for every OCR'd cell, ~2M openpyxl style assignments per sheet)
    runs once per workbook instead. src/ is not modified."""
    import importlib.util
    import re
    if not hoist_borders:
        import transcription as tr
        return tr
    src = (REPO / "src" / "transcription.py").read_text(encoding="utf-8")
    inner = re.compile(
        r"\n( {24})# Loop through cells to apply borders\n"
        r" {24}for row in ws\.iter_rows\(min_row=1, max_row=no_of_rows, min_col=1, max_col=no_of_columns\):\n"
        r" {28}for cell in row:\n"
        r" {32}cell\.border = thin_border\n")
    src, n = inner.subn("\n", src)
    assert n == 1, f"expected exactly one per-cell border loop, found {n}"
    marker = "        # Insert two columns on the left side of the excel sheet\n"
    assert src.count(marker) == 1
    src = src.replace(marker, (
        "        _tb = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))\n"
        "        for _row in ws.iter_rows(min_row=1, max_row=no_of_rows, min_col=1, max_col=no_of_columns):\n"
        "            for _cell in _row:\n"
        "                _cell.border = _tb\n") + marker)
    spec = importlib.util.spec_from_loader("transcription_hoisted", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(REPO / "src" / "transcription.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def run_pipeline(label: str, args, cfg, patch_ocr=None, roi_source: str = "binarized", row_grouping: str = "kmeans", hoist_borders: bool = False) -> Path:
    import pytesseract
    import image_preprocessing_module as prep
    import table_and_cell_detection_model as det
    tr = load_transcription(hoist_borders)

    out = HERE / "sheet_runs" / args.file / label
    transient, pre_qaqc = out / "transient", out / "pre_QA_QC"
    (transient / args.station).mkdir(parents=True, exist_ok=True)
    pre_qaqc.mkdir(parents=True, exist_ok=True)
    for old in (transient / args.station).glob("*.zip"):  # transcription() appends to its zips
        old.unlink()

    image_path = REPO / cfg["Directories"]["full_datadir"] / args.station / args.file
    gray, binar, orig = prep.image_preprocessing(str(image_path))
    t = cfg["TableAndCellDetection"]
    detected = det.table_and_cell_detection(
        gray, binar, orig, args.station, args.file, str(transient),
        clip_up=int(t["clip_up"]), clip_down=int(t["clip_down"]), clip_left=int(t["clip_left"]), clip_right=int(t["clip_right"]),
        max_table_width=int(t["max_table_width"]), max_table_height=int(t["max_table_height"]),
        min_cell_width_threshold=int(t["min_cell_width_threshold"]), max_cell_width_threshold=int(t["max_cell_width_threshold"]),
        min_cell_height_threshold=int(t["min_cell_height_threshold"]), max_cell_height_threshold=int(t["max_cell_height_threshold"]),
        space_height_threshold=int(t["space_height_threshold"]), space_width_threshold=int(t["space_width_threshold"]),
        max_cell_height_per_box=int(t["max_cell_height_per_box"]), no_of_rows=int(t["no_of_rows"]), no_of_columns=int(t["no_of_columns"]),
    )

    if roi_source == "original":
        # transcription() cuts its ROIs from detected[2] (binarized, lines/dots removed). Swap in the
        # un-binarized clip of the same table (detected[3]) so ROIs have the same coordinates.
        assert detected[3].shape[:2] == detected[2].shape[:2], (detected[3].shape, detected[2].shape)
        detected = list(detected)
        detected[2] = detected[3].copy()

    saved_org = (tr.organize_contours_top, tr.organize_contours_midpoint)
    if row_grouping == "columnwise":  # experiment: replace the KMeans row grouping (see row_grouping.py)
        from row_grouping import column_wise_rows
        grouping = lambda cnts, filename: column_wise_rows(cnts, tr.get_max_rows_from_filename(filename))
        tr.organize_contours_top = tr.organize_contours_midpoint = grouping

    original = pytesseract.image_to_string
    if patch_ocr:
        pytesseract.image_to_string = patch_ocr
    t0 = time.monotonic()
    try:
        tr.transcription(
            detected, "Tesseract-OCR", "tesseract", str(transient), str(pre_qaqc), args.station, args.file,
            no_of_rows=int(t["no_of_rows"]), no_of_columns=int(t["no_of_columns"]),
            no_of_rows_including_headers=int(t["no_of_rows_including_headers"]),
        )
    finally:
        pytesseract.image_to_string = original
        tr.organize_contours_top, tr.organize_contours_midpoint = saved_org
    print(f"[{label}] transcription took {time.monotonic() - t0:.0f}s", flush=True)
    return pre_qaqc / f"{args.file}_pre_QA_QC.xlsx"


def read_grid(path: Path) -> dict[tuple[int, int], str]:
    import openpyxl
    ws = openpyxl.load_workbook(path).active
    return {(c.row, c.column): ("" if c.value is None else str(c.value).strip()) for row in ws.iter_rows() for c in row}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True)
    ap.add_argument("--file", required=True, help="Sheet image file name inside the station folder")
    ap.add_argument("--model", default=str(HERE / "runs/full/merged"))
    ap.add_argument("--resize-h", type=int, default=0, help="Rescale ROIs to this height before GLM (0 = as detected)")
    ap.add_argument("--roi-source", choices=["binarized", "original"], default="binarized",
                    help="Cut GLM's ROIs from the binarized table (what MeteoSaver feeds Tesseract) or from the original colour clip")
    ap.add_argument("--row-grouping", choices=["kmeans", "columnwise"], default="kmeans")
    ap.add_argument("--label", default="glm", help="Output sub-folder name for the GLM run")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--hoist-borders", action="store_true", help="Run the whole-sheet border loop once instead of per cell (identical output, ~7x faster pipeline)")
    ap.add_argument("--fast-glm", action="store_true", help="Collect all ROIs, run GLM once in batches, then replay (same output, much faster)")
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(REPO / "configuration.ini")
    os.environ["TESSDATA_PREFIX"] = str(REPO / "OCR_HTR_models")  # repo's own models, not the Windows path in the ini

    base = HERE / "sheet_runs" / args.file / "baseline" / "pre_QA_QC" / f"{args.file}_pre_QA_QC.xlsx"
    if not args.skip_baseline:
        base = run_pipeline("baseline", args, cfg)
    client = GlmClient(args.model, args.resize_h)
    try:
        if args.fast_glm:
            t0 = time.monotonic()
            col = Collector(HERE / "sheet_runs" / args.file / f"{args.label}_rois")
            run_pipeline(f"{args.label}_collect", args, cfg, patch_ocr=col, roi_source=args.roi_source, row_grouping=args.row_grouping, hoist_borders=args.hoist_borders)
            t_collect = time.monotonic() - t0
            keys = list(col.paths)
            texts, t_glm = client.batch([col.paths[k] for k in keys])
            t1 = time.monotonic()
            glm = run_pipeline(args.label, args, cfg, patch_ocr=Replay(dict(zip(keys, texts))), roi_source=args.roi_source, row_grouping=args.row_grouping, hoist_borders=args.hoist_borders)
            t_replay = time.monotonic() - t1
            print(f"fast-glm: {col.calls} OCR calls -> {len(keys)} unique ROIs | collect {t_collect:.0f}s | GLM batch {t_glm:.1f}s | replay {t_replay:.0f}s")
            client.log = [{"text": t} for t in texts]
        else:
            glm = run_pipeline(args.label, args, cfg, patch_ocr=client, roi_source=args.roi_source, row_grouping=args.row_grouping, hoist_borders=args.hoist_borders)
    finally:
        client.close()
    (glm.parent.parent / "glm_calls.json").write_text(json.dumps(client.log))
    print(f"baseline: {base}\nglm:      {glm}\nGLM calls: {len(client.log)}")


if __name__ == "__main__":
    main()
