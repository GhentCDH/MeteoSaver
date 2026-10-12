"""Score Tesseract (MeteoSaver's exact config) and GLM-OCR checkpoints on the test split.

Run with BelHisFirm's .venv-glm python (GLM needs transformers>=5; Tesseract is called via CLI).

Usage:
    .venv-glm/bin/python evaluate.py --limit 20 \
        --glm zeroshot=zai-org/GLM-OCR --glm n2000=runs/n2000/merged
    .venv-glm/bin/python evaluate.py --no-tesseract --glm n2000=runs/n2000/merged   # (re)score GLM only
Writes results.csv + summary.json next to this script (merging with earlier columns is not done; list all systems in one run).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image

from common import DATA_DIR, HERE, PROMPT, clean_prediction, edit_distance, load_split, read_gt

MODEL_DIR = Path.home() / "Documents/Visual Code Repos/MeteoSaver/OCR_HTR_models"


def tesseract(image: Path, whitelist: str, lang: str, pad: int = 0) -> str:
    if pad:  # white border: raw crops have glyphs touching the edge, which makes Tesseract return nothing
        from PIL import ImageOps
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            ImageOps.expand(Image.open(image).convert("L"), pad, fill=255).save(tmp.name)
            return tesseract(Path(tmp.name), whitelist, lang)
    r = subprocess.run(
        ["tesseract", str(image), "stdout", "-l", lang, "--oem", "1", "--psm", "7", "-c", f"tessedit_char_whitelist={whitelist}"],
        capture_output=True, text=True, env={**os.environ, "TESSDATA_PREFIX": str(MODEL_DIR)},
    )
    return r.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-tesseract", action="store_true")
    ap.add_argument("--tess-lang", action="append", help="Tesseract model(s) in OCR_HTR_models (default: V6, V8_501, V9_405)")
    ap.add_argument("--easyocr-preds", type=Path, help="easyocr_preds.json from easyocr_predict.py (adds EasyOCR and Tesseract->EasyOCR fallback systems)")
    ap.add_argument("--glm", action="append", default=[], metavar="NAME=MODEL_ID_OR_DIR")
    args = ap.parse_args()

    stems = load_split()["test"][: args.limit]
    gts = [read_gt(args.data, s) for s in stems]
    preds: dict[str, list[str]] = {}
    latency: dict[str, float] = {}

    if not args.no_tesseract:
        # "digits" = MeteoSaver's exact whitelist; "punct" also allows the , - = that occur in the labels.
        for lang in args.tess_lang or ["cobecore-V6", "cobecore-V8_501", "cobecore-V9_405"]:
            for kind, wl, pad in [("digits", "0123456789", 0), ("punct", "0123456789,-=", 0), ("punct_pad", "0123456789,-=", 20)]:
                name = f"tess_{lang.replace('cobecore-', '')}_{kind}"
                t0 = time.monotonic()
                preds[name] = [clean_prediction(tesseract(args.data / f"{s}.png", wl, lang, pad)) for s in stems]
                latency[name] = (time.monotonic() - t0) / len(stems)
                print(f"{name}: done", flush=True)

    if args.easyocr_preds:
        eo = json.loads(args.easyocr_preds.read_text())
        for name in ("easyocr_digits", "easyocr_punct"):
            preds[name] = [clean_prediction(eo[name].get(s, "")) for s in stems]
            latency[name] = eo.get("_sec_per_cell", float("nan"))
        # MeteoSaver's actual chain: configured Tesseract model, EasyOCR only when Tesseract returns nothing.
        tess = next((n for n in preds if n.startswith("tess_") and n.endswith("_digits") and "V9_405" in n), None)
        if tess:
            preds["meteosaver_chain"] = [t or e for t, e in zip(preds[tess], preds["easyocr_digits"])]
            latency["meteosaver_chain"] = latency[tess] + latency["easyocr_digits"]

    if args.glm:
        from glm_ocr import load_model, ocr_image
        for spec in args.glm:
            name, _, path = spec.partition("=")
            path = str(Path(path).resolve()) if Path(path).exists() else path
            model, processor = load_model(path)
            t0 = time.monotonic()
            raw = [ocr_image(model, processor, Image.open(args.data / f"{s}.png").convert("RGB"), PROMPT, max_new_tokens=16) for s in stems]
            latency[name] = (time.monotonic() - t0) / len(stems)
            preds[name] = [clean_prediction(r) for r in raw]
            leaked = sum("<think>" in r for r in raw)
            print(f"{name}: done ({leaked} outputs had <think> leakage, stripped)", flush=True)
            del model
            import torch
            torch.cuda.empty_cache()

    summary = {}
    for name, ps in preds.items():
        edits = [edit_distance(g, p) for g, p in zip(gts, ps)]
        summary[name] = {
            "cer": sum(edits) / sum(len(g) for g in gts),
            "exact_match": sum(e == 0 for e in edits) / len(gts),
            "empty_rate": sum(not p for p in ps) / len(gts),
            "sec_per_cell": latency[name],
            "exact_match_with_comma": sum(p == g for p, g in zip(ps, gts) if "," in g) / max(sum("," in g for g in gts), 1),
        }
    with open(HERE / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "gt", *preds])
        for i, s in enumerate(stems):
            w.writerow([s, gts[i], *(preds[n][i] for n in preds)])
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{len(stems)} test cells")
    print(f"{'system':<20}{'CER':>8}{'exact':>8}{'empty':>8}{'s/cell':>9}{'exact(,)':>10}")
    for n, m in summary.items():
        print(f"{n:<20}{m['cer']:>8.4f}{m['exact_match']:>8.3f}{m['empty_rate']:>8.3f}{m['sec_per_cell']:>9.3f}{m['exact_match_with_comma']:>10.3f}")
    for n, ps in preds.items():
        worst = sorted(range(len(stems)), key=lambda i: -edit_distance(gts[i], ps[i]))[:10]
        print(f"\nworst 10 for {n}:")
        for i in worst:
            print(f"  {stems[i]}: gt={gts[i]!r} pred={ps[i]!r}")


if __name__ == "__main__":
    main()
