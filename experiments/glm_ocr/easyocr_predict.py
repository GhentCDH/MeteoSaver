"""EasyOCR predictions for the test split, as MeteoSaver calls it
(`Reader(['en']).readtext(path, detail=0, allowlist='0123456789')`, results joined).

Run with .venv-easyocr/bin/python (kept separate from .venv-glm). Writes easyocr_preds.json
({stem: prediction}) for evaluate.py's --extra-preds.

Usage: .venv-easyocr/bin/python easyocr_predict.py [--limit N] [--gpu]
"""

from __future__ import annotations

import argparse
import json
import time

import easyocr

from common import DATA_DIR, HERE, load_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gpu", action="store_true", help="CPU by default (venv ships CPU torch)")
    args = ap.parse_args()

    stems = load_split()["test"][: args.limit]
    reader = easyocr.Reader(["en"], gpu=args.gpu)
    out: dict[str, dict[str, str]] = {"easyocr_digits": {}, "easyocr_punct": {}}
    t0 = time.monotonic()
    for i, s in enumerate(stems, 1):
        p = str(DATA_DIR / f"{s}.png")
        out["easyocr_digits"][s] = "".join(reader.readtext(p, detail=0, allowlist="0123456789"))
        out["easyocr_punct"][s] = "".join(reader.readtext(p, detail=0, allowlist="0123456789,-="))
        if i % 200 == 0:
            print(f"{i}/{len(stems)}", flush=True)
    out["_sec_per_cell"] = (time.monotonic() - t0) / (2 * len(stems))
    (HERE / "easyocr_preds.json").write_text(json.dumps(out))
    print(f"wrote easyocr_preds.json ({out['_sec_per_cell']:.3f} s/cell/variant)")


if __name__ == "__main__":
    main()
