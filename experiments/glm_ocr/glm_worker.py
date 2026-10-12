"""Persistent GLM-OCR worker: reads image paths on stdin, writes one JSON line per image on stdout.

Run with BelHisFirm's .venv-glm python (GLM-OCR needs transformers>=5, which the MeteoSaver
modules' environment does not have). compare_sheet.py spawns this itself.
"""

from __future__ import annotations

import argparse
import json
import sys

from PIL import Image

from common import PROMPT, clean_prediction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--resize-h", type=int, default=0, help="Rescale crops to this height (training crops are 128 px high); 0 = as is")
    args = ap.parse_args()

    from glm_batch import ocr_images, resize_to_height
    from glm_ocr import load_model, ocr_image

    model, processor = load_model(args.model)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        if path.startswith("["):  # batched request: JSON list of image paths -> one JSON line with all texts
            import time
            t0 = time.monotonic()
            imgs = [Image.open(q).convert("RGB") for q in json.loads(path)]
            if args.resize_h:
                imgs = [resize_to_height(i, args.resize_h) for i in imgs]
            texts = ocr_images(model, processor, imgs, batch_size=args.batch_size, max_new_tokens=10, prefill_think=True)
            print(json.dumps({"texts": texts, "seconds": time.monotonic() - t0}), flush=True)
            continue
        try:
            img = Image.open(path).convert("RGB")
            if args.resize_h:
                img = img.resize((max(1, round(img.width * args.resize_h / img.height)), args.resize_h), Image.BICUBIC)
            raw = ocr_image(model, processor, img, PROMPT, max_new_tokens=16)
            out = {"raw": raw, "text": clean_prediction(raw)}
        except Exception as e:  # keep the pipeline going; surface as empty text
            out = {"raw": "", "text": "", "error": repr(e)}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
