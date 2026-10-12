"""Throughput + parity benchmark for batched GLM-OCR on the test split (run with .venv-glm python).

Baseline = the current path (glm_ocr.ocr_image, batch size 1, 16 new tokens, no prefill).
Usage: .venv-glm/bin/python bench_glm.py [--n 600] [--model runs/full/merged] [--resize-h 0]
"""
from __future__ import annotations

import argparse
import time

import torch
from PIL import Image

from common import DATA_DIR, PROMPT, clean_prediction, load_split, read_gt, edit_distance
from glm_batch import ocr_images, resize_to_height
from glm_ocr import load_model, ocr_image


def cer(gts, preds):
    return sum(edit_distance(g, p) for g, p in zip(gts, preds)) / sum(len(g) for g in gts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--model", default="runs/full/merged")
    ap.add_argument("--resize-h", type=int, default=0)
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[8, 16, 32, 64])
    args = ap.parse_args()

    stems = load_split()["test"][: args.n]
    gts = [read_gt(DATA_DIR, s) for s in stems]
    imgs = [Image.open(DATA_DIR / f"{s}.png").convert("RGB") for s in stems]
    if args.resize_h:
        imgs = [resize_to_height(i, args.resize_h) for i in imgs]
    model, processor = load_model(args.model)
    ocr_images(model, processor, imgs[:16], batch_size=16)  # warm-up (CUDA kernels, caches)

    t0 = time.monotonic()
    base = [clean_prediction(ocr_image(model, processor, im, PROMPT, max_new_tokens=16)) for im in imgs]
    t_base = time.monotonic() - t0
    print(f"{'config':<34}{'s/cell':>9}{'cells/s':>9}{'speedup':>9}{'same as base':>14}{'CER':>8}{'peak GB':>9}")
    print(f"{'baseline bs=1, 16 tok':<34}{t_base / len(imgs):>9.4f}{len(imgs) / t_base:>9.1f}{1.0:>9.1f}{'-':>14}{cer(gts, base):>8.4f}{'-':>9}")
    for prefill in (False, True):
        for bs in args.batch_sizes:
            torch.cuda.reset_peak_memory_stats()
            st: dict = {}
            preds = ocr_images(model, processor, imgs, batch_size=bs, max_new_tokens=10, prefill_think=prefill, stats=st)
            same = sum(a == b for a, b in zip(preds, base)) / len(base)
            name = f"batch {bs}, 10 tok, prefill={'on' if prefill else 'off'}"
            print(f"{name:<34}{st['seconds'] / len(imgs):>9.4f}{len(imgs) / st['seconds']:>9.1f}{t_base / st['seconds']:>9.1f}{same:>14.1%}{cer(gts, preds):>8.4f}{st['peak_mem_gb']:>9.1f}", flush=True)
            if same < 1.0 and bs == args.batch_sizes[-1]:
                diffs = [(s, g, b, p) for s, g, b, p in zip(stems, gts, base, preds) if b != p][:8]
                for d in diffs:
                    print("    diff:", d)


if __name__ == "__main__":
    main()
