"""Batched GLM-OCR inference for short cell crops.

Replaces glm_ocr.ocr_image's one-generate()-per-crop loop: crops are sorted by width, grouped into
batches, and decoded together with left padding. `prefill_think=True` appends the assistant prefix the
fine-tune always emits (`<think></think>\\n`) to the prompt, so only the digits and EOS are decoded.

Run under BelHisFirm's .venv-glm python (GLM-OCR needs transformers>=5).
"""

from __future__ import annotations

import time
from typing import Sequence

import torch
from PIL import Image

from common import PROMPT, clean_prediction

THINK_PREFIX = "<think></think>\n"


def resize_to_height(img: Image.Image, h: int = 128) -> Image.Image:
    return img.resize((max(1, round(img.width * h / img.height)), h), Image.BICUBIC)


@torch.no_grad()
def ocr_images(model, processor, images: Sequence[Image.Image], *, batch_size: int = 32, max_new_tokens: int = 10,
               prefill_think: bool = True, prompt: str = PROMPT, stats: dict | None = None) -> list[str]:
    """Return cleaned predictions in the same order as `images`."""
    tok = processor.tokenizer
    tok.padding_side = "left"
    order = sorted(range(len(images)), key=lambda i: images[i].width)  # similar widths -> similar image-token counts
    out = [""] * len(images)
    t0 = time.monotonic()
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        msgs = [[{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": prompt}]}] for i in idx]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        if prefill_think:
            texts = [t + THINK_PREFIX for t in texts]
        inputs = processor(text=texts, images=[images[i] for i in idx], padding=True, return_tensors="pt").to(model.device)
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = gen[:, inputs["input_ids"].shape[1]:]
        for i, ids in zip(idx, new):
            out[i] = clean_prediction(processor.decode(ids, skip_special_tokens=True))
    if stats is not None:
        stats["seconds"] = time.monotonic() - t0
        stats["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out
