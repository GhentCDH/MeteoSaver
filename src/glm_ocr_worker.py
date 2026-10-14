#!/usr/bin/env python
"""
GLM-OCR worker process: keeps the model in memory and reads batches of cell images on request.

This script is started by glm_ocr_client.py (see GlmOcrClient) with the Python interpreter set as `glm_python` in the
configuration file. It runs in its own process because GLM-OCR needs transformers>=5.3 and torch>=2.10, which
may not be installable next to the other MeteoSaver requirements. It only needs torch, transformers and pillow.

Protocol (one JSON object per line):
    stdout, once the model is loaded : {"ready": true}
    stdin                            : ["path/to/cell_0.png", "path/to/cell_1.png", ...]
    stdout                           : {"texts": ["20,2", "18,9", ...]}   (same order as the request)
"""
import argparse
import json
import re
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

# The prompt the model was fine-tuned with on the labelled cell images
PROMPT = "Read the number in this image. Output only the text."

# The fine-tuned model always starts its answer with this prefix. Appending it to the prompt means only the digits are decoded
THINK_PREFIX = "<think></think>\n"


def load_model(model):
    '''
    Loads the GLM-OCR model and its processor, on the GPU if one is available.
    model is a Hugging Face model ID: it is downloaded on first use, and then read from the Hugging Face cache.
    '''
    processor = AutoProcessor.from_pretrained(model)
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        glm_model = GlmOcrForConditionalGeneration.from_pretrained(model, dtype=dtype).eval().cuda()
    else:
        glm_model = GlmOcrForConditionalGeneration.from_pretrained(model).eval()
    return glm_model, processor


def clean_prediction(text):
    '''
    Removes <think></think> tags, any other markup and whitespace from the model output.
    '''
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", "", text)


@torch.no_grad()
def read_images(model, processor, images, batch_size, max_new_tokens=10):
    '''
    Reads a list of cell images in batches and returns the recognized text for each, in the same order.
    '''
    processor.tokenizer.padding_side = "left"
    order = sorted(range(len(images)), key=lambda i: images[i].width)  # Similar widths in a batch means less padding
    texts = [""] * len(images)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        prompts = []
        for i in batch:
            messages = [{"role": "user", "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": PROMPT}]}]
            prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + THINK_PREFIX)
        inputs = processor(text=prompts, images=[images[i] for i in batch], padding=True, return_tensors="pt").to(model.device)
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        for i, ids in zip(batch, generated[:, inputs["input_ids"].shape[1]:]):
            texts[i] = clean_prediction(processor.decode(ids, skip_special_tokens=True))
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID of the (fine-tuned) GLM-OCR model, e.g. GhentCDH/GLM-OCR-MeteoSaver")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resize-height", type=int, default=128, help="Rescale the cell images to this height in pixels (0 = keep as is)")
    args = parser.parse_args()

    model, processor = load_model(args.model)
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        images = [Image.open(path).convert("RGB") for path in json.loads(line)]
        if args.resize_height:
            images = [image.resize((max(1, round(image.width * args.resize_height / image.height)), args.resize_height), Image.BICUBIC) for image in images]
        print(json.dumps({"texts": read_images(model, processor, images, args.batch_size)}), flush=True)


if __name__ == "__main__":
    main()
