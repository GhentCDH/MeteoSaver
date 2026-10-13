"""Shared helpers for the GLM-OCR vs Tesseract cell-OCR experiment."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DATA_DIR = Path.home() / "Documents/Visual Code Data/tesseract"
GLM_DIR = Path.home() / (
    "Documents/Visual Code Repos/BelHisFirm---HisTableFinder/"
    "src/belhisfirm_histablefinder/transcribe_tables/GLM"
)
HERE = Path(__file__).resolve().parent
SPLIT_PATH = HERE / "split.json"

PROMPT = "Read the number in this image. Output only the text."

# glm_ocr.py / finetune_glm_ocr.py live in the BelHisFirm repo; import them rather than copy.
if str(GLM_DIR) not in sys.path:
    sys.path.insert(0, str(GLM_DIR))


def read_gt(data_dir: Path, stem: str) -> str:
    return (data_dir / f"{stem}.gt.txt").read_text(encoding="utf-8").strip()


def load_split(path: Path = SPLIT_PATH) -> dict[str, list[str]]:
    return json.loads(path.read_text())


_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean_prediction(text: str) -> str:
    """Drop <think></think> leakage (seen in an earlier GLM-OCR fine-tune), HTML and whitespace."""
    text = _THINK.sub("", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", "", text)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

#hklejhgskljehjgspogikesgkmoskjeg