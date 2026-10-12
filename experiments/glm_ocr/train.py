"""LoRA fine-tune GLM-OCR on tesseract-cell crops (run with BelHisFirm's .venv-glm python).

Reuses GlmOcrDataCollator from BelHisFirm's finetune_glm_ocr.py, but with a lazy dataset
(images are read on demand) and the `X.png` + `X.gt.txt` layout. One epoch over the first
--n-train stems of split.json's train list; batch size 1 with --grad-accum.

Usage:
    .venv-glm/bin/python train.py --n-train 2000 --name n2000
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from PIL import Image

from common import DATA_DIR, HERE, PROMPT, clean_prediction, edit_distance, load_split, read_gt


class LazyCells(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, stems: list[str]):
        self.data_dir, self.stems = data_dir, stems

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, i: int) -> dict:
        s = self.stems[i]
        return {"image": Image.open(self.data_dir / f"{s}.png").convert("RGB"), "text": read_gt(self.data_dir, s)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--n-train", type=int, required=True, help="Use the first N train stems (0 = all)")
    ap.add_argument("--name", required=True, help="Run name -> runs/<name>/")
    ap.add_argument("--max-steps", type=int, default=-1, help="Override the 1-epoch step count (smoke tests)")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--val-limit", type=int, default=300, help="Val cells scored after training")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration, Trainer, TrainingArguments

    from finetune_glm_ocr import GlmOcrDataCollator
    from glm_ocr import MODEL_ID, ocr_image

    split = load_split()
    stems = split["train"][: args.n_train] if args.n_train else split["train"]
    out = HERE / "runs" / args.name
    steps = args.max_steps if args.max_steps > 0 else math.ceil(len(stems) / args.grad_accum)
    print(f"{len(stems)} train cells -> {steps} optimizer steps", flush=True)

    bf16 = torch.cuda.is_bf16_supported()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = GlmOcrForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16 if bf16 else torch.float16).cuda()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        data_collator=GlmOcrDataCollator(processor, prompt=PROMPT),
        train_dataset=LazyCells(args.data, stems),
        args=TrainingArguments(
            per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
            max_steps=steps, warmup_steps=max(1, steps // 20), learning_rate=args.lr,
            lr_scheduler_type="linear", weight_decay=0.001, optim="adamw_torch", seed=args.seed,
            logging_steps=10, bf16=bf16, fp16=not bf16, output_dir=str(out), report_to="none",
            remove_unused_columns=False, gradient_checkpointing=True, save_strategy="no",
        ),
    )
    t0 = time.monotonic()
    trainer.train()
    print(f"train time {(time.monotonic() - t0) / 60:.1f} min", flush=True)

    merged = model.merge_and_unload().eval()
    merged.save_pretrained(str(out / "merged"))
    processor.save_pretrained(str(out / "merged"))
    print(f"saved {out / 'merged'}")

    # Quick val CER (full comparison vs Tesseract happens in evaluate.py on the test split).
    edits = total = 0
    for s in split["val"][: args.val_limit]:
        pred = clean_prediction(ocr_image(merged, processor, Image.open(args.data / f"{s}.png").convert("RGB"), PROMPT, max_new_tokens=16))
        gt = read_gt(args.data, s)
        edits += edit_distance(gt, pred)
        total += len(gt)
    print(f"val CER ({min(args.val_limit, len(split['val']))} cells) = {edits / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
