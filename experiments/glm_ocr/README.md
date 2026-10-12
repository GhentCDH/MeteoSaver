# GLM-OCR vs Tesseract on MeteoSaver cell crops

Tests whether fine-tuning GLM-OCR (0.9B VLM) on the labelled cell crops in
`~/Documents/Visual Code Data/tesseract` beats MeteoSaver's Tesseract setup.

Everything runs with the BelHisFirm GLM venv (transformers>=5 is needed for GLM-OCR):

```
PY="$HOME/Documents/Visual Code Repos/BelHisFirm---HisTableFinder/.venv-glm/bin/python"
cd experiments/glm_ocr
```

Do not pass `-I` to these scripts: it strips the script directory from `sys.path`, so `import common` fails.

## 1. Split (seeded, stems only; writes `split.json`)
```
$PY make_split.py
```
32,107 train / 714 val / 2,854 test. Note that 99.2% of test labels also occur in train
(labels are short numerics), so there is no label-level held-out. Sheet-level leakage
cannot be checked because stems carry no source-sheet id.

## 2. Train (LoRA, one epoch over the first N train cells)
```
$PY train.py --n-train 500   --name n500
$PY train.py --n-train 2000  --name n2000
$PY train.py --n-train 8000  --name n8000
$PY train.py --n-train 0     --name full      # all train cells
```
Writes `runs/<name>/merged` and prints a quick val CER.

## 3. Evaluate on the test split
```
$PY evaluate.py --glm zeroshot=zai-org/GLM-OCR --glm n2000=runs/n2000/merged ...
```
Tesseract systems are scored with all three models in `MeteoSaver/OCR_HTR_models`,
each with MeteoSaver's digit whitelist (`digits`) and with `,-=` added (`punct`).
Outputs `results.csv` (per cell) and `summary.json`; the console shows CER, exact
match, empty rate, seconds per cell, and the worst errors per system.
Use `--limit N` for a quick check and `--no-tesseract` to skip the baseline.

## Caveats
- `cobecore-V9_405` (the model `configuration.ini` / `transcription.py` use) returns empty
  or wrong output on most of these 128-px-high crops, while `cobecore-V6` reads most of them.
  Check which model the crops were meant for before reading the baseline as Tesseract's best.
- GLM-OCR zero-shot returns `.` where labels use `,`; fine-tuning is expected to fix this.
- A previous fine-tune leaked `<think></think>` into outputs; `evaluate.py` strips it and
  reports how many raw outputs had it.
